#!/usr/bin/env python3
"""Qualification avec de VRAIS modèles d'une AI Gateway Open.

À exécuter contre une Gateway lancée devant un Ollama et un TEI ISOLÉS
(boucle locale, aucun GPU partagé). Le script ne télécharge rien, ne modifie
aucun service ; il refuse une cible hors boucle locale. Il produit un fichier
JSON de preuves : versions, digests observés, latences, réponses tronquées
(aucun prompt long, pas de données réelles — textes de sonde synthétiques).

Épreuves :
  1. /health/live et /health/runtime SANS inférence : Ollama /api/ps doit
     rester vide après ces appels (preuve non inférentielle côté runtime) ;
  2. catalogue /v1/models ;
  3. génération réelle (/v1/generate) sur le modèle Compact installé ;
  4. génération réelle en flux NDJSON (delta* → done) ;
  5. embeddings réels avec identité vérifiée (digest observé = attendu,
     pipeline_id, norme) ;
  6. reranking réel (TEI) avec identité /info vérifiée par /health/rerank ;
  7. négatifs : modèle enregistré NON installé => 503 model_not_installed ;
     modèle hors registre => 400 model_not_registered ; X-Request-ID relayé.

Usage : python tools/open_live_qualification.py --gateway http://127.0.0.1:28100
          --ollama http://127.0.0.1:21434 --tei http://127.0.0.1:28300
          --generate-model compact-generation-8 --missing-model compact-generation
          --out evidence.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from urllib.parse import urlparse

import requests


def loopback(url: str) -> bool:
    return urlparse(url).hostname in ("127.0.0.1", "localhost", "::1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", required=True)
    ap.add_argument("--ollama", required=True)
    ap.add_argument("--tei", required=True)
    ap.add_argument("--generate-model", default="compact-generation-8")
    ap.add_argument("--missing-model", default="compact-generation")
    ap.add_argument("--embedding-model", default="fr-en-retrieval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for url in (a.gateway, a.ollama, a.tei):
        if not loopback(url):
            print(f"REFUS : cible hors boucle locale : {url}", file=sys.stderr)
            return 2
    G = a.gateway.rstrip("/")
    ev: dict = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "targets": {
        "gateway": G, "ollama": a.ollama, "tei": a.tei}, "checks": {}}
    verdicts: dict = {}

    def check(name, ok, **details):
        verdicts[name] = "PASS" if ok else "FAIL"
        ev["checks"][name] = {"verdict": verdicts[name], **details}
        print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(details, ensure_ascii=False)[:300]}")

    # Versions des runtimes
    ev["ollama_version"] = requests.get(f"{a.ollama}/api/version", timeout=5).json()
    tei_info = requests.get(f"{a.tei}/info", timeout=5).json()
    ev["tei_info"] = {k: tei_info.get(k) for k in ("model_id", "model_sha", "model_type", "version",
                                                    "max_client_batch_size", "max_batch_tokens")}
    tags = requests.get(f"{a.ollama}/api/tags", timeout=5).json()["models"]
    ev["ollama_installed"] = [{k: m.get(k) for k in ("name", "digest", "size")} | {"details": m.get("details")}
                              for m in tags]

    # 1. Sondes non inférentielles : /api/ps vide avant et après.
    ps_before = requests.get(f"{a.ollama}/api/ps", timeout=5).json().get("models", [])
    t = time.time(); live = requests.get(f"{G}/health/live", timeout=5); live_ms = int((time.time() - t) * 1000)
    t = time.time(); runtime = requests.get(f"{G}/health/runtime", timeout=5); runtime_ms = int((time.time() - t) * 1000)
    models = requests.get(f"{G}/v1/models", timeout=10)
    ps_after = requests.get(f"{a.ollama}/api/ps", timeout=5).json().get("models", [])
    check("health_live_non_inferential", live.status_code == 200 and live.json()["inferential"] is False
          and ps_before == [] and ps_after == [], status=live.status_code, latency_ms=live_ms,
          body=live.json(), loaded_models_after=[m.get("name") for m in ps_after])
    rt = runtime.json()
    check("health_runtime_non_inferential", runtime.status_code == 200 and ps_after == [],
          latency_ms=runtime_ms, capabilities={k: v.get("state") for k, v in rt["capabilities"].items() if k != "inventory"},
          inventory=rt["capabilities"]["inventory"], limits=rt["limits"],
          generation_models=rt["capabilities"]["generation"]["models"],
          embeddings_models=rt["capabilities"]["embeddings"]["models"])
    # 2. Catalogue
    cat = models.json()
    check("catalog", models.status_code == 200 and any(m.get("model_logical") == a.embedding_model for m in cat["models"]),
          models=[{k: m.get(k) for k in ("name", "model_logical", "kind", "digest", "registry_status")} for m in cat["models"]])

    # 3. Génération réelle
    rid = "open3-qualif-001"
    t = time.time()
    g = requests.post(f"{G}/v1/generate", json={"prompt": "Réponds en un mot : quelle est la couleur du ciel par beau temps ?",
                                                "model": a.generate_model, "options": {"num_predict": 16},
                                                "temperature": 0.0}, headers={"X-Request-ID": rid}, timeout=600)
    gen_ms = int((time.time() - t) * 1000)
    gb = g.json() if g.headers.get("content-type", "").startswith("application/json") else {}
    check("generation", g.status_code == 200 and bool(gb.get("text", "").strip()) and gb.get("done") is True
          and g.headers.get("X-Request-ID") == rid,
          status=g.status_code, latency_ms=gen_ms, model=gb.get("model"), done_reason=gb.get("done_reason"),
          tokens=gb.get("tokens"), text_preview=(gb.get("text") or "")[:80], request_id_echo=g.headers.get("X-Request-ID"))

    # 4. Flux NDJSON
    t = time.time(); events = []; first_ms = None
    with requests.post(f"{G}/v1/generate/stream", json={"prompt": "Compte de 1 à 5 séparés par des virgules.",
                                                        "model": a.generate_model, "options": {"num_predict": 24},
                                                        "temperature": 0.0}, stream=True, timeout=600) as s:
        ctype = s.headers.get("content-type", "")
        for line in s.iter_lines():
            if not line:
                continue
            if first_ms is None:
                first_ms = int((time.time() - t) * 1000)
            events.append(json.loads(line))
    types = [e["type"] for e in events]
    check("generation_stream", ctype.startswith("application/x-ndjson") and types and types[-1] == "done"
          and all(x == "delta" for x in types[:-1]) and types.count("done") == 1,
          content_type=ctype, events=len(events), first_event_ms=first_ms, total_ms=int((time.time() - t) * 1000),
          done=events[-1] if events else None, text_preview="".join(e.get("delta", "") for e in events)[:80])

    # 5. Embeddings réels + identité
    t = time.time()
    e = requests.post(f"{G}/v1/embeddings", json={"texts": ["Le contrat est signé.", "The contract is signed."],
                                                  "model": a.embedding_model, "task": "document"}, timeout=300)
    emb_ms = int((time.time() - t) * 1000)
    eb = e.json()
    vec = eb.get("embeddings", [[]])[0]
    norm = math.sqrt(sum(x * x for x in vec)) if vec else None
    check("embeddings_identity", e.status_code == 200 and eb.get("identity_verified") is True
          and eb.get("expected_digest") == eb.get("model_digest") and eb.get("dimension") == 768
          and str(eb.get("pipeline_id", "")).startswith("emb-v2:") and norm is not None and abs(norm - 1.0) < 1e-3,
          status=e.status_code, latency_ms=emb_ms, model=eb.get("model"), model_logical=eb.get("model_logical"),
          dimension=eb.get("dimension"), pipeline_id=eb.get("pipeline_id"), expected_digest=eb.get("expected_digest"),
          model_digest=eb.get("model_digest"), identity_verified=eb.get("identity_verified"),
          verified_at=eb.get("verified_at"), normalized=eb.get("normalized"), norm=norm, upstream_endpoint=eb.get("upstream_endpoint"))
    ready = requests.get(f"{G}/health/ready", timeout=120)
    check("health_ready_explicit_probe", ready.status_code == 200,
          status=ready.status_code, body={k: ready.json().get(k) for k in ("status", "registry_status", "identity_verified",
                                                                           "dimension", "norm", "embedding_latency_ms", "pipeline_id")})

    # 6. Reranking réel
    hr = requests.get(f"{G}/health/rerank", timeout=120)
    hrb = hr.json() if hr.status_code == 200 else hr.json().get("detail", {})
    t = time.time()
    r = requests.post(f"{G}/v1/rerank", json={"query": "Quelle est la capitale de la France ?", "documents": [
        {"id": "d1", "text": "Le chat dort sur le canapé."},
        {"id": "d2", "text": "Paris est la capitale de la France."},
        {"id": "d3", "text": "La recette de la tarte aux pommes."}]}, timeout=300)
    rr_ms = int((time.time() - t) * 1000)
    rb = r.json()
    check("rerank", hr.status_code == 200 and hrb.get("status") == "ok" and r.status_code == 200
          and rb.get("results", [{}])[0].get("id") == "d2" and rb.get("model_revision") == hrb.get("observed_revision"),
          health=hrb, status=r.status_code, latency_ms=rr_ms, results=rb.get("results"),
          model=rb.get("model"), model_revision=rb.get("model_revision"), score_semantics=rb.get("score_semantics"))

    # 7. Négatifs
    m = requests.post(f"{G}/v1/generate", json={"prompt": "x", "model": a.missing_model}, timeout=120)
    md = m.json().get("detail", {})
    check("registered_but_not_installed", m.status_code == 503 and md.get("code") == "model_not_installed"
          and md.get("retryable") is False and "Retry-After" not in m.headers,
          status=m.status_code, detail=md)
    u = requests.post(f"{G}/v1/generate", json={"prompt": "x", "model": "llama3.1:8b"}, timeout=30)
    check("unregistered_model_refused", u.status_code == 400 and u.json()["detail"].get("code") == "model_not_registered",
          status=u.status_code, detail=u.json().get("detail"))
    ue = requests.post(f"{G}/v1/embeddings", json={"texts": ["x"], "model": "mxbai-embed-large"}, timeout=30)
    check("unregistered_embedding_refused", ue.status_code == 400, status=ue.status_code, detail=ue.json().get("detail"))

    ps_end = requests.get(f"{a.ollama}/api/ps", timeout=5).json().get("models", [])
    ev["ollama_loaded_at_end"] = [{k: m.get(k) for k in ("name", "size", "size_vram")} for m in ps_end]
    ev["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ev["verdicts"] = verdicts
    ev["overall"] = "PASS" if all(v == "PASS" for v in verdicts.values()) else "FAIL"
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(ev, fh, indent=1, ensure_ascii=False)
    print("OVERALL", ev["overall"], "->", a.out)
    return 0 if ev["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
