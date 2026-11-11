#!/usr/bin/env python3
"""Construit le contrat candidat `gateway-api/1.0.yaml`.

Source de vérité : `app.main.app.openapi()` — schémas Pydantic RÉELS et routes
réellement déclarées — enrichi de sections contractuelles `x-gateway-*`
(enveloppe d'erreur, événements NDJSON, en-têtes, états de capacités, codes
d'erreur, limites). Rien n'est écrit à la main dans les schémas : le test
`tests/open/test_open_openapi_conformance.py` reconstruit la spécification et
la compare au fichier.

Usage : python tools/build_openapi.py [--check]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

CONTRACT_VERSION = "1.0.0-candidate.2"
CONTRACT_STATUS = "CANDIDATE_FOR_RATIFICATION"
OUTPUT = os.path.join(ROOT, "gateway-api", "1.0.yaml")

REQUIRED_ROUTES = (
    ("GET", "/health/runtime"), ("GET", "/health/live"), ("GET", "/v1/models"),
    ("POST", "/v1/generate"), ("POST", "/v1/generate/stream"),
    ("POST", "/v1/embeddings"), ("POST", "/v1/rerank"),
)

# Capacité portée par chaque route et son état par défaut dans la distribution.
ROUTE_CAPABILITY = {
    "/health/live": ("health", "ENABLED", True),
    "/health/runtime": ("health", "ENABLED", True),
    "/health": ("health", "ENABLED", False),
    "/health/ready": ("embeddings", "ENABLED", False),
    "/health/embeddings": ("embeddings", "ENABLED", False),
    "/health/rerank": ("rerank", "DISABLED", False),
    "/health/asr": ("asr", "DISABLED", False),
    "/health/vision": ("vision", "DISABLED", False),
    "/v1/models": ("catalog", "ENABLED", True),
    "/v1/generate": ("generation", "ENABLED", True),
    "/v1/generate/stream": ("generation", "ENABLED", True),
    "/v1/chat": ("chat", "DISABLED", False),
    "/v1/chat/stream": ("chat", "DISABLED", False),
    "/v1/embeddings": ("embeddings", "ENABLED", True),
    "/v1/rerank": ("rerank", "DISABLED", True),
    "/v1/vision/analyze": ("vision", "DISABLED", False),
    "/v1/vision/capacity": ("vision", "DISABLED", False),
    "/v1/audio/transcriptions": ("asr", "DISABLED", False),
    "/v1/audio/transcriptions/{job_id}": ("asr", "DISABLED", False),
}
# Routes dont la sonde est INFÉRENTIELLE (jamais pour le monitoring automatique).
INFERENTIAL_PROBES = {"/health/ready", "/health/embeddings", "/health/rerank"}
NON_INFERENTIAL = {"/health/live", "/health/runtime", "/health", "/v1/models", "/v1/vision/capacity",
                   "/health/asr", "/health/vision"}

ERROR_CODES = [
    # code, http, retryable, where
    ("model_not_registered", 400, False, "generate, chat, embeddings (registre strict)"),
    ("completion_not_supported", 400, False, "generate : modèle sans capability completion"),
    ("chat_not_supported", 400, False, "chat : modèle sans capability chat"),
    ("structured_output_not_supported", 400, False, "generate/chat : capability absente"),
    ("structured_output_reasoning_incompatible", 400, False, "generate : canal de raisonnement actif"),
    ("structured_output_streaming_not_supported", 400, False, "generate/stream, chat/stream"),
    ("generate_format_conflict", 400, False, "generate : format et response_format exclusifs"),
    ("keep_alive_not_overridable", 400, False, "generate/chat : profil de rétention verrouillé"),
    ("embedding_dimension_incompatible", 400, False, "embeddings : dimension ≠ index"),
    ("invalid_content_length", 400, False, "middleware"),
    ("length_required", 411, False, "middleware : POST JSON sans Content-Length"),
    ("request_too_large", 413, False, "middleware : Content-Length > AI_GATEWAY_REQUEST_MAX_BYTES"),
    ("output_empty", 502, False, "sortie structurée vide"),
    ("output_invalid", 502, False, "sortie structurée non JSON strict"),
    ("output_mismatch", 502, False, "sortie structurée non conforme"),
    ("unsupported_upstream_response", 502, False, "chat : tool_calls sans outil"),
    ("model_not_installed", 503, False, "generate, chat, embeddings : modèle absent du runtime (aucun téléchargement)"),
    ("generation_saturated", 503, True, "generate, generate/stream, chat, chat/stream (+Retry-After)"),
    ("embeddings_saturated", 503, True, "embeddings (+Retry-After)"),
    ("rerank_saturated", 503, True, "rerank (+Retry-After)"),
    ("vision_not_enabled", 503, False, "vision : capacité OFF"),
    ("vision_task_unavailable", 503, False, "vision : semantic_extract non inclus"),
    ("vision_overloaded", 503, True, "vision (+Retry-After)"),
    ("asr_unavailable", 503, True, "asr : capacité OFF ou backend absent"),
    ("resource_saturated", 503, True, "asr (+Retry-After)"),
]

STREAM_EVENTS = {
    "/v1/generate/stream": ["GenerateStreamDelta", "GenerateStreamDone", "GenerateStreamError"],
    "/v1/chat/stream": ["ChatStreamThinking", "ChatStreamContent", "ChatStreamDone", "GenerateStreamError"],
}


def _error_envelope_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["detail"],
        "description": "Enveloppe d'erreur unique de la Gateway. `message` est fixe et public "
                       "(jamais d'URL interne ni de contenu) ; `code` est l'identifiant stable quand il existe.",
        "properties": {
            "detail": {
                "type": "object",
                "required": ["status", "provider", "message"],
                "properties": {
                    "status": {"type": "string", "const": "error"},
                    "provider": {"type": "string", "enum": ["ai_gateway", "ollama", "local_cross_encoder"]},
                    "message": {"type": "string"},
                    "code": {"type": "string"},
                    "retryable": {"type": "boolean"},
                    "retry_after_seconds": {"type": "integer"},
                    "errors": {"type": "array", "items": {"type": "object"},
                               "description": "Détail de validation (400 payload invalide)"},
                },
                "additionalProperties": True,
            }
        },
    }


def _ndjson_schema(path: str, components: Dict[str, Any]) -> Dict[str, Any]:
    from app import schemas as S
    names = STREAM_EVENTS[path]
    for name in names:
        model = getattr(S, name)
        generated = model.model_json_schema(ref_template="#/components/schemas/{model}")
        defs = generated.pop("$defs", {})
        for dname, dschema in defs.items():
            components.setdefault(dname, dschema)
        components[name] = generated
    return {"oneOf": [{"$ref": f"#/components/schemas/{n}"} for n in names]}


def build_spec() -> Dict[str, Any]:
    from app.main import app, REQUEST_ID_HEADER, GENERATE_RETRY_AFTER_SECONDS, EMBEDDINGS_RETRY_AFTER_SECONDS
    from app.main import RERANK_RETRY_AFTER_SECONDS, VISION_RETRY_AFTER_SECONDS, MEDIA_TYPE_NDJSON
    from app.settings import Settings

    spec = copy.deepcopy(app.openapi())
    spec["info"] = {
        "title": "AI Gateway Open — gateway-api",
        "version": CONTRACT_VERSION,
        "description": (
            "Contrat API commun RAGatm Open / AI Gateway Open / AI Gateway Pool Open. "
            f"État : {CONTRACT_STATUS} (ratification en attente). Dérivé des schémas réellement "
            "implémentés (app/schemas.py) par tools/build_openapi.py ; toute divergence avec l'API "
            "servie est détectée par tests/open/test_open_openapi_conformance.py."
        ),
    }
    spec["servers"] = [{"url": "http://127.0.0.1:8100", "description": "Gateway locale (boucle locale)"}]
    components = spec.setdefault("components", {}).setdefault("schemas", {})
    components["ErrorEnvelope"] = _error_envelope_schema()
    spec["components"]["headers"] = {
        "X-Request-ID": {"description": "Identifiant de corrélation : relayé s'il respecte "
                                        "^[A-Za-z0-9._:-]{1,128}$, généré (UUID v4) sinon. Présent sur toute réponse.",
                         "schema": {"type": "string", "pattern": "^[A-Za-z0-9._:-]{1,128}$"}},
        "Retry-After": {"description": "Délai de politesse (secondes entières) avant nouvel essai sur une saturation 503.",
                        "schema": {"type": "integer", "minimum": 1}},
    }
    spec["components"]["parameters"] = {
        "XRequestID": {"name": REQUEST_ID_HEADER, "in": "header", "required": False,
                       "description": "Identifiant de corrélation fourni par le client (optionnel).",
                       "schema": {"type": "string", "pattern": "^[A-Za-z0-9._:-]{1,128}$"}},
    }
    error_ref = {"$ref": "#/components/schemas/ErrorEnvelope"}
    for path, item in spec["paths"].items():
        cap = ROUTE_CAPABILITY.get(path)
        for method, op in item.items():
            if method not in ("get", "post", "delete", "put", "patch"):
                continue
            op.setdefault("parameters", []).append({"$ref": "#/components/parameters/XRequestID"})
            responses = op.setdefault("responses", {})
            for status in ("400", "502", "503"):
                if method != "get" or status != "400":
                    responses.setdefault(status, {"description": {"400": "Requête ou gouvernance refusée",
                                                                  "502": "Upstream en échec, délai ou sortie hors contrat",
                                                                  "503": "Capacité désactivée, modèle non installé, saturation ou upstream indisponible"}[status],
                                                  "content": {"application/json": {"schema": error_ref}}})
            if method == "post" and not path.startswith(("/v1/audio/", "/v1/vision/")):
                responses.setdefault("413", {"description": "Corps > AI_GATEWAY_REQUEST_MAX_BYTES (request_too_large)",
                                             "content": {"application/json": {"schema": error_ref}}})
            for status, resp in responses.items():
                resp.setdefault("headers", {})["X-Request-ID"] = {"$ref": "#/components/headers/X-Request-ID"}
                if status == "503":
                    resp["headers"]["Retry-After"] = {"$ref": "#/components/headers/Retry-After"}
            if path in STREAM_EVENTS:
                responses["200"] = {
                    "description": "Flux NDJSON (application/x-ndjson, PAS du SSE) : une ligne JSON par événement ; "
                                   "exactement un `done` par flux réussi ; un flux en erreur émet `error` et jamais `done`.",
                    "headers": {"X-Request-ID": {"$ref": "#/components/headers/X-Request-ID"}},
                    "content": {MEDIA_TYPE_NDJSON: {"schema": _ndjson_schema(path, components)}},
                }
                op["x-gateway-stream"] = {"media_type": MEDIA_TYPE_NDJSON, "framing": "ndjson",
                                          "events": [e.replace("GenerateStream", "").replace("ChatStream", "").lower()
                                                     for e in STREAM_EVENTS[path]],
                                          "client_cancellation": "fermer la connexion ferme l'appel upstream"}
            if cap:
                op["x-gateway-capability"] = {"name": cap[0], "default_state": cap[1], "required_for_compact": cap[2]}
            op["x-gateway-inferential"] = path not in NON_INFERENTIAL
            if path in INFERENTIAL_PROBES:
                op["x-gateway-probe"] = "explicit_only"
    defaults = Settings.__new__(Settings)
    try:
        import os as _os
        saved = {k: _os.environ.pop(k) for k in list(_os.environ) if k.startswith(("AI_GATEWAY_", "OLLAMA_"))}
        defaults = Settings()
    finally:
        _os.environ.update(saved)
    spec["x-gateway-contract"] = {
        "status": CONTRACT_STATUS, "version": CONTRACT_VERSION, "ratification": "pending",
        "required_routes": [f"{m} {p}" for m, p in REQUIRED_ROUTES],
        "optional_capabilities_default_off": ["asr", "vision", "chat"],
        "not_included": ["semantic_extract", "structured_extract", "v2h", "host_admission_controller"],
        "capability_states": ["ENABLED", "DISABLED", "UNAVAILABLE", "UNKNOWN"],
        "model_substitution": "never — le nom demandé est résolu (logique -> tag) ou refusé",
        "external_providers": "none — runtime local uniquement, aucun repli cloud, aucun téléchargement à la requête",
    }
    spec["x-gateway-error-codes"] = [
        {"code": c, "http_status": s, "retryable": r, "where": w} for c, s, r, w in ERROR_CODES
    ]
    spec["x-gateway-limits"] = {
        "timeout_seconds_default": defaults.timeout_seconds,
        "request_max_bytes_default": defaults.request_max_bytes,
        "generate_max_concurrency_default": defaults.generate_max_concurrency,
        "embeddings_max_concurrency_default": defaults.embeddings_max_concurrency,
        "rerank_max_concurrency_default": defaults.rerank_max_concurrency,
        "embeddings_max_batch_default": defaults.embeddings_max_batch,
        "embeddings_dimension_default": defaults.embeddings_dimension,
        "retry_after_seconds": {"generation_saturated": GENERATE_RETRY_AFTER_SECONDS,
                                "embeddings_saturated": EMBEDDINGS_RETRY_AFTER_SECONDS,
                                "rerank_saturated": RERANK_RETRY_AFTER_SECONDS,
                                "vision_overloaded": VISION_RETRY_AFTER_SECONDS},
        "model_name_pattern": "^[A-Za-z0-9._:/-]+$", "model_name_max_chars": 256,
    }
    spec["x-gateway-embeddings-identity"] = {
        "pipeline_id_format": "emb-v2:<sha256[:32] du payload canonique>",
        "payload_fields": ["schema", "model_digest", "task", "prefix", "overflow", "normalized", "implementation"],
        "response_fields": ["pipeline_id", "expected_digest", "model_digest", "identity_verified", "verified_at",
                            "governed", "normalized", "dimension"],
        "compact_v1_dimension": 768,
    }
    spec["x-gateway-pool-compatibility"] = {
        "request_id_header": REQUEST_ID_HEADER,
        "gateway_never_emits_headers_prefixed": "X-AIGP-",
        "distinguishable_errors": "detail.code + detail.retryable + HTTP status",
        "streaming": "NDJSON relayé octet pour octet ; Content-Type application/x-ndjson",
    }
    return spec


def dump(spec: Dict[str, Any]) -> str:
    return yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=120)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="vérifie que le fichier est à jour (exit 1 sinon)")
    a = ap.parse_args()
    text = dump(build_spec())
    if a.check:
        with open(OUTPUT, "r", encoding="utf-8") as fh:
            return 0 if fh.read() == text else 1
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"écrit {OUTPUT} ({len(text)} octets, {len(build_spec()['paths'])} routes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
