"""Lot P1 — `pipeline_id` dérivé de la transformation effective.

Ancien identifiant : `<logical>@<digest[:8]>/<task>-v1`. Sous-sensible (un
changement de préfixe, de normalisation ou de politique de troncature ne le
changeait pas) et sur-sensible (un simple renommage logique le changeait).

Nouvel identifiant : `emb-v2:<32 hex>` = SHA-256 d'un payload canonique
(JSON trié, séparateurs compacts) qui ne contient QUE ce qui façonne le
vecteur : digest complet, task, préfixe effectif de la task, politique
d'overflow si elle transforme le texte, normalisation, version du code de
préparation. Tout ce qui est documentaire (nom logique, langues,
`cross_lingual`, batch, allowlists, environnement) en est EXCLU.
"""

import dataclasses
import hashlib
import json
import re
import textwrap

import pytest

from app import main
from app.registry import (
    EMBEDDING_PIPELINE_SCHEMA_VERSION,
    PIPELINE_ID_HEX_CHARS,
    PREPROCESSING_VERSION,
    canonical_json,
    embedding_pipeline_id,
    embedding_pipeline_identity,
    load_registry,
)

ID_RE = re.compile(r"^emb-v2:[0-9a-f]{32}$")

# Contrat du schéma courant : ces valeurs sont FIGÉES. Elles ne bougent que si
# la transformation (ou le schéma/version d'implémentation) bouge — et alors
# une réindexation des collections concernées est due.
SNAPSHOTS = {
    ("legacy-768", "query"): "emb-v2:3a49e14ec5df7088cebb7630fd1d27eb",
    ("legacy-768", "document"): "emb-v2:9e883f0fe1d326ef17f86027a1d4a5d8",
    ("fr-en-retrieval", "query"): "emb-v2:da35d8329da5f8dc1b57d0b675fca563",
    ("fr-en-retrieval", "document"): "emb-v2:c8215aca68cbf785bf977a46ead4b23c",
}


def _legacy():
    return main.registry.models["legacy-768"]


def _same_but(profile, **changes):
    return dataclasses.replace(profile, **changes)


# --------------------------------------------------------------------------- #
# 1. Forme, algorithme, versions
# --------------------------------------------------------------------------- #
def test_format_and_algorithm():
    p = _legacy()
    ident = p.pipeline_identity("query")
    pid = p.pipeline_id("query")
    assert ID_RE.match(pid)
    expected = hashlib.sha256(canonical_json(ident).encode("utf-8")).hexdigest()[:PIPELINE_ID_HEX_CHARS]
    assert pid == f"emb-v{EMBEDDING_PIPELINE_SCHEMA_VERSION}:{expected}"
    assert EMBEDDING_PIPELINE_SCHEMA_VERSION == 2 and PIPELINE_ID_HEX_CHARS == 32


def test_canonical_payload_contents():
    ident = _legacy().pipeline_identity("query")
    assert ident == {
        "schema": 2,
        "model_digest": "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
        "task": "query",
        "prefix": "search_query: ",
        "overflow": {"policy": "reject"},
        "normalized": True,
        "implementation": PREPROCESSING_VERSION,
    }
    for excluded in ("logical_name", "ollama_name", "languages", "cross_lingual",
                     "max_batch_size", "dimension", "production_allowed",
                     "staging_allowed", "environment", "max_input_chars"):
        assert excluded not in ident


def test_canonical_json_is_sorted_compact_utf8():
    s = canonical_json({"b": 1, "a": "é", "c": {"z": 0, "y": 1}})
    assert s == '{"a":"é","b":1,"c":{"y":1,"z":0}}'
    assert canonical_json(json.loads(s)) == s


# --------------------------------------------------------------------------- #
# 2. Snapshots — contrat du schéma courant, un par pipeline embarqué
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", sorted(SNAPSHOTS))
def test_shipped_pipeline_snapshots(key):
    logical, task = key
    pid = main.registry.models[logical].pipeline_id(task)
    assert ID_RE.match(pid)
    assert pid == SNAPSHOTS[key], f"{key} : contrat du schéma modifié — réindexation due ?"


def test_query_and_document_differ_for_every_shipped_model():
    for p in main.registry.models.values():
        assert p.pipeline_id("query") != p.pipeline_id("document"), p.logical_name
        assert p.pipeline_identity("query")["prefix"] == p.query_prefix
        assert p.pipeline_identity("document")["prefix"] == p.document_prefix


def test_all_shipped_ids_are_distinct():
    ids = [p.pipeline_id(t) for p in main.registry.models.values() for t in ("query", "document")]
    assert len(set(ids)) == len(ids) == 4


# --------------------------------------------------------------------------- #
# 3. Sensibilité — DOIT changer
# --------------------------------------------------------------------------- #
def test_changes_with_full_digest_even_if_first_8_equal():
    p = _legacy()
    same8 = p.digest[:8] + "f" * 56
    assert same8 != p.digest and same8[:8] == p.digest[:8]
    assert _same_but(p, digest=same8).pipeline_id("query") != p.pipeline_id("query")


def test_changes_with_task():
    p = _legacy()
    assert p.pipeline_id("query") != p.pipeline_id("document")


def test_changes_with_query_prefix_for_query_task_only():
    p = _legacy()
    q = _same_but(p, query_prefix="AUTRE: ")
    assert q.pipeline_id("query") != p.pipeline_id("query")
    assert q.pipeline_id("document") == p.pipeline_id("document")     # non concerné


def test_changes_with_document_prefix_for_document_task_only():
    p = _legacy()
    d = _same_but(p, document_prefix="passage: ")
    assert d.pipeline_id("document") != p.pipeline_id("document")
    assert d.pipeline_id("query") == p.pipeline_id("query")


def test_changes_with_normalized():
    p = _legacy()
    assert _same_but(p, normalized=False).pipeline_id("query") != p.pipeline_id("query")


def test_changes_with_truncating_overflow_policy():
    p = _legacy()
    t = _same_but(p, on_overflow="truncate_and_flag")
    assert t.pipeline_id("query") != p.pipeline_id("query")
    assert t.pipeline_identity("query")["overflow"] == {"policy": "truncate", "max_chars": 6000}
    # Sous troncature, la borne façonne le texte : elle compte.
    assert _same_but(t, max_input_chars=100).pipeline_id("query") != t.pipeline_id("query")


def test_changes_with_implementation_version():
    p = _legacy()
    base = dict(digest=p.digest, task="query", prefix=p.query_prefix, normalized=p.normalized,
                on_overflow=p.on_overflow, max_input_chars=p.max_input_chars)
    v1 = embedding_pipeline_id(embedding_pipeline_identity(**base, implementation_version="1"))
    v2 = embedding_pipeline_id(embedding_pipeline_identity(**base, implementation_version="2"))
    assert v1 != v2


def test_changes_with_schema_version():
    p = _legacy()
    base = dict(digest=p.digest, task="query", prefix=p.query_prefix, normalized=p.normalized,
                on_overflow=p.on_overflow, max_input_chars=p.max_input_chars)
    s2 = embedding_pipeline_id(embedding_pipeline_identity(**base, schema_version=2))
    s3 = embedding_pipeline_id(embedding_pipeline_identity(**base, schema_version=3))
    assert s2 != s3 and s3.startswith("emb-v3:")


# --------------------------------------------------------------------------- #
# 4. Insensibilité — NE DOIT PAS changer
# --------------------------------------------------------------------------- #
def test_rename_logical_name_keeps_identity():
    p = _legacy()
    renamed = _same_but(p, logical_name="renamed-768")
    assert renamed.pipeline_id("query") == p.pipeline_id("query")
    assert renamed.pipeline_id("document") == p.pipeline_id("document")
    assert renamed.logical_name != p.logical_name                   # exposé à part


def test_max_input_chars_under_reject_is_admission_only():
    p = _legacy()
    assert p.on_overflow == "reject"
    assert _same_but(p, max_input_chars=100).pipeline_id("query") == p.pipeline_id("query")


@pytest.mark.parametrize("field,value", [
    ("languages", ["fr", "en", "de"]), ("cross_lingual", True), ("max_batch_size", 8),
    ("production_allowed", False), ("staging_allowed", False), ("ollama_name", "renamed:latest"),
])
def test_documentary_fields_keep_identity(field, value):
    p = _legacy()
    assert _same_but(p, **{field: value}).pipeline_id("query") == p.pipeline_id("query")


def test_environment_keeps_identity(monkeypatch):
    p = _legacy()
    monkeypatch.setattr(main.settings, "environment", "staging")
    a = p.pipeline_id("query")
    monkeypatch.setattr(main.settings, "environment", "production")
    assert p.pipeline_id("query") == a


# --------------------------------------------------------------------------- #
# 5. Stabilité
# --------------------------------------------------------------------------- #
YAML_A = """
    version: 1
    models:
      emb:
        provider: ollama
        ollama_name: nomic-embed-text
        digest: "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
        kind: embedding
        dimension: 768
        normalized: true
        query_prefix: "search_query: "
        document_prefix: "search_document: "
        max_batch_size: 64
        max_input_chars: 6000
        on_overflow: reject
        production_allowed: true
        staging_allowed: true
"""
# Mêmes valeurs, clés dans un autre ordre, commentaire ajouté, nom logique différent.
YAML_B = """
    version: 1
    # commentaire purement documentaire
    models:
      autre-nom:
        staging_allowed: true
        production_allowed: true
        on_overflow: reject          # idem
        max_input_chars: 6000
        max_batch_size: 64
        document_prefix: "search_document: "
        query_prefix: "search_query: "
        normalized: true
        dimension: 768
        kind: embedding
        digest: "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
        ollama_name: nomic-embed-text
        provider: ollama
"""


def _load(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return load_registry(str(path))


def test_stable_across_reload_instances_key_order_and_comments(tmp_path):
    a1 = _load(tmp_path, "a1.yaml", YAML_A).models["emb"]
    a2 = _load(tmp_path, "a2.yaml", YAML_A).models["emb"]
    b = _load(tmp_path, "b.yaml", YAML_B).models["autre-nom"]
    for task in ("query", "document"):
        assert a1.pipeline_id(task) == a2.pipeline_id(task) == b.pipeline_id(task)
    # Et identique au profil embarqué (même transformation) malgré le nom.
    assert a1.pipeline_id("query") == _legacy().pipeline_id("query")


def test_hash_does_not_depend_on_payload_key_insertion_order():
    ident = _legacy().pipeline_identity("query")
    shuffled = dict(reversed(list(ident.items())))
    assert list(shuffled) != list(ident)
    assert embedding_pipeline_id(shuffled) == embedding_pipeline_id(ident)


def test_hash_is_not_python_hash():
    """`hash()` Python est salé par process ; SHA-256 ne l'est pas."""
    ident = _legacy().pipeline_identity("query")
    digest = hashlib.sha256(canonical_json(ident).encode("utf-8")).hexdigest()
    assert embedding_pipeline_id(ident) == f"emb-v2:{digest[:32]}"


# --------------------------------------------------------------------------- #
# 6. Surface HTTP : même identifiant partout, jamais loggé en entier
# --------------------------------------------------------------------------- #
def test_http_surfaces_expose_the_same_identity(clientapp, monkeypatch):
    from app.ollama_client import ENDPOINT_EMBED, EmbedResult
    monkeypatch.setattr(main.client, "embed",
                        lambda model, inputs, keep_alive=None: EmbedResult(
                            [[1.0] + [0.0] * 767 for _ in inputs], ENDPOINT_EMBED,
                            fallback_used=False, ollama_metadata={}))
    p = _legacy()
    r = clientapp.post("/v1/embeddings", json={"texts": ["a"], "task": "query"}).json()
    assert r["pipeline_id"] == p.pipeline_id("query") == SNAPSHOTS[("legacy-768", "query")]
    assert r["model_logical"] == "legacy-768" and r["model_digest"] == p.digest
    ready = clientapp.get("/health/ready").json()
    assert ready["pipeline_id"] == p.pipeline_id("document")
    entry = [m for m in clientapp.get("/health/embeddings").json()["models"]
             if m["model_logical"] == "legacy-768"][0]
    assert entry["pipeline_id_query"] == p.pipeline_id("query")
    assert entry["pipeline_id_document"] == p.pipeline_id("document")
