"""Lot build-provenance-observability — la provenance est observable en HTTP.

Deux familles d'invariants :

* **surface HTTP** : `/health/runtime` expose `build.git_sha` et
  `rerank.backend.{implementation,image_digest}` sans jamais inventer une
  valeur absente (`not_reported`) ni laisser fuir une URL/un port interne ;
* **chaîne de déploiement** : le SHA est injecté au build (Dockerfile ARG→ENV,
  jamais de `git` runtime) et le digest TEI n'a qu'UNE source de vérité, le
  fichier `.env.example` versionné (copié en `.env`, ignoré par git), consommé
  par les fichiers Compose (profil Open : `docker-compose.yml` et
  `docker-compose.reranker.yml`).
"""

import json
import os
import re

import pytest
import yaml

from app import main
from app.settings import Settings

# Profil FAST : module sans client simulé lourd, exécuté sur chaque PR.
pytestmark = pytest.mark.fast

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLD_SHA = "d55d905ceed6e54d0bb086545407e114a82e2b26"
GOLD_DIGEST = (
    "sha256:7fe3079ce91cd926f3f5e0bc6dfda14b1d94dbff2e0e0f587ac48ef42c6907b9"
)


def _env_file(path):
    values = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.split("#")[0].strip()
    return values


# --------------------------------------------------------------- settings
def test_build_sha_valid_is_kept(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_BUILD_SHA", GOLD_SHA.upper())
    assert Settings().build_sha == GOLD_SHA  # normalisé en minuscules


def test_build_sha_missing_or_invalid_is_none(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_BUILD_SHA", raising=False)
    assert Settings().build_sha is None
    for bad in ("", "  ", "d55d905", "not-a-sha", GOLD_SHA + "0", "x" * 40):
        monkeypatch.setenv("AI_GATEWAY_BUILD_SHA", bad)
        assert Settings().build_sha is None, bad


def test_backend_digest_valid_is_kept(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_RERANK_BACKEND_IMAGE_DIGEST", GOLD_DIGEST)
    assert Settings().rerank_backend_image_digest == GOLD_DIGEST


def test_backend_digest_missing_or_invalid_is_none(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_RERANK_BACKEND_IMAGE_DIGEST", raising=False)
    assert Settings().rerank_backend_image_digest is None
    for bad in ("7fe3079c", "sha256:7fe3", "md5:" + "0" * 64, "sha256:" + "g" * 64):
        monkeypatch.setenv("AI_GATEWAY_RERANK_BACKEND_IMAGE_DIGEST", bad)
        assert Settings().rerank_backend_image_digest is None, bad


# ------------------------------------------------------- /health/runtime
def test_runtime_exposes_build_provenance(clientapp, monkeypatch):
    monkeypatch.setattr(main.settings, "build_sha", GOLD_SHA)
    build = clientapp.get("/health/runtime").json()["build"]
    assert build["git_sha"] == GOLD_SHA
    assert build["version"] == main.__version__
    # Image construite localement, jamais poussée : pas de digest de registre,
    # et on n'en fabrique pas un depuis le filesystem.
    assert build["image_digest"] == "not_reported"


def test_runtime_build_sha_never_invented(clientapp, monkeypatch):
    monkeypatch.setattr(main.settings, "build_sha", None)
    build = clientapp.get("/health/runtime").json()["build"]
    assert build["git_sha"] == "not_reported"


def test_runtime_exposes_rerank_backend_identity(clientapp, monkeypatch):
    monkeypatch.setattr(main.settings, "rerank_backend_image_digest", GOLD_DIGEST)
    rerank = clientapp.get("/health/runtime").json()["rerank"]
    assert rerank["backend"]["implementation"] == "tei"
    assert rerank["backend"]["image_digest"] == GOLD_DIGEST
    # Le modèle et sa révision restent portés par le profil du registre.
    assert rerank["models"][0]["artifact"] == "BAAI/bge-reranker-v2-m3"
    assert re.fullmatch(r"[0-9a-f]{40}", rerank["models"][0]["revision"])


def test_runtime_backend_digest_never_invented(clientapp, monkeypatch):
    monkeypatch.setattr(main.settings, "rerank_backend_image_digest", None)
    rerank = clientapp.get("/health/runtime").json()["rerank"]
    assert rerank["backend"]["image_digest"] == "not_reported"


def test_runtime_provenance_leaks_nothing_sensitive(clientapp, monkeypatch):
    """Ni URL, ni port interne, ni chemin d'hôte — provenance = identités pures."""
    monkeypatch.setattr(main.settings, "build_sha", GOLD_SHA)
    monkeypatch.setattr(main.settings, "rerank_backend_image_digest", GOLD_DIGEST)
    text = clientapp.get("/health/runtime").text
    for forbidden in ("http://", "https://", "127.0.0.1", "8300", "11434",
                      "backend_url", "/opt/", "/data"):
        assert forbidden not in text, forbidden


def test_runtime_contract_stays_backward_compatible(clientapp):
    """Les champs pré-lot gardent nom et sens : ajout STRICTEMENT additif."""
    body = clientapp.get("/health/runtime").json()
    for key in ("provider", "version", "environment", "registry_version",
                "llm_models", "context_profiles", "keep_alive_profiles",
                "inflight", "rerank", "queue_time"):
        assert key in body, key
    assert body["queue_time"] == "not_observable"
    assert body["rerank"]["enabled"] is False
    assert body["rerank"]["default_model"] == "rerank-default"


def test_staging_and_production_builds_are_distinguishable(clientapp, monkeypatch):
    """Deux Gateways partageant le même sidecar restent discernables : le
    couple (environment, build.git_sha) diffère dès que l'env ou le build
    diffère — même digest backend des deux côtés."""
    monkeypatch.setattr(main.settings, "environment", "staging")
    monkeypatch.setattr(main.settings, "build_sha", "a" * 40)
    one = clientapp.get("/health/runtime").json()

    monkeypatch.setattr(main.settings, "environment", "production")
    monkeypatch.setattr(main.settings, "build_sha", "b" * 40)
    two = clientapp.get("/health/runtime").json()

    assert one["environment"] == "staging" and two["environment"] == "production"
    assert one["build"]["git_sha"] != two["build"]["git_sha"]
    assert one["rerank"]["backend"] == two["rerank"]["backend"]
    # La gouvernance est VISIBLE des deux côtés (true depuis le lot
    # rerank-production-capability-readiness — la frontière OFF est RagATM).
    prod_profile = {m["model_logical"]: m for m in two["rerank"]["models"]}
    assert prod_profile["rerank-default"]["production_allowed"] is True


# ----------------------------------------------- chaîne de déploiement
def test_dockerfile_injects_sha_at_build():
    with open(os.path.join(REPO, "Dockerfile"), "r", encoding="utf-8") as handle:
        content = handle.read()
    assert 'ARG AI_GATEWAY_BUILD_SHA=""' in content
    assert "ENV AI_GATEWAY_BUILD_SHA=${AI_GATEWAY_BUILD_SHA}" in content
    assert "org.opencontainers.image.revision=${AI_GATEWAY_BUILD_SHA}" in content
    # Jamais de git runtime : aucune instruction n'installe ni n'exécute git
    # dans l'image, et le dépôt (.git compris) n'y est jamais copié.
    instructions = [l for l in content.splitlines()
                    if l.strip() and not l.strip().startswith("#")]
    assert not [l for l in instructions if l.startswith("RUN") and "git" in l]
    assert not [l for l in instructions
                if l.startswith("COPY") and not l.startswith(("COPY requirements.txt", "COPY app/"))]


def test_build_arg_is_wired_by_the_open_compose():
    """Profil Open : le Compose de développement autonome
    construit l'image localement et passe le SHA par l'argument de build
    `AI_GATEWAY_BUILD_SHA` (jamais par `environment`, jamais par un `git`
    exécuté dans l'image)."""
    with open(os.path.join(REPO, "docker-compose.yml"), "r", encoding="utf-8") as handle:
        service = yaml.safe_load(handle)["services"]["ai-gateway"]
    assert "AI_GATEWAY_BUILD_SHA" in service["build"]["args"]
    assert "environment" not in service


def test_env_example_is_the_single_source_of_backend_digest():
    """Le digest TEI vit dans .env.example (versionné, copié en .env) et NULLE PART ailleurs ;
    aucun .env actif n'est distribué avec le dépôt."""
    assert not os.path.exists(os.path.join(REPO, ".env")), ".env (copie locale) ne doit pas être distribué"
    example = os.path.join(REPO, ".env.example")
    values = _env_file(example)
    assert "AI_GATEWAY_RERANK_BACKEND_IMAGE_DIGEST" in values, "digest TEI absent de .env.example"
    digest = values["AI_GATEWAY_RERANK_BACKEND_IMAGE_DIGEST"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    # Aucun secret dans l'exemple distribué : seules des valeurs de configuration.
    assert not any(k.endswith(("_TOKEN", "_SECRET", "_PASSWORD", "_KEY")) for k in values), sorted(values)
    with open(example, "r", encoding="utf-8") as handle:
        assert handle.read().count(digest) == 1, "digest dupliqué dans .env.example"

    # Aucune duplication du littéral dans les fichiers Compose ni le Dockerfile :
    # uniquement l'interpolation `${AI_GATEWAY_RERANK_BACKEND_IMAGE_DIGEST:?…}`.
    for name in ("docker-compose.yml", "docker-compose.reranker.yml", "Dockerfile"):
        with open(os.path.join(REPO, name), "r", encoding="utf-8") as handle:
            assert digest not in handle.read(), f"digest dupliqué dans {name}"


def test_reranker_compose_pins_image_via_dotenv():
    path = os.path.join(REPO, "docker-compose.reranker.yml")
    with open(path, "r", encoding="utf-8") as handle:
        image = yaml.safe_load(handle)["services"]["reranker"]["image"]
    assert image.startswith("ghcr.io/huggingface/text-embeddings-inference@")
    assert "${AI_GATEWAY_RERANK_BACKEND_IMAGE_DIGEST:?" in image


def test_gateway_composes_feed_digest_via_env_file_not_environment():
    """Pas de bloc `environment:` — le digest arrive par env_file (.env, copie
    locale de .env.example)."""
    for name in ("docker-compose.yml",):
        with open(os.path.join(REPO, name), "r", encoding="utf-8") as handle:
            service = yaml.safe_load(handle)["services"]["ai-gateway"]
        assert "environment" not in service, name
        assert "./.env" in service["env_file"], name
