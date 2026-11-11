"""Tests du registre de modèles : chargement, validation, résolution."""

import textwrap

import pytest

from app.registry import (
    STATUS_DEGRADED,
    STATUS_READY,
    STATUS_UNAVAILABLE,
    RegistryError,
    load_registry,
)

# Profil FAST : module sans client simulé lourd, exécuté sur chaque PR.
pytestmark = pytest.mark.fast

VALID = """
version: 1
models:
  legacy-768:
    provider: ollama
    ollama_name: nomic-embed-text
    digest: "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
    kind: embedding
    dimension: 768
    languages: [en]
    cross_lingual: false
    normalized: true
    query_prefix: "search_query: "
    document_prefix: "search_document: "
    max_batch_size: 64
    max_input_chars: 6000
    on_overflow: reject
    production_allowed: true
    staging_allowed: true
"""


# Second modèle, à concaténer tel quel à VALID (indentation déjà correcte :
# textwrap.dedent la casserait, la clé retomberait au niveau racine).
EXTRA = """  {name}:
    provider: ollama
    ollama_name: {ollama_name}
    digest: "1111111111111111111111111111111111111111111111111111111111111111"
    kind: embedding
    dimension: 768
    max_batch_size: 8
    max_input_chars: 100
    on_overflow: reject
    production_allowed: true
    staging_allowed: true
"""


def _write(tmp_path, content):
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(path)


def _mutate(field, value):
    """Remplace une ligne `field: ...` du registre valide."""
    out = []
    for line in VALID.splitlines():
        if line.strip().startswith(f"{field}:"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}{field}: {value}")
        else:
            out.append(line)
    return "\n".join(out)


# ------------------------------------------------------------------ nominal
def test_load_valid_registry(tmp_path):
    reg = load_registry(_write(tmp_path, VALID))
    assert reg.version == 1
    assert set(reg.models) == {"legacy-768"}
    profile = reg.models["legacy-768"]
    assert profile.dimension == 768
    assert profile.normalized is True
    assert profile.on_overflow == "reject"
    assert profile.query_prefix == "search_query: "


def test_repository_registry_is_valid():
    """Le registre réellement livré dans l'image doit charger."""
    from app import main

    assert main.registry.version == 1
    assert "legacy-768" in main.registry.models
    assert main.registry.models["legacy-768"].ollama_name == "nomic-embed-text"


# --------------------------------------------------------------- validation
def test_missing_version(tmp_path):
    content = VALID.replace("version: 1", "")
    with pytest.raises(RegistryError, match="version"):
        load_registry(_write(tmp_path, content))


def test_unknown_version(tmp_path):
    with pytest.raises(RegistryError, match="version"):
        load_registry(_write(tmp_path, VALID.replace("version: 1", "version: 99")))


def test_no_models(tmp_path):
    with pytest.raises(RegistryError, match="aucun modèle"):
        load_registry(_write(tmp_path, "version: 1\nmodels: {}\n"))


def test_duplicate_logical_name(tmp_path):
    content = VALID + EXTRA.format(name="legacy-768", ollama_name="autre-modele")
    with pytest.raises(RegistryError, match="dupliquée"):
        load_registry(_write(tmp_path, content))


# Un champ invalide est refusé en NOMMANT le champ fautif (le `match` porte sur son nom).
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ollama_name", '""'),
        ("kind", "generation"),
        ("max_batch_size", "0"),
        ("max_input_chars", "0"),
        ("on_overflow", "ignore"),
        ("query_prefix", "42"),
        ("staging_allowed", '"oui"'),
    ],
    ids=["empty-ollama-name", "bad-kind", "bad-max-batch-size", "bad-max-input-chars",
         "bad-on-overflow", "non-textual-prefix", "non-boolean-environment-authorisation"],
)
def test_invalid_field_value_is_refused_naming_the_field(tmp_path, field, value):
    with pytest.raises(RegistryError, match=field):
        load_registry(_write(tmp_path, _mutate(field, value)))


@pytest.mark.parametrize("bad", ['""', '"zz"', '"0a109f42"', '"sha256:"'])
def test_bad_digest(tmp_path, bad):
    with pytest.raises(RegistryError, match="digest"):
        load_registry(_write(tmp_path, _mutate("digest", bad)))


def test_digest_accepts_sha256_prefix(tmp_path):
    content = _mutate(
        "digest",
        '"sha256:0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"',
    )
    reg = load_registry(_write(tmp_path, content))
    assert not reg.models["legacy-768"].digest.startswith("sha256:")


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_bad_dimension(tmp_path, bad):
    with pytest.raises(RegistryError, match="dimension"):
        load_registry(_write(tmp_path, _mutate("dimension", bad)))


def test_missing_environment_authorisation(tmp_path):
    content = "\n".join(
        line for line in VALID.splitlines() if "production_allowed" not in line
    )
    with pytest.raises(RegistryError, match="production_allowed"):
        load_registry(_write(tmp_path, content))


def test_missing_file(tmp_path):
    with pytest.raises(RegistryError, match="introuvable"):
        load_registry(str(tmp_path / "absent.yaml"))


def test_ambiguous_ollama_name(tmp_path):
    content = VALID + EXTRA.format(
        name="autre-768", ollama_name='"nomic-embed-text:latest"'
    )
    with pytest.raises(RegistryError, match="ambigu"):
        load_registry(_write(tmp_path, content))


def test_two_distinct_models_are_accepted(tmp_path):
    content = VALID + EXTRA.format(name="autre-768", ollama_name="bge-m3")
    reg = load_registry(_write(tmp_path, content))
    assert set(reg.models) == {"legacy-768", "autre-768"}
    assert reg.resolve("bge-m3").logical_name == "autre-768"


# --------------------------------------------------------------- résolution
@pytest.mark.parametrize(
    "name",
    ["legacy-768", "nomic-embed-text", "nomic-embed-text:latest"],
    ids=["logical-name", "ollama-name", "ollama-alias-latest"],
)
def test_resolve_accepts_every_known_name(tmp_path, name):
    reg = load_registry(_write(tmp_path, VALID))
    assert reg.resolve(name).logical_name == "legacy-768"


def test_resolve_unknown_returns_none(tmp_path):
    reg = load_registry(_write(tmp_path, VALID))
    assert reg.resolve("bge-m3") is None
    assert reg.resolve("") is None


def test_environment_authorisation(tmp_path):
    profile = load_registry(_write(tmp_path, VALID)).models["legacy-768"]
    assert profile.allowed_in("production") is True
    assert profile.allowed_in("staging") is True
    assert profile.allowed_in("inconnu") is False


# ------------------------------------------------------------- pipeline_id
def test_pipeline_id_is_deterministic(tmp_path):
    import re
    profile = load_registry(_write(tmp_path, VALID)).models["legacy-768"]
    again = load_registry(_write(tmp_path, VALID)).models["legacy-768"]
    for task in ("document", "query"):
        assert re.match(r"^emb-v2:[0-9a-f]{32}$", profile.pipeline_id(task))
        assert profile.pipeline_id(task) == again.pipeline_id(task)
    assert profile.pipeline_id("document") != profile.pipeline_id("query")


def test_pipeline_id_changes_with_digest(tmp_path):
    a = load_registry(_write(tmp_path, VALID)).models["legacy-768"]
    other = _mutate(
        "digest", '"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"'
    )
    b = load_registry(_write(tmp_path, other)).models["legacy-768"]
    assert a.pipeline_id("document") != b.pipeline_id("document")


# ---------------------------------------------------------- vérif. digests
class _FakeClient:
    def __init__(self, models=None, boom=False):
        self._models = models or []
        self._boom = boom

    def list_models(self, timeout=5):
        if self._boom:
            raise RuntimeError("upstream down")
        return self._models


def test_verify_digest_ready(tmp_path):
    reg = load_registry(_write(tmp_path, VALID))
    digest = reg.models["legacy-768"].digest
    reg.verify_digests(_FakeClient([{"name": "nomic-embed-text:latest", "digest": digest}]))
    profile = reg.models["legacy-768"]
    assert profile.status == STATUS_READY
    assert profile.governed is True


def test_verify_digest_mismatch_degrades(tmp_path):
    reg = load_registry(_write(tmp_path, VALID))
    reg.verify_digests(
        _FakeClient([{"name": "nomic-embed-text:latest", "digest": "b" * 64}])
    )
    profile = reg.models["legacy-768"]
    assert profile.status == STATUS_DEGRADED
    assert profile.governed is False
    assert profile.observed_digest == "b" * 64


def test_verify_model_absent(tmp_path):
    reg = load_registry(_write(tmp_path, VALID))
    reg.verify_digests(_FakeClient([{"name": "llama3.1:8b", "digest": "c" * 64}]))
    profile = reg.models["legacy-768"]
    assert profile.status == STATUS_UNAVAILABLE
    # Un modèle absent reste gouverné : on échoue « ouvert » sur un incident de
    # transport, « fermé » seulement sur une divergence prouvée.
    assert profile.governed is True


def test_verify_upstream_down_never_raises(tmp_path):
    reg = load_registry(_write(tmp_path, VALID))
    reg.verify_digests(_FakeClient(boom=True))
    assert reg.models["legacy-768"].status == STATUS_UNAVAILABLE
