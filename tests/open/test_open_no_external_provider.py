"""Conformité Open — aucun fournisseur distant, aucun repli cloud, aucun
téléchargement implicite, configuration réseau générique."""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(ROOT, "app")

PROVIDER_TOKENS = ("openai", "anthropic", "mistral.ai", "api.cohere", "googleapis", "azure", "bedrock",
                   "together.ai", "groq.com", "replicate", "huggingface.co/api")
# Le code ne doit jamais invoquer un téléchargement de poids (la commande
# `ollama pull` n'apparaît que dans un commentaire d'exploitation).
PULL_TOKENS = ("/api/pull", "hf_hub_download", "snapshot_download", "subprocess")


def _app_sources():
    for name in sorted(os.listdir(APP)):
        if name.endswith(".py"):
            yield name, open(os.path.join(APP, name), encoding="utf-8").read()


def test_no_remote_provider_in_code():
    for name, text in _app_sources():
        low = text.lower()
        for token in PROVIDER_TOKENS:
            assert token not in low, (name, token)


def test_no_implicit_model_download():
    for name, text in _app_sources():
        for token in PULL_TOKENS:
            assert token not in text, (name, token)


def test_no_absolute_remote_url_in_code():
    for name, text in _app_sources():
        for m in re.finditer(r"https?://([A-Za-z0-9.-]+)", text):
            host = m.group(1)
            assert host in ("127.0.0.1", "localhost", "example.com", "example.invalid", "json-schema.org"), (name, host)


def test_requirements_contain_no_provider_sdk():
    direct = open(os.path.join(ROOT, "requirements.in"), encoding="utf-8").read().lower()
    for sdk in ("openai", "anthropic", "google-generativeai", "cohere", "mistralai", "boto3", "litellm", "langchain"):
        assert sdk not in direct, sdk


def test_upstreams_are_loopback_by_default_and_env_configurable():
    from app.settings import Settings
    s = Settings()
    assert s.ollama_base_url.startswith("http://") and "127.0.0.1" in s.ollama_base_url or "ollama" in s.ollama_base_url
    assert "127.0.0.1" in s.rerank_backend_url
    text = open(os.path.join(ROOT, ".env.example"), encoding="utf-8").read()
    assert "OLLAMA_BASE_URL=http://ollama:11434" in text          # nom de service Compose, pas une machine
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b(?<!127\.0\.0\.1)", text)
