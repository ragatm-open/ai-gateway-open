"""Conformité Open — gateway-api/1.0.yaml == API réellement servie."""

import os
import sys

import yaml

from app import main

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_openapi  # noqa: E402

REQUIRED = {("get", "/health/runtime"), ("get", "/health/live"), ("get", "/v1/models"),
            ("post", "/v1/generate"), ("post", "/v1/generate/stream"),
            ("post", "/v1/embeddings"), ("post", "/v1/rerank")}


def _file_spec():
    with open(os.path.join(ROOT, "gateway-api", "1.0.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_contract_file_matches_the_served_api():
    assert build_openapi.build_spec() == _file_spec(), "régénérer : python tools/build_openapi.py"


def test_required_routes_present_in_file_and_app():
    spec = _file_spec()
    served = {(m, p) for p, item in main.app.openapi()["paths"].items() for m in item if m in ("get", "post", "delete")}
    for method, path in REQUIRED:
        assert method in spec["paths"][path], (method, path)
        assert (method, path) in served
    assert "/v1/vision/structured-extract" not in spec["paths"]


def test_contract_status_is_candidate_not_ratified():
    spec = _file_spec()
    assert spec["x-gateway-contract"]["status"] == "CANDIDATE_FOR_RATIFICATION"
    assert spec["info"]["version"].startswith("1.0.0-candidate")


def test_streaming_is_ndjson_not_sse():
    spec = _file_spec()
    for path in ("/v1/generate/stream", "/v1/chat/stream"):
        op = spec["paths"][path]["post"]
        assert list(op["responses"]["200"]["content"]) == ["application/x-ndjson"]
        assert op["x-gateway-stream"]["framing"] == "ndjson"
    assert spec["paths"]["/v1/generate/stream"]["post"]["x-gateway-stream"]["events"] == ["delta", "done", "error"]
    schemas = spec["components"]["schemas"]
    assert schemas["GenerateStreamDelta"]["properties"]["type"]["const"] == "delta"
    assert schemas["GenerateStreamDone"]["properties"]["type"]["const"] == "done"
    assert schemas["GenerateStreamError"]["properties"]["type"]["const"] == "error"


def test_error_envelope_headers_and_codes_documented():
    spec = _file_spec()
    env = spec["components"]["schemas"]["ErrorEnvelope"]["properties"]["detail"]
    assert set(env["required"]) == {"status", "provider", "message"}
    gen = spec["paths"]["/v1/generate"]["post"]
    for status in ("400", "413", "502", "503"):
        assert status in gen["responses"]
        assert "X-Request-ID" in gen["responses"][status]["headers"]
    assert "Retry-After" in gen["responses"]["503"]["headers"]
    codes = {e["code"] for e in spec["x-gateway-error-codes"]}
    assert {"model_not_registered", "model_not_installed", "generation_saturated", "embeddings_saturated",
            "rerank_saturated", "request_too_large", "embedding_dimension_incompatible",
            "vision_not_enabled", "asr_unavailable"} <= codes
    for entry in spec["x-gateway-error-codes"]:
        if entry["http_status"] == 503 and entry["retryable"]:
            assert "Retry-After" in entry["where"] or entry["code"] == "asr_unavailable"


def test_optional_capabilities_marked_disabled_by_default():
    spec = _file_spec()
    for path in ("/v1/vision/analyze", "/v1/audio/transcriptions", "/v1/chat"):
        op = next(iter(spec["paths"][path].values()))
        assert op["x-gateway-capability"]["default_state"] == "DISABLED"
    assert spec["paths"]["/health/live"]["get"]["x-gateway-inferential"] is False
    assert spec["paths"]["/health/runtime"]["get"]["x-gateway-inferential"] is False
    assert spec["paths"]["/health/ready"]["get"]["x-gateway-probe"] == "explicit_only"
