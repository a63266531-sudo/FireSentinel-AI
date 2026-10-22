"""FastAPI behaviour tests. They need `fastapi` + `httpx` (requirements-dev.txt) and are SKIPPED otherwise."""
import os
import unittest
from unittest import mock


def _client():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        raise unittest.SkipTest("fastapi/httpx not installed")
    from firesentinel import api
    return TestClient(api.app), api


def test_health_and_error_schema():
    c, api = _client()
    assert c.get("/health").json()["status"] == "ok"
    r = c.get("/validate", params={"region": "Atlantis"})
    assert r.status_code == 404 and "error" in r.json()
    assert c.get("/risk", params={"weeks": 3}).status_code == 422          # input validation
    assert c.get("/risk", params={"weather": "x"}).status_code == 422


def test_api_key_is_enforced_when_configured():
    c, api = _client()
    with mock.patch.dict(os.environ, {"FIRESENTINEL_API_KEY": "k-test-123456"}):
        assert c.get("/model").status_code == 401
        assert c.get("/model", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/model", headers={"X-API-Key": "k-test-123456"}).status_code == 200


def test_rate_limit():
    c, api = _client()
    api._hits.clear()
    with mock.patch.object(api, "RATE_LIMIT_PER_MIN", 3):
        codes = [c.get("/model").status_code for _ in range(5)]
    api._hits.clear()
    assert codes[:3] == [200, 200, 200] and 429 in codes[3:]
