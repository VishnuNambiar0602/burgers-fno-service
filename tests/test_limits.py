"""Rate limit, concurrency cap, MCP switch and the commit hash in /health. Added 2026-11-12 for the public deployment."""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import limits
from app.main import app

X = (np.arange(256) + 0.5) / 256
BODY = {"u0": np.sin(2 * np.pi * X).tolist(), "include_solver": False}
PING = {"jsonrpc": "2.0", "id": 1, "method": "ping"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _fresh_counters():
    limits.reset()
    yield
    limits.reset()


def test_defaults_are_twenty_per_minute_and_two_at_once(monkeypatch):
    for k in ("RATE_LIMIT_PER_MIN", "MAX_CONCURRENT", "TRUST_PROXY"):
        monkeypatch.delenv(k, raising=False)
    assert limits.limit_per_min() == 20 and limits.max_concurrent() == 2


def test_fourth_predict_in_a_minute_is_refused_with_retry_after(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "3")
    assert [client.post("/predict", json=BODY).status_code for _ in range(3)] == [200, 200, 200]
    r = client.post("/predict", json=BODY)
    assert r.status_code == 429 and 1 <= int(r.headers["retry-after"]) <= 60 and "Rate limit exceeded" in r.json()["detail"]
    assert all(client.get("/health").status_code == 200 for _ in range(10))             # only /predict and /mcp are limited
    assert client.post("/predict", json=BODY, headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 200   # another client has its own allowance


def test_limit_also_covers_mcp_and_answers_in_jsonrpc(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    assert client.post("/mcp", json=PING).status_code == 200 and client.post("/mcp", json=PING).status_code == 200
    r = client.post("/mcp", json=PING)
    assert r.status_code == 429 and r.json()["error"]["code"] == -32000 and int(r.headers["retry-after"]) >= 1


def test_clients_are_told_apart_by_forwarded_address_unless_proxy_trust_is_off(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    assert client.post("/mcp", json=PING, headers={"X-Forwarded-For": "198.51.100.1, 10.0.0.1"}).status_code == 200
    assert client.post("/mcp", json=PING, headers={"X-Forwarded-For": "198.51.100.1"}).status_code == 429
    assert client.post("/mcp", json=PING, headers={"X-Forwarded-For": "198.51.100.2"}).status_code == 200
    limits.reset()
    monkeypatch.setenv("TRUST_PROXY", "0")                                              # header ignored: everyone is the socket address
    assert client.post("/mcp", json=PING, headers={"X-Forwarded-For": "198.51.100.3"}).status_code == 200
    assert client.post("/mcp", json=PING, headers={"X-Forwarded-For": "198.51.100.4"}).status_code == 429


def test_zero_switches_the_limit_off(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "0")
    assert all(client.post("/mcp", json=PING).status_code == 200 for _ in range(30))


def test_concurrency_cap_answers_busy_instead_of_crashing(client, monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT", "1")
    with limits.Slot():                                                                  # one forecast "running"
        r = client.post("/predict", json=BODY)
        assert r.status_code == 429 and r.headers["retry-after"] == "3" and "busy" in r.json()["detail"]
        res = client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "predict", "arguments": {"preset": "sine"}}}).json()["result"]
        assert res["isError"] is True and "busy" in res["content"][0]["text"]
    assert client.post("/predict", json=BODY).status_code == 200                       # the slot was released
    assert limits.max_concurrent() == 1


def test_mcp_is_off_by_default_and_the_switch_works(client, monkeypatch):
    monkeypatch.delenv("MCP_ENABLED", raising=False)
    assert client.post("/mcp", json=PING).status_code == 404 and client.get("/mcp").status_code == 404 and "MCP_ENABLED=1" in client.post("/mcp", json=PING).json()["detail"]
    m = client.get("/agent.json").json()["interfaces"]["mcp"]
    assert m["enabled"] is False and "tools" not in m and client.get("/api").json()["mcp"] is None
    assert client.post("/predict", json=BODY).status_code == 200                       # the REST API is unaffected
    monkeypatch.setenv("MCP_ENABLED", "1")
    assert client.post("/mcp", json=PING).status_code == 200 and client.get("/mcp").status_code == 405
    m = client.get("/agent.json").json()["interfaces"]["mcp"]
    assert m["enabled"] is True and len(m["tools"]) >= 7 and client.get("/api").json()["mcp"] == "/mcp"
    monkeypatch.setenv("MCP_ENABLED", "0")
    assert client.post("/mcp", json=PING).status_code == 404


def test_commit_hash_is_in_health_and_model_info_from_the_platform_variable(monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", "0123456789abcdef0123456789abcdef01234567")
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["commit"] == "0123456789abcdef0123456789abcdef01234567" and h["commit_source"] == "RENDER_GIT_COMMIT"
        assert c.get("/model-info").json()["runtime"]["commit"] == h["commit"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_commit_hash_falls_back_to_git_head(monkeypatch):
    for k in ("RENDER_GIT_COMMIT", "GIT_COMMIT", "SOURCE_COMMIT"):
        monkeypatch.delenv(k, raising=False)
    root = os.path.join(os.path.dirname(__file__), "..")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["commit"] == head and h["commit_source"] == "git"
