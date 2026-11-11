"""MCP endpoint, machine manifest and run data. Written 2026-11-11; NOT yet run (the user asked for no test runs until told)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def rpc(client, method, params=None, id_=1, headers=None):
    return client.post("/mcp", json={"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}}, headers=headers or {})


def call(client, name, args):
    r = rpc(client, "tools/call", {"name": name, "arguments": args})
    assert r.status_code == 200, r.text
    return r.json()["result"]


def test_initialize_and_notification(client):
    r = rpc(client, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
    res = r.json()["result"]
    assert r.status_code == 200 and res["protocolVersion"] == "2025-06-18" and res["serverInfo"]["name"] == "burgers-fno-service" and "tools" in res["capabilities"]
    assert rpc(client, "initialize", {"protocolVersion": "1999-01-01"}).json()["result"]["protocolVersion"] == "2025-06-18"     # falls back to our newest
    n = client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert n.status_code == 202 and n.content == b""
    assert rpc(client, "ping").json()["result"] == {}


def test_tools_list_has_schemas_and_read_only_hints(client):
    tools = rpc(client, "tools/list").json()["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"list_presets", "make_wave", "predict", "get_profile", "get_model_info", "get_status", "explain"} <= names
    for t in tools:
        assert t["inputSchema"]["type"] == "object" and t["description"] and t["annotations"]["readOnlyHint"] is True and "fn" not in t


def test_predict_preset_returns_a_verdict_and_small_profiles(client):
    res = call(client, "predict", {"preset": "sine"})
    s = res["structuredContent"]
    assert res["isError"] is False and s["verdict"]["level"] == "good" and s["verdict"]["best_answer"] == "ai"
    assert len(s["run_id"]) == 12 and s["links"]["csv"] == f"/runs/{s['run_id']}.csv" and 0 < s["difference_percent"] < 6
    p = s["profiles"]
    assert p["t"] == [0.0, 0.5, 1.0, 2.0] and len(p["x"]) == 32 and np.asarray(p["ai"]).shape == (4, 32) and np.asarray(p["solver"]).shape == (4, 32)
    assert '"verdict"' in res["content"][0]["text"]                       # clients without structuredContent still get the data


def test_square_and_tall_are_not_trusted(client):
    sq = call(client, "predict", {"preset": "square"})["structuredContent"]["verdict"]
    assert sq["level"] in ("bad", "warn") and sq["best_answer"] == "solver"
    tall = call(client, "predict", {"preset": "tall"})["structuredContent"]
    assert tall["verdict"]["level"] == "bad" and tall["ood"]["is_ood"] and tall["verdict"]["reasons"]


def test_own_values_and_components_and_no_solver(client):
    x = (np.arange(256) + 0.5) / 256
    own = call(client, "predict", {"u0": np.sin(2 * np.pi * x).tolist(), "include_solver": False})["structuredContent"]
    assert own["verdict"]["level"] == "unchecked" and own["profiles"]["solver"] is None and own["difference_percent"] is None
    comp = call(client, "predict", {"components": [{"amp": 1.0, "k": 1}], "mean": 0.0})["structuredContent"]
    assert comp["input"]["source"] == "components" and comp["verdict"]["level"] in ("good", "warn", "bad")


@pytest.mark.parametrize("args", [{}, {"preset": "sine", "components": [{"amp": 1, "k": 1}]}, {"preset": "nope"}, {"components": [{"amp": 1, "k": 0}]},
                                  {"components": [{"amp": 1, "k": 1.5}]}, {"u0": [0.0] * 10}, {"preset": "sine", "include_solver": "yes"}, {"components": []}])
def test_bad_arguments_are_tool_errors_not_crashes(client, args):
    res = call(client, "predict", args)
    assert res["isError"] is True and res["content"][0]["text"].startswith("Invalid arguments")


def test_profile_and_csv_of_a_run(client):
    rid = call(client, "predict", {"preset": "bump"})["structuredContent"]["run_id"]
    g = call(client, "get_profile", {"run_id": rid, "times": [0, 1.0, 2], "stride": 16})["structuredContent"]
    assert g["t"] == [0.0, 1.0, 2.0] and len(g["x"]) == 16 and np.asarray(g["ai"]).shape == (3, 16)
    assert call(client, "get_profile", {"run_id": "nope"})["isError"] is True
    r = client.get(f"/runs/{rid}.csv")
    lines = r.text.split("\n")
    assert r.status_code == 200 and lines[0] == "t,x,u_fno,u_solver,abs_error" and len(lines) == 201 * 256 + 1
    assert client.get("/runs/doesnotexist.csv").status_code == 404


def test_errors_resources_and_protocol_edges(client):
    assert rpc(client, "tools/call", {"name": "nope", "arguments": {}}).json()["error"]["code"] == -32602
    assert rpc(client, "frobnicate").json()["error"]["code"] == -32601
    uris = {r["uri"] for r in rpc(client, "resources/list").json()["result"]["resources"]}
    assert {"burgers://model-card", "burgers://limitations", "burgers://guide"} <= uris
    card = rpc(client, "resources/read", {"uri": "burgers://model-card"}).json()["result"]["contents"][0]
    assert card["mimeType"] == "application/json" and '"fno_s1"' in card["text"]
    assert rpc(client, "resources/read", {"uri": "burgers://nope"}).json()["error"]["code"] == -32002
    assert client.get("/mcp").status_code == 405 and client.delete("/mcp").status_code == 405
    assert client.post("/mcp", content=b"{not json", headers={"content-type": "application/json"}).status_code == 400
    batch = client.post("/mcp", json=[{"jsonrpc": "2.0", "id": 1, "method": "ping"}, {"jsonrpc": "2.0", "method": "notifications/initialized"}])
    assert batch.status_code == 200 and len(batch.json()) == 1


def test_foreign_browser_origins_are_rejected(client):
    assert rpc(client, "ping", headers={"Origin": "https://evil.example"}).status_code == 403
    assert rpc(client, "ping", headers={"Origin": "http://localhost:3000"}).status_code == 200
    assert rpc(client, "ping").status_code == 200                                  # non-browser clients send no Origin


def test_agent_manifest_and_rest_verdict(client):
    m = client.get("/agent.json").json()
    assert m["interfaces"]["mcp"]["url"].endswith("/mcp") and len(m["interfaces"]["mcp"]["tools"]) >= 7 and m["rules_for_agents"]
    assert m["interfaces"]["mcp"]["client_config_example"]["mcpServers"]["burgers"]["url"].endswith("/mcp")
    x = (np.arange(256) + 0.5) / 256
    j = client.post("/predict", json={"u0": np.sin(2 * np.pi * x).tolist(), "include_solver": True}).json()
    assert j["verdict"]["level"] == "good" and "fno" in j and "solver" in j           # the web page's contract is unchanged, plus a verdict


def test_default_page_contains_nothing_machine_oriented(client):
    html = client.get("/").text
    for needle in ("agent-manifest", "agent-state", "/mcp", "application/json", "agent.js", "pdf.js"):
        assert needle not in html, needle
