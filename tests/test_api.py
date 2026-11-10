import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app
from src.models import FNO1d

ROOT = os.path.join(os.path.dirname(__file__), "..")
VAL = np.load(os.path.join(ROOT, "data", "processed", "val.npy"), mmap_mode="r")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def offline_prediction(u0):
    """Prediction from the ORIGINAL training checkpoint (not the packaged file), to verify the packaging."""
    ck = torch.load(os.path.join(ROOT, "checkpoints", "fno_s1", "best.pt"), map_location="cpu", weights_only=False)
    a = ck["args"]
    m = FNO1d(201, a["modes"], a["width"], a["layers"]); m.load_state_dict(ck["model"]); m.eval()
    with torch.no_grad():
        return m(torch.tensor(np.asarray(u0, dtype=np.float32))[None])[0].numpy()


def test_health_and_model_info(client):
    r = client.get("/health"); assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_loaded"]
    j = client.get("/model-info").json()
    assert j["name"] == "fno_s1" and j["parameters"] == 837513 and len(j["runtime"]["served_sha256"]) == 64
    assert j["metrics"]["deployed_seed1_test"] > 0 and j["known_limitations"]


def test_predict_matches_offline_original_checkpoint(client):
    u0 = np.asarray(VAL[7, 0], dtype=np.float64)
    r = client.post("/predict", json={"u0": u0.tolist()})
    assert r.status_code == 200
    j = r.json()
    u = np.asarray(j["fno"]["u"])
    assert u.shape == (201, 256) and len(j["x"]) == 256 and len(j["t"]) == 201
    off = offline_prediction(u0)
    assert np.abs(u - off).max() < 2e-6          # JSON rounding to 6 decimals is the only difference
    assert j["ood"]["is_ood"] is False and "solver" not in j


def test_predict_accuracy_on_validation_truth(client):
    errs = []
    for i in (0, 1, 2):
        u = np.asarray(client.post("/predict", json={"u0": np.asarray(VAL[i, 0], dtype=float).tolist()}).json()["fno"]["u"])
        tr = np.asarray(VAL[i], dtype=float)
        errs.append(np.sqrt(((u - tr) ** 2).sum() / (tr ** 2).sum()))
    assert max(errs) < 0.06, errs


def test_include_solver_and_agreement(client):
    u0 = np.asarray(VAL[1, 0], dtype=np.float64)
    j = client.post("/predict", json={"u0": u0.tolist(), "include_solver": True}).json()
    s = np.asarray(j["solver"]["u"]); tr = np.asarray(VAL[1], dtype=float)
    assert np.sqrt(((s - tr) ** 2).sum() / (tr ** 2).sum()) < 0.01            # solver matches the stored data
    assert j["solver"]["rel_l2_fno_vs_solver_whole"] < 0.06 and j["solver"]["mass_drift"] < 1e-9


def test_512_points_accepted(client):
    x = (np.arange(512) + 0.5) / 512
    j = client.post("/predict", json={"u0": np.sin(2 * np.pi * x).tolist()}).json()
    assert np.asarray(j["fno"]["u"]).shape == (201, 512)


@pytest.mark.parametrize("bad", [[0.0] * 255, [0.0] * 300, [float("nan")] + [0.0] * 255, [1e6] + [0.0] * 255])
def test_bad_input_rejected(client, bad):
    body = json.dumps({"u0": bad}, allow_nan=True)
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_missing_body_rejected(client):
    assert client.post("/predict", json={}).status_code == 422


def _flag(client, u0):
    return client.post("/predict", json={"u0": list(map(float, u0))}).json()["ood"]


def test_ood_flags(client):
    x = (np.arange(256) + 0.5) / 256
    rng = np.random.default_rng(0)
    # in-distribution: none of the first 100 validation ICs may be flagged
    assert not any(_flag(client, VAL[i, 0])["is_ood"] for i in range(100))
    # clearly out-of-distribution families must be flagged
    assert _flag(client, 3 * np.asarray(VAL[0, 0]))["is_ood"] or np.abs(3 * np.asarray(VAL[0, 0])).max() <= 3.35
    assert _flag(client, 5.0 * np.sin(2 * np.pi * x))["is_ood"]                     # too tall
    assert _flag(client, np.sin(2 * np.pi * x) + 3.0)["is_ood"]                     # mean beyond range
    assert _flag(client, 0.3 * rng.standard_normal(256))["is_ood"]                  # white noise
    assert _flag(client, np.sin(2 * np.pi * 40 * x))["is_ood"]                      # very wiggly
    assert not _flag(client, np.sin(2 * np.pi * x))["is_ood"]                       # plain sine is fine


def test_unknown_route_404(client):
    assert client.get("/nope").status_code == 404


def test_api_root_lists_endpoints(client):
    j = client.get("/api").json()
    assert j["service"] == "burgers-fno-service" and "POST /predict" in j["endpoints"]
