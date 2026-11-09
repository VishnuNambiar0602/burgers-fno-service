"""End-to-end check of a RUNNING service (local or deployed) against the offline result.

  python scripts/live_check.py http://127.0.0.1:8000 [--tag local]

Takes validation sample row 7 as the fixed input, calls /health, /model-info and /predict over real HTTP, and compares
the returned FNO solution with (a) the offline prediction of the ORIGINAL training checkpoint and (b) the stored truth.
Writes reports/live_check_<tag>.json. Exit code is non-zero if the comparison fails.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import FNO1d  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("--tag", default="local")
ap.add_argument("--row", type=int, default=7)
a = ap.parse_args()
base = a.base.rstrip("/")


def http(path, body=None, timeout=120):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "accept-encoding": "identity"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw), (time.perf_counter() - t0) * 1000, len(raw)


out = {"base": base, "time": time.strftime("%F %T %z"), "row": a.row}
val = np.load("data/processed/val.npy", mmap_mode="r")
u0 = np.asarray(val[a.row, 0], dtype=np.float64)
truth = np.asarray(val[a.row], dtype=np.float64)

h, ms, _ = http("/health"); out["health"] = h; print("health:", h, f"({ms:.0f} ms)")
info, _, _ = http("/model-info")
local_sha = hashlib.sha256(open("model/fno_s1.pt", "rb").read()).hexdigest()
out["served_sha256"], out["local_sha256"] = info["runtime"]["served_sha256"], local_sha
print("model-info: name", info["name"], "| served sha256 == local file:", info["runtime"]["served_sha256"] == local_sha)

lat = []
for _ in range(6):
    j, ms, nbytes = http("/predict", {"u0": u0.tolist()}); lat.append(ms)
out["predict_http_ms_first"], out["predict_http_ms_median_of_rest"] = lat[0], float(np.median(lat[1:]))
out["response_bytes"] = nbytes
u = np.asarray(j["fno"]["u"], dtype=np.float64)

ck = torch.load("checkpoints/fno_s1/best.pt", map_location="cpu", weights_only=False)
c = ck["args"]
m = FNO1d(201, c["modes"], c["width"], c["layers"]); m.load_state_dict(ck["model"]); m.eval()
with torch.no_grad():
    off = m(torch.tensor(u0, dtype=torch.float32)[None])[0].numpy().astype(np.float64)
rel = lambda p, q: float(np.sqrt(((p - q) ** 2).sum() / (q ** 2).sum()))
out["max_abs_diff_live_vs_offline"] = float(np.abs(u - off).max())
out["relL2_live_vs_offline"] = rel(u, off)
out["relL2_live_vs_truth"] = rel(u, truth)
out["relL2_offline_vs_truth"] = rel(off, truth)
out["ood"] = j["ood"]["is_ood"]
print(f"live vs offline: max|diff| {out['max_abs_diff_live_vs_offline']:.2e} | relL2 {out['relL2_live_vs_offline']:.2e}")
print(f"vs truth: live {out['relL2_live_vs_truth']:.5f} | offline {out['relL2_offline_vs_truth']:.5f} | ood flag {out['ood']}")
print(f"HTTP /predict: first {lat[0]:.0f} ms, median of rest {out['predict_http_ms_median_of_rest']:.0f} ms, {nbytes/1e3:.0f} kB body")

js, ms, _ = http("/predict", {"u0": u0.tolist(), "include_solver": True})
out["solver_http_ms"] = ms
out["solver_seconds_server"] = js["solver"]["seconds"]
out["relL2_solver_vs_truth"] = rel(np.asarray(js["solver"]["u"]), truth)
print(f"with solver: {ms:.0f} ms total, server solver {js['solver']['seconds']:.2f} s, solver vs truth {out['relL2_solver_vs_truth']:.5f}")

ok = out["max_abs_diff_live_vs_offline"] < 2e-6 and out["served_sha256"] == local_sha and out["health"]["status"] == "ok"
out["PASS"] = bool(ok)
json.dump(out, open(f"reports/live_check_{a.tag}.json", "w"), indent=1)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
