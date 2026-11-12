"""Evaluate the M7 FluxStepper against the deployed FNO (and optionally an FNO retrained on the same data).

  python scripts/m7_eval.py --m7 checkpoints/m7_s0/best.pt [--fno-aug checkpoints/m7_fno_aug/best.pt] [--out reports/m7_eval_val.json]
  python scripts/m7_eval.py ... --test --final      # adds the PDEBench TEST set (one final evaluation only; refused without --final)

Metric (same as every earlier evaluation): relative L2 over (t, x) per sample, mean over samples. Truth for the generated sets = our solver (256 cells).
Sets: PDEBench validation; held-out waves of the TRAIN styles; four UNSEEN styles that were never used for training or model selection.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference import load_model, predict  # noqa: E402
from src.m7_model import FluxStepper  # noqa: E402
from src.models import FNO1d  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--m7", required=True)
ap.add_argument("--fno-aug", default="")
ap.add_argument("--out", default="reports/m7_eval_val.json")
ap.add_argument("--test", action="store_true")
ap.add_argument("--final", action="store_true")
a = ap.parse_args()
if a.test and not a.final:
    sys.exit("Refusing to open the PDEBench test set without --final.")
torch.set_num_threads(4)

sets = {"PDEBench validation (500)": "data/processed/val.npy", "held-out, TRAIN styles (500)": "data/m7/heldout_style.npy"}
for s in ("gp", "saw", "fm", "packet"):
    sets[f"UNSEEN {s} (200)"] = f"data/m7/unseen_{s}.npy"
if a.test:
    sets["PDEBench TEST (1000)"] = "data/processed/test.npy"

ck = torch.load(a.m7, map_location="cpu", weights_only=False)
m7 = FluxStepper(ck["args"]["ch"], ck["args"]["layers"])
m7.load_state_dict(ck["model"])
m7.eval()
fno, _ = load_model("model/fno_s1.pt")
fno_aug = None
if a.fno_aug:
    c2 = torch.load(a.fno_aug, map_location="cpu", weights_only=False)["args"]
    fno_aug = FNO1d(201, c2["modes"], c2["width"], c2["layers"])
    fno_aug.load_state_dict(torch.load(a.fno_aug, map_location="cpu", weights_only=False)["model"])
    fno_aug.eval()


def rel(p, t):
    return np.sqrt(((p - t) ** 2).sum((1, 2)) / (t ** 2).sum((1, 2)))


def run(name, u0):
    with torch.no_grad():
        if name == "FNO seed 1 (deployed)":
            return predict(fno, u0).astype(np.float64)
        if name == "FNO retrained on the same data":
            return np.concatenate([fno_aug(torch.tensor(u0[i:i + 100], dtype=torch.float32)).numpy() for i in range(0, len(u0), 100)]).astype(np.float64)
        clamp = name.endswith("(clamped)")
        return np.concatenate([m7(torch.tensor(u0[i:i + 100], dtype=torch.float32), 200, clamp=clamp).numpy() for i in range(0, len(u0), 100)]).astype(np.float64)


models = ["FNO seed 1 (deployed)", "M7 FluxStepper", "M7 FluxStepper (clamped)"] + (["FNO retrained on the same data"] if fno_aug else [])
res, pooled = {}, {m: [] for m in models}
print(f"{'set':32s}" + "".join(f"{m[:26]:>28s}" for m in models))
for sname, path in sets.items():
    truth = np.load(path).astype(np.float64)
    row = {}
    for m in models:
        p = run(m, truth[:, 0])
        err = rel(p, truth)
        err = np.where(np.isfinite(err), err, np.inf)
        mass = np.abs(p.sum(-1) / 256 - (truth[:, 0].sum(-1) / 256)[:, None]).max(1)
        row[m] = {"mean": float(np.mean(err)), "median": float(np.median(err)), "p90": float(np.quantile(err, .9)), "n_nonfinite": int((~np.isfinite(err)).sum()),
                  "mass_drift_mean": float(np.nanmean(np.where(np.isfinite(mass), mass, np.nan)))}
        if sname.startswith("UNSEEN"):
            pooled[m].append(err)
    res[sname] = row
    print(f"{sname:32s}" + "".join(f"{row[m]['mean']:>14.4f} (p90 {row[m]['p90']:.3f}) " for m in models))
res["UNSEEN pooled (800)"] = {m: {"mean": float(np.mean(np.concatenate(pooled[m]))), "median": float(np.median(np.concatenate(pooled[m]))),
                                  "p90": float(np.quantile(np.concatenate(pooled[m]), .9))} for m in models}
print(f"{'UNSEEN pooled (800)':32s}" + "".join(f"{res['UNSEEN pooled (800)'][m]['mean']:>14.4f} (p90 {res['UNSEEN pooled (800)'][m]['p90']:.3f}) " for m in models))

# speed on one CPU process (4 threads)
u0 = np.load("data/m7/unseen_gp.npy")[:32, 0].astype(np.float32)
timing = {}
for m in models:
    run(m, u0[:2])
    t0 = time.perf_counter()
    for i in range(5):
        run(m, u0[i:i + 1])
    t1 = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    run(m, u0)
    timing[m] = {"one_wave_ms": 1000 * t1, "batch32_ms_per_wave": 1000 * (time.perf_counter() - t0) / 32}
res["timing_cpu_4_threads"] = timing
print("\nCPU time per wave (4 threads): " + ", ".join(f"{m}: {v['one_wave_ms']:.1f} ms (batch of 32: {v['batch32_ms_per_wave']:.1f} ms each)" for m, v in timing.items()))
res["m7_checkpoint"] = {"path": a.m7, "iter": ck.get("iter"), "val_pde_rollout": ck.get("val_pde"), "val_style_rollout": ck.get("val_style")}
json.dump(res, open(a.out, "w"), indent=1)
print("saved", a.out)
