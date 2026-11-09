"""Independent validation of the M2 results (VALIDATION data and synthetic inputs only; test is never opened).

A. metric re-implemented in numpy float64 vs the number evaluate.py reported
B. processed arrays vs the raw HDF5 rows they claim to come from (train/val/test index mapping, no evaluation)
C. sensitivity: wrong-IC control, 1% input noise, time off-by-one alignment
D. translation: periodic shift of the IC must shift the solution (exact PDE symmetry)
E. out-of-distribution inputs, truth from src/solver.py (256 cells, convention found earlier)
F. zero-shot resolution change (512 points)
Writes reports/validate_m2.json.
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import FNO1d, UNet1d  # noqa: E402
from src.solver import solve  # noqa: E402

NU = 0.01 / np.pi
t_out = np.arange(201) * 0.01
RAW = "data/raw/1D_Burgers_Sols_Nu0.01.hdf5"
res = {}


def load(name):
    ck = torch.load(f"checkpoints/{name}/best.pt", map_location="cpu", weights_only=False)
    c = ck["args"]
    m = FNO1d(201, c["modes"], c["width"], c["layers"]) if name.startswith("fno") else UNet1d(201, c["base"])
    m.load_state_dict(ck["model"]); return m.eval()


models = {"fno_s1": load("fno_s1"), "unet_s1": load("unet_s1")}


def predict(m, u0):
    with torch.no_grad():
        return m(torch.from_numpy(np.asarray(u0)).float()).numpy().astype(np.float64)


def rel(a, b, axis=(1, 2)):
    return np.sqrt(((a - b) ** 2).sum(axis=axis) / (b ** 2).sum(axis=axis))


val = np.load("data/processed/val.npy").astype(np.float64)
split = json.load(open("splits/burgers_nu0.01_seed0.json"))

# ---------------- A. metric
reported = json.load(open("reports/eval_val_fno_s1.json"))["whole_mean"]
mine = rel(predict(models["fno_s1"], val[:, 0]), val).mean()
res["A_metric"] = {"evaluate.py": reported, "numpy_float64": float(mine), "abs_diff": abs(reported - float(mine))}
print(f"A. whole-trajectory val relL2 fno_s1: evaluate.py {reported:.6f} | independent numpy float64 {mine:.6f}")
print(f"   sanity: pred=truth -> {rel(val, val).mean():.1f}; pred=0 -> {rel(np.zeros_like(val), val).mean():.3f}")

# ---------------- B. raw-file index mapping
ok = True
with h5py.File(RAW, "r") as f:
    T = f["tensor"]
    for name, ids in [("train", split["train_order"][:5]), ("val", split["val"][:5]), ("test", split["test"][:3])]:
        arr = np.load(f"data/processed/{name}.npy", mmap_mode="r")[:len(ids)]
        raw = np.stack([T[i][::1, ::4] for i in ids])
        same = np.array_equal(np.asarray(arr), raw.astype(np.float32))
        ok &= same
        print(f"B. {name}.npy rows 0..{len(ids) - 1} equal raw HDF5 rows {ids[:3]}...: {same}")
res["B_index_mapping_ok"] = bool(ok)

# ---------------- C. sensitivity
out = {}
for n, m in models.items():
    p = predict(m, val[:200, 0])
    wrong = rel(predict(m, np.roll(val[:200, 0], 1, axis=0)), val[:200]).mean()
    rng = np.random.default_rng(0)
    noisy = val[:200, 0] + 0.01 * val[:200, 0].std(axis=1, keepdims=True) * rng.standard_normal((200, 256))
    pn = predict(m, noisy)
    d_out = np.sqrt(((pn - p) ** 2).sum((1, 2)) / (p ** 2).sum((1, 2))).mean()
    shifts = {d: float(np.sqrt(((p[:, max(0, -d):201 - max(0, d)] - val[:200, max(0, d):201 - max(0, -d)]) ** 2).sum((1, 2)) /
                               (val[:200, max(0, d):201 - max(0, -d)] ** 2).sum((1, 2))).mean()) for d in (-2, -1, 0, 1, 2)}
    out[n] = {"wrong_IC_relL2": float(wrong), "output_change_for_1pct_noise_in_input": float(d_out), "time_offset_relL2": shifts}
    print(f"C. {n}: wrong-IC control {wrong:.3f} (should be large) | 1% input noise changes output by {d_out:.4f} | "
          f"time offset d->relL2 {({k: round(v, 4) for k, v in shifts.items()})} (min must be at 0)")
res["C_sensitivity"] = out

# ---------------- D. translation
out = {}
for n, m in models.items():
    row = {}
    for s in (0, 17, 64, 128):
        u0, tr = np.roll(val[:200, 0], s, axis=1), np.roll(val[:200], s, axis=2)
        row[s] = float(rel(predict(m, u0), tr).mean())
    out[n] = row
    print(f"D. {n}: relL2 on periodically shifted val ICs (shift in grid cells): {({k: round(v, 4) for k, v in row.items()})}")
res["D_translation"] = out

# ---------------- E. OOD families
x = (np.arange(256) + 0.5) / 256
rng = np.random.default_rng(1)
base = val[:50, 0]
fam = {
    "val_IC x1 (control, solver truth)": base,
    "amplitude x0.5": 0.5 * base, "amplitude x2": 2 * base, "amplitude x3": 3 * base,
    "mean offset +1.5": base + 1.5, "mean offset +3.0": base + 3.0,
    "gaussian bump": np.exp(-((x - 0.5) / 0.08) ** 2)[None] * np.ones((4, 1)),
    "square wave": np.where((x > 0.25) & (x < 0.75), 1.0, -1.0)[None] * np.ones((4, 1)),
    "white noise sd 0.3": 0.3 * rng.standard_normal((8, 256)),
    "sine k=40 (random phase)": np.sin(2 * np.pi * 40 * x[None] + rng.uniform(0, 6.28, (8, 1))),
    "sine k=100 (random phase)": np.sin(2 * np.pi * 100 * x[None] + rng.uniform(0, 6.28, (8, 1))),
    "constant 0.7": 0.7 * np.ones((2, 256)),
}
out = {}
print("E. OOD (FNO | U-Net relL2 vs solver truth, whole trajectory; max|u0| of the input in brackets; train max|u0| <= 3.35)")
for name, u0 in fam.items():
    truth = solve(u0.astype(np.float64), NU, t_out, L=1.0, a=1.0)
    e = {n: float(rel(predict(m, u0), truth).mean()) for n, m in models.items()}
    out[name] = {**e, "max_abs_u0": float(np.abs(u0).max())}
    print(f"   {name:36s} FNO {e['fno_s1']:.4f} | U-Net {e['unet_s1']:.4f}  [{np.abs(u0).max():.2f}]", flush=True)
res["E_ood"] = out

# ---------------- F. resolution
with h5py.File(RAW, "r") as f:
    T = f["tensor"]
    ids = sorted(split["val"])[:50]
    full = np.stack([T[i][:, ::2] for i in ids]).astype(np.float64)    # (50, 201, 512)
out = {}
for n, m in models.items():
    try:
        out[n] = float(rel(predict(m, full[:, 0]), full).mean())
    except Exception as ex:  # report, do not hide
        out[n] = f"failed: {ex!r}"
print("F. 512-point input vs stored truth (trained on 256):", out)
res["F_resolution_512"] = out
json.dump(res, open("reports/validate_m2.json", "w"), indent=1, default=str)
