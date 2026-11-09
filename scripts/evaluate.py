"""Evaluate a checkpoint (or a trivial baseline) on val or test.

  python scripts/evaluate.py --ckpt checkpoints/fno_main/best.pt --model fno --split val
  python scripts/evaluate.py --split test --final ...      # test requires the explicit --final flag

Metrics (all relative L2 = ||pred-true||_2 / ||true||_2, per sample, then averaged over samples):
  whole : norm over (t, x) of the full trajectory; also median and 95th percentile over samples
  per_k : norm over x at one time index k (time t = 0.01 k); mean over samples
  near / away from shocks: at every (sample, k) a grid point is a "shock point" if the TRUE field has a
        compressive gradient du/dx < -SHOCK_GRAD (central differences on the 256-point grid, periodic);
        "near" = within +-NEAR points of a shock point. Error is ||.||_2 over the selected points,
        divided by ||true||_2 over the same points, aggregated over all (sample, k) with k >= 1 by summing
        squared numerators and denominators (so it is a pooled ratio, not a mean of ratios).
  mass  : max_t |M_pred(t) - M_pred(0)| averaged over samples, M = sum(u) dx; and the same for the truth.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import FNO1d, UNet1d, rel_l2  # noqa: E402

SHOCK_GRAD, NEAR = 20.0, 5
ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["val", "test"], default="val")
ap.add_argument("--final", action="store_true", help="required to open the test split")
ap.add_argument("--model", choices=["fno", "unet", "copy_ic", "const_icmean"], required=True)
ap.add_argument("--ckpt", default="")
ap.add_argument("--out", default="")
ap.add_argument("--label", default="")
a = ap.parse_args()
if a.split == "test" and not a.final:
    sys.exit("Refusing to open the test split without --final.")

dev = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.from_numpy(np.load(f"data/processed/{a.split}.npy")).to(dev)        # (N, nt, nx)
N, nt, nx = x.shape
dx = 1.0 / nx

if a.model in ("fno", "unet"):
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    c = ck["args"]
    m = FNO1d(nt, c["modes"], c["width"], c["layers"]) if a.model == "fno" else UNet1d(nt, c["base"])
    m.load_state_dict(ck["model"]); m.to(dev).eval()
    with torch.no_grad():
        pred = torch.cat([m(x[i:i + 100, 0]) for i in range(0, N, 100)])
elif a.model == "copy_ic":
    pred = x[:, :1].expand_as(x).clone()
else:
    pred = x[:, :1].mean(dim=2, keepdim=True).expand_as(x).clone()

whole = rel_l2(pred, x, (1, 2))
rel_k = rel_l2(pred, x, (2,))                                   # (N, nt)
ks = [0, 1, 5, 10, 25, 50, 100, 150, 200]
ux = (torch.roll(x, -1, dims=2) - torch.roll(x, 1, dims=2)) / (2 * dx)
shock = ux < -SHOCK_GRAD
near = shock.clone()
for s in range(1, NEAR + 1):
    near |= torch.roll(shock, s, dims=2) | torch.roll(shock, -s, dims=2)
err2, tru2 = (pred - x) ** 2, x ** 2
sel = torch.zeros_like(near)
sel[:, 1:] = True


def pooled(mask):
    mk = mask & sel
    return float(torch.sqrt(err2[mk].sum() / tru2[mk].sum())), float(mk.float().mean())


r_near, f_near = pooled(near)
r_away, f_away = pooled(~near)
mass_p = pred.sum(dim=2) * dx
mass_t = x.sum(dim=2) * dx
res = {
    "label": a.label, "model": a.model, "ckpt": a.ckpt, "split": a.split, "n": N,
    "method": "relative L2 per sample, mean over samples; see module docstring",
    "whole_mean": whole.mean().item(), "whole_median": whole.median().item(),
    "whole_p95": torch.quantile(whole, 0.95).item(),
    "per_k": {str(k): rel_k[:, k].mean().item() for k in ks},
    "mean_k1_200": rel_k[:, 1:].mean().item(),
    "shock_def": f"du/dx < -{SHOCK_GRAD} (true field), near = within +-{NEAR} grid points",
    "near_shock": {"pooled_relL2": r_near, "fraction_of_points": f_near},
    "away_from_shock": {"pooled_relL2": r_away, "fraction_of_points": f_away},
    "mass_drift_pred": (mass_p - mass_p[:, :1]).abs().max(dim=1).values.mean().item(),
    "mass_drift_true": (mass_t - mass_t[:, :1]).abs().max(dim=1).values.mean().item(),
}
print(json.dumps(res, indent=1))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
