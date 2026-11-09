"""Spot check on 3 VALIDATION samples: FNO vs stored truth vs our own solver (src/solver.py, 256 cells).

Equation convention found in scripts/check_solver.py: u_t + (u^2/2)_x = (0.01/pi) u_xx on [0,1), periodic.
Reports relative L2 (whole trajectory, and at t = 0.5, 1.0, 2.0) of: FNO vs truth, solver vs truth, FNO vs solver,
the FNO error at t=0 (how well it reproduces its own input), and the mass drift of each.
Saves reports/spotcheck.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import FNO1d  # noqa: E402
from src.solver import solve, mass  # noqa: E402

ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/fno_s1/best.pt"
val = np.load("data/processed/val.npy").astype(np.float64)
split = json.load(open("splits/burgers_nu0.01_seed0.json"))
sel = [0, 1, 2]                      # first three val rows (sorted ids), NOT chosen by looking at errors
ids = [split["val"][i] for i in sel]
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
c = ck["args"]
m = FNO1d(201, c["modes"], c["width"], c["layers"]); m.load_state_dict(ck["model"]); m.eval()
truth = val[sel]
with torch.no_grad():
    fno = m(torch.from_numpy(val[sel, 0]).float()).numpy().astype(np.float64)
t = np.arange(201) * 0.01
sol = solve(truth[:, 0], 0.01 / np.pi, t, L=1.0, a=1.0)


def rel(a, b, axis):
    return np.sqrt(((a - b) ** 2).sum(axis=axis) / (b ** 2).sum(axis=axis))


print("checkpoint:", ckpt, "| val rows", sel, "(sample ids in the raw file:", ids, ")")
for k in range(3):
    row = lambda a, b: (rel(a, b, (0, 1)), rel(a, b, 1)[[0, 50, 100, 200]])
    f_t, f_t_k = row(fno[k], truth[k]); s_t, s_t_k = row(sol[k], truth[k]); f_s, f_s_k = row(fno[k], sol[k])
    print(f"-- sample {ids[k]}")
    print(f"   FNO vs truth    whole {f_t:.4f} | t=0,0.5,1,2: {np.round(f_t_k, 4)}")
    print(f"   solver vs truth whole {s_t:.4f} | t=0,0.5,1,2: {np.round(s_t_k, 4)}")
    print(f"   FNO vs solver   whole {f_s:.4f} | t=0,0.5,1,2: {np.round(f_s_k, 4)}")
    md = lambda u: np.abs(mass(u) - mass(u)[0]).max()
    print(f"   mass drift max_t|M(t)-M(0)|: truth {md(truth[k]):.2e} | solver {md(sol[k]):.2e} | FNO {md(fno[k]):.2e}")

fig, ax = plt.subplots(3, 4, figsize=(16, 9))
for k in range(3):
    vmax = np.abs(truth[k]).max()
    for j, (name, arr) in enumerate([("truth", truth[k]), ("FNO", fno[k]), ("solver", sol[k])]):
        im = ax[k, j].imshow(arr, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[0, 1, 0, 2])
        ax[k, j].set_title(f"sample {ids[k]}: {name}")
    im = ax[k, 3].imshow(np.abs(fno[k] - truth[k]), aspect="auto", origin="lower", cmap="magma", extent=[0, 1, 0, 2])
    ax[k, 3].set_title("|FNO - truth|"); fig.colorbar(im, ax=ax[k, 3])
fig.tight_layout(); fig.savefig("reports/spotcheck.png", dpi=80)
print("saved reports/spotcheck.png")
