"""Compare src/solver.py with the stored PDEBench solutions on 3 VALIDATION samples (never test).

Phase A (fast, 256 cells): try the equation conventions  u_t + a*(u^2/2)_x = nu_eff * u_xx
  for a in {1, 2} and nu_eff in {0.01, 0.01/pi}, to see which one the data follows.
Phase B (1024 cells = the data's own grid): the best convention, reporting relative L2 error
  per sample and at selected times, mass drift of the data vs the solver, and the error location.

Relative L2 here: ||pred - true||_2 / ||true||_2 over x (at one time) or over (x, t) (whole trajectory).
Frame k of the stored tensor is assumed to be t = 0.01 * k (frame 0 = the initial condition).
"""
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.solver import solve, mass  # noqa: E402

RAW = "data/raw/1D_Burgers_Sols_Nu0.01.hdf5"
split = json.load(open("splits/burgers_nu0.01_seed0.json"))
ids = sorted(split["val"])[:3]
print("validation sample ids used:", ids, flush=True)
with h5py.File(RAW, "r") as f:
    T = f["tensor"]
    full = np.stack([T[i] for i in ids]).astype(np.float64)   # (3, 201, 1024)
t_out = np.arange(201) * 0.01


def rel(a, b, axis):
    return np.sqrt(((a - b) ** 2).sum(axis=axis) / (b ** 2).sum(axis=axis))


print("\n== Phase A: convention search, 256 cells (stride-4 point values), 3 samples")
ref256 = full[:, :, ::4]
for a in (1.0, 2.0):
    for nu_name, nu in (("0.01", 0.01), ("0.01/pi", 0.01 / np.pi)):
        t0 = time.time()
        out = solve(ref256[:, 0], nu, t_out, L=1.0, a=a)
        e_traj = rel(out, ref256, axis=(1, 2))
        e_t = rel(out, ref256, axis=2)
        print(f"a={a:.0f} nu_eff={nu_name:8s} traj relL2 per sample {np.round(e_traj, 4)} | "
              f"mean at t=0.5/1/2: {e_t[:, 50].mean():.4f} {e_t[:, 100].mean():.4f} {e_t[:, 200].mean():.4f} "
              f"| {time.time() - t0:.0f}s", flush=True)
    best_hint = None

print("\n== Phase B is run by the caller with the chosen convention (see argv)")
if len(sys.argv) == 3:
    a, nu = float(sys.argv[1]), float(sys.argv[2])
    print(f"== Phase B: a={a} nu_eff={nu}, 1024 cells", flush=True)
    t0 = time.time()
    out = solve(full[:, 0], nu, t_out, L=1.0, a=a)
    print(f"solved in {time.time() - t0:.0f}s")
    for k, i in enumerate(ids):
        e = rel(out[k], full[k], axis=1)
        d = np.abs(out[k] - full[k])
        md, ms = mass(full[k]), mass(out[k])
        print(f"sample {i}: traj relL2={rel(out[k], full[k], axis=(0, 1)):.5f} | "
              f"relL2 at t=0.5/1.0/2.0 = {e[50]:.5f} {e[100]:.5f} {e[200]:.5f} | max rel over t = {e.max():.5f}")
        print(f"   mass drift max_t|M(t)-M(0)|: data={np.abs(md - md[0]).max():.3e}  solver={np.abs(ms - ms[0]).max():.3e}"
              f" | max abs error {d.max():.4f} at (t={t_out[np.unravel_index(d.argmax(), d.shape)[0]]:.2f}, "
              f"x={(np.unravel_index(d.argmax(), d.shape)[1] + 0.5) / 1024:.3f})")
