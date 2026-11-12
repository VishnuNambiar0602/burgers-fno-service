"""M7 data: solve generated waves with OUR solver (src/solver.py, 256 cells, equation convention verified against PDEBench).
Resumable: each chunk of 64 waves is a separate file and existing chunks are skipped; every chunk uses its own seed.

  python scripts/m7_gen_data.py            -> data/m7/train/*.npy (10000), data/m7/heldout_style.npy (500), data/m7/unseen_<style>.npy (200 each)
Labels are our solver's, not PDEBench's: part of the M7 training data is self-generated and the README must say so.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import wavegen  # noqa: E402
from src.inference import NU_EFF  # noqa: E402
from src.solver import solve  # noqa: E402

OUT = os.path.join("data", "m7")
t = np.arange(201) * 0.01
os.makedirs(os.path.join(OUT, "train"), exist_ok=True)
CHUNK = 64


def log(m):
    print(time.strftime("%F %T"), m, flush=True)


def solve_batch(u0):
    return solve(u0, NU_EFF, t, L=1.0, a=1.0).astype(np.float32)


t_all = time.time()
# ---- training waves: 10,000 from the TRAIN styles
n_train = 10000
for ci in range(int(np.ceil(n_train / CHUNK))):
    f = os.path.join(OUT, "train", f"chunk_{ci:04d}.npy")
    if os.path.exists(f):
        continue
    rng = np.random.default_rng(1000 + ci)
    n = min(CHUNK, n_train - ci * CHUNK)
    u0, _ = wavegen.train_mix(rng, n)
    t0 = time.time()
    sol = solve_batch(u0)
    np.save(f + ".tmp.npy", sol)
    os.replace(f + ".tmp.npy", f)
    if ci % 10 == 0:
        log(f"train chunk {ci + 1}/{int(np.ceil(n_train / CHUNK))}: {time.time() - t0:.1f}s, finite={np.isfinite(sol).all()}, max|u|={np.abs(sol).max():.2f}")

# ---- held-out waves in the TRAIN styles (different seeds) and 4 UNSEEN styles
sets = {"heldout_style": lambda rng: wavegen.train_mix(rng, 500)[0]}
for name, fn in wavegen.UNSEEN.items():
    sets[f"unseen_{name}"] = (lambda fn: (lambda rng: fn(rng, 200)))(fn)
for k, (name, make) in enumerate(sets.items()):
    f = os.path.join(OUT, f"{name}.npy")
    if os.path.exists(f):
        continue
    u0 = make(np.random.default_rng(900000 + k))
    sol = np.concatenate([solve_batch(u0[i:i + CHUNK]) for i in range(0, len(u0), CHUNK)])
    np.save(f + ".tmp.npy", sol)
    os.replace(f + ".tmp.npy", f)
    log(f"{name}: {sol.shape}, max|u|={np.abs(sol).max():.2f}, finite={np.isfinite(sol).all()}")
log(f"DONE in {(time.time() - t_all) / 60:.1f} min")
