"""Create the split file and the processed (subsampled) arrays from the raw HDF5.

Usage (repo root):
  python scripts/make_subset.py --n-train 4000 --x-stride 4 --t-stride 1
Writes splits/burgers_nu0.01_seed0.json (small, committed) and
data/processed/{train,val,test}.npy (float32, git-ignored). The TEST file is written but must not
be opened for any model decision until final evaluation.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import make_split, save_split, read_samples  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw/1D_Burgers_Sols_Nu0.01.hdf5")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-total", type=int, default=10000)
ap.add_argument("--n-test", type=int, default=1000)
ap.add_argument("--n-val", type=int, default=500)
ap.add_argument("--n-train", type=int, default=4000)
ap.add_argument("--x-stride", type=int, default=4)
ap.add_argument("--t-stride", type=int, default=1)
a = ap.parse_args()

split = make_split(a.n_total, a.seed, a.n_test, a.n_val)
split["n_train_used"] = a.n_train
split["x_stride"], split["t_stride"] = a.x_stride, a.t_stride
save_split(split, f"splits/burgers_nu0.01_seed{a.seed}.json")

os.makedirs("data/processed", exist_ok=True)
for name, idx in [("train", split["train_order"][:a.n_train]), ("val", split["val"]), ("test", split["test"])]:
    arr = read_samples(a.raw, idx, a.x_stride, a.t_stride)
    np.save(f"data/processed/{name}.npy", arr)
    print(f"{name}: {arr.shape} {arr.dtype} {arr.nbytes/1e6:.0f} MB  min={arr.min():.4f} max={arr.max():.4f} "
          f"nan={int(np.isnan(arr).sum())}", flush=True)
