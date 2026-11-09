"""Sample-level splits and lazy HDF5 access for the PDEBench Burgers file.

Assumes (to be confirmed by scripts/inspect_h5.py) that the file has a dataset "tensor" of shape
(n_samples, n_t, n_x). Nothing here loads the whole file: reads are by sorted sample index, in chunks.
"""
import json
import os

import h5py
import numpy as np


def make_split(n_total, seed=0, n_test=1000, n_val=500):
    """Split SAMPLE indices with a fixed seed.

    One permutation of all samples; test is the first n_test, val the next n_val, train the rest.
    Taking a smaller train set later means taking a prefix of `train`, so test and val never change
    and never overlap with train. Splitting by sample (not by time step) is what prevents leakage
    between the splits of the same trajectory.
    """
    perm = np.random.default_rng(seed).permutation(n_total)
    split = {"seed": seed, "n_total": n_total,
             "test": np.sort(perm[:n_test]).tolist(),
             "val": np.sort(perm[n_test:n_test + n_val]).tolist(),
             "train_order": perm[n_test + n_val:].tolist()}   # prefix = smaller train sets
    assert not (set(split["test"]) & set(split["val"]))
    assert not (set(split["test"]) | set(split["val"])) & set(split["train_order"])
    return split


def save_split(split, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f)


def load_split(path):
    with open(path) as f:
        return json.load(f)


def read_samples(h5_path, idx, x_stride=1, t_stride=1, chunk=100, dtype=np.float32):
    """Read samples `idx` (any order) -> array (len(idx), ceil(nt/t_stride), ceil(nx/x_stride)).

    h5py needs increasing indices, so we sort, read in chunks, then restore the requested order.
    Strided subsampling is applied while reading to keep RAM small.
    """
    idx = np.asarray(idx)
    order = np.argsort(idx)
    s_idx = idx[order]
    parts = []
    with h5py.File(h5_path, "r") as f:
        T = f["tensor"]
        for i in range(0, len(s_idx), chunk):
            block = s_idx[i:i + chunk]
            parts.append(T[block.tolist(), ::t_stride, ::x_stride].astype(dtype))
    out = np.concatenate(parts, axis=0)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return out[inv]
