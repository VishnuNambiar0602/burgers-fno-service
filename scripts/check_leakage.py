"""Duplicate / near-duplicate initial-condition check between VALIDATION and TRAIN (test is not touched here).

For each val IC, the relative distance ||a-b||/||a|| to its nearest train IC, compared with the typical
nearest-neighbour distance inside train. If val ICs were near-copies of train ICs, the minimum would be ~0.
(The same check is to be run on test only at final evaluation.)
"""
import numpy as np

tr = np.load("data/processed/train.npy", mmap_mode="r")[:, 0, :].astype(np.float64)
va = np.load("data/processed/val.npy", mmap_mode="r")[:, 0, :].astype(np.float64)


def nn_rel(a, b, same=False):
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    if same:
        np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.maximum(d2.min(axis=1), 0)) / np.sqrt((a ** 2).sum(1))


v = nn_rel(va, tr)
t = nn_rel(tr, tr, same=True)
q = lambda x: {k: round(float(np.quantile(x, k)), 4) for k in (0, 0.01, 0.5)}
print("val->train nearest IC rel. distance quantiles (min, 1%, median):", q(v))
print("train->train nearest IC rel. distance quantiles (min, 1%, median):", q(t))
print("val ICs with rel. distance < 1e-3 to a train IC:", int((v < 1e-3).sum()), "of", len(v))
