"""Read-only inspection of the PDEBench Burgers HDF5 file. Reads in chunks, never the whole file.

Usage (repo root): python scripts/inspect_h5.py data/raw/1D_Burgers_Sols_Nu0.01.hdf5
Writes reports/inspect_summary.json and reports/samples.png.
"""
import json
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
CH = 100  # samples per chunk: 100*201*1024*4 B = 82 MB
out = {}

with h5py.File(path, "r") as f:
    print("top-level attrs:", dict(f.attrs))
    for k, v in f.items():
        print(f"key={k!r} type={type(v).__name__} shape={getattr(v, 'shape', None)} "
              f"dtype={getattr(v, 'dtype', None)} chunks={getattr(v, 'chunks', None)} "
              f"compression={getattr(v, 'compression', None)} attrs={dict(v.attrs)}")
    T = f["tensor"]
    N, nt, nx = T.shape
    out["shape"] = [N, nt, nx]
    out["dtype"] = str(T.dtype)
    x = f["x-coordinate"][:] if "x-coordinate" in f else None
    t = f["t-coordinate"][:] if "t-coordinate" in f else None
    if x is not None:
        print(f"x: n={len(x)} min={x.min():.6f} max={x.max():.6f} dx={np.diff(x).mean():.6e} "
              f"dx_std={np.diff(x).std():.2e}")
        out["x"] = [float(x.min()), float(x.max()), float(np.diff(x).mean())]
    if t is not None:
        print(f"t: n={len(t)} min={t.min():.6f} max={t.max():.6f} dt={np.diff(t).mean():.6e}")
        out["t"] = [float(t.min()), float(t.max()), float(np.diff(t).mean())]

    # Full scan in chunks: NaN/inf, global range, per-time max|u|, mass drift, periodic seam.
    nan = inf = 0
    gmin, gmax = np.inf, -np.inf
    maxabs_t = np.zeros(nt)
    mass_drift = []  # max_t |sum(u_t)-sum(u_0)| * dx   (per sample)
    ic_max = []
    seam = []  # |u[0]-u[-1]| at t=0 per sample, to see if the last point repeats the first
    dx = float(np.diff(x).mean()) if x is not None else 1.0
    for s in range(0, N, CH):
        a = T[s:s + CH]
        nan += int(np.isnan(a).sum())
        inf += int(np.isinf(a).sum())
        gmin, gmax = min(gmin, float(a.min())), max(gmax, float(a.max()))
        maxabs_t = np.maximum(maxabs_t, np.abs(a).max(axis=(0, 2)))
        m = a.sum(axis=2) * dx
        mass_drift.append(np.abs(m - m[:, :1]).max(axis=1))
        ic_max.append(np.abs(a[:, 0]).max(axis=1))
        seam.append(np.abs(a[:, 0, 0] - a[:, 0, -1]))
        if (s // CH) % 20 == 0:
            print(f"  scanned {min(s + CH, N)}/{N}", flush=True)
    mass_drift = np.concatenate(mass_drift)
    ic_max = np.concatenate(ic_max)
    seam = np.concatenate(seam)
    out.update(nan=nan, inf=inf, min=gmin, max=gmax,
               maxabs_u_t0=float(maxabs_t[0]), maxabs_u_tend=float(maxabs_t[-1]),
               mass_drift_max=float(mass_drift.max()), mass_drift_median=float(np.median(mass_drift)),
               ic_maxabs_quantiles={q: float(np.quantile(ic_max, q)) for q in (0, .01, .5, .99, 1)},
               seam_t0_median=float(np.median(seam)), seam_t0_max=float(seam.max()))
    print(json.dumps(out, indent=1))

    # a few samples to disk
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(N, 4, replace=False).tolist())
    fig, ax = plt.subplots(2, 4, figsize=(16, 6))
    for j, i in enumerate(idx):
        u = T[i]
        im = ax[0, j].imshow(u, aspect="auto", origin="lower", cmap="RdBu_r",
                             extent=[x[0], x[-1], t[0], t[-1]] if x is not None else None)
        ax[0, j].set_title(f"sample {i}: u(x,t)"); ax[0, j].set_xlabel("x"); ax[0, j].set_ylabel("t")
        fig.colorbar(im, ax=ax[0, j])
        for tt in (0, nt // 4, nt // 2, nt - 1):
            ax[1, j].plot(x, u[tt], label=f"t={t[tt]:.2f}")
        ax[1, j].legend(fontsize=7); ax[1, j].set_xlabel("x")
    fig.tight_layout(); fig.savefig("reports/samples.png", dpi=90)
    out["plotted_indices"] = idx

json.dump(out, open("reports/inspect_summary.json", "w"), indent=1)
print("saved reports/samples.png and reports/inspect_summary.json")
