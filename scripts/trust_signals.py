"""Can the FNO's own output tell us when it is wrong, without running the solver? And do cheap post-hoc fixes help?
Validation data and synthetic inputs only (the test set is not opened). Truth for synthetic inputs = src/solver.py.

Signals computed from the FNO output alone (all cheap):
  mass   : max_t |M(t) - M(0)| / max(mean|u0|, 0.05)        (the equation conserves mass exactly)
  t0     : relative error between the FNO's frame 0 and the input u0   (the network should reproduce its input)
  resid  : PDE residual of the FNO output by finite differences, relative to the size of the terms
  tail   : share of the output's energy at wavenumbers above 32 (ripples / Gibbs-type noise)
Targets: relative L2 error of the FNO against truth. 'bad' means error > 0.06.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference import NU_EFF, load_model, ood_check, load_json, predict  # noqa: E402
from src.solver import solve  # noqa: E402

model, _ = load_model("model/fno_s1.pt")
stats = load_json("model/ood_stats.json")
t = np.arange(201) * 0.01
x = (np.arange(256) + 0.5) / 256
dx, dt = 1 / 256, 0.01
rng = np.random.default_rng(11)


def rel(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=(-2, -1)) / (b ** 2).sum(axis=(-2, -1)))


def signals(u0, u):                         # u0 (B,nx), u (B,nt,nx)
    M = u.sum(-1) * dx
    mass = np.abs(M - (u0.sum(-1) * dx)[:, None]).max(1) / np.maximum(np.abs(u0).mean(1), 0.05)
    t0 = np.sqrt(((u[:, 0] - u0) ** 2).sum(-1) / (u0 ** 2).sum(-1))
    ut = (u[:, 2:] - u[:, :-2]) / (2 * dt)
    um = u[:, 1:-1]
    fx = (np.roll(um ** 2 / 2, -1, -1) - np.roll(um ** 2 / 2, 1, -1)) / (2 * dx)
    uxx = NU_EFF * (np.roll(um, -1, -1) - 2 * um + np.roll(um, 1, -1)) / dx ** 2
    R = ut + fx - uxx
    resid = np.sqrt((R ** 2).sum((1, 2))) / (np.sqrt((ut ** 2).sum((1, 2))) + np.sqrt((fx ** 2).sum((1, 2))) + np.sqrt((uxx ** 2).sum((1, 2))) + 1e-9)
    P = np.abs(np.fft.rfft(u[:, ::10], axis=-1)) ** 2
    tail = P[..., 33:].sum((1, 2)) / (P[..., 1:].sum((1, 2)) + 1e-12)
    return {"mass": mass, "t0": t0, "resid": resid, "tail": tail}


def auc(score, bad):                         # probability that a bad case scores higher than a good one
    s, b = np.asarray(score), np.asarray(bad)
    if b.all() or (~b).all():
        return float("nan")
    r = s.argsort().argsort() + 1.0
    return float((r[b].sum() - b.sum() * (b.sum() + 1) / 2) / (b.sum() * (~b).sum()))


# ---------------- inputs
val = np.load("data/processed/val.npy").astype(np.float64)
sets = {"validation (stored truth)": (val[:300, 0], val[:300])}
drawn = []
for _ in range(60):
    n = rng.integers(6, 14); px = np.sort(rng.uniform(0, 1, n)); py = rng.uniform(-2, 2, n)
    u = np.interp(x, px, py, period=1.0) + 0.05 * rng.standard_normal(256)
    F = np.fft.rfft(u); F[9:] = 0; drawn.append(np.clip(np.fft.irfft(F, n=256), -3.35, 3.35))
sets["hand-drawn-like (snapped)"] = (np.array(drawn), None)
pre = []
for k in range(40):
    r = np.random.default_rng(k + 1); comps = [(0.3 + 0.9 * r.random(), 1 + int(r.integers(0, 8)), 6.28 * r.random()) for _ in range(1 + int(r.integers(0, 3)))]
    s = min(1.0, 2.4 / sum(a for a, _, _ in comps)); pre.append(sum(a * s * np.sin(2 * np.pi * kk * x + p) for a, kk, p in comps) + (r.random() - 0.5) * 0.8)
sets["random sine mixes"] = (np.array(pre), None)
sets["two waves / square / bump"] = (np.array([0.8 * np.sin(4 * np.pi * x + 0.5) + 0.5 * np.sin(10 * np.pi * x + 1.2), np.where((x > .25) & (x < .75), 1.0, -1.0), np.exp(-((x - .5) / .08) ** 2)]), None)
sets["too tall / noise / wiggly"] = (np.array([5 * np.sin(2 * np.pi * x), 0.3 * rng.standard_normal(256), np.sin(2 * np.pi * 40 * x), 3 * np.asarray(val[3, 0]), val[5, 0] + 3.0]), None)

rows = {}
for name, (u0, truth) in sets.items():
    pr = predict(model, u0).astype(np.float64)
    tr = truth if truth is not None else solve(u0, NU_EFF, t)
    err = rel(pr, tr)
    rows[name] = (u0, pr, tr, err, signals(u0, pr), np.array([ood_check(u, stats)["is_ood"] for u in u0]))
    print(f"{name:28s} n={len(u0):3d}  FNO error median {np.median(err):.3f}  p90 {np.quantile(err, .9):.3f}  max {err.max():.3f}  | frac error>6%: {np.mean(err > .06):.0%}  | current OOD flag: {np.mean(rows[name][5]):.0%}")

# pooled: do the signals rank bad cases above good ones?
err = np.concatenate([r[3] for r in rows.values()]); bad = err > 0.06
sig = {k: np.concatenate([r[4][k] for r in rows.values()]) for k in ("mass", "t0", "resid", "tail")}
cur = np.concatenate([r[5] for r in rows.values()])
print(f"\npooled n={len(err)}, bad (error>6%) = {bad.mean():.0%}.  AUC = chance a bad case scores higher than a good one (0.5 = useless, 1 = perfect)")
vs = rows["validation (stored truth)"][4]
for k in sig:
    thr = np.quantile(vs[k], 0.99)                      # 1% false-alarm threshold on in-distribution validation inputs
    flag = sig[k] > thr
    print(f"  {k:6s} AUC {auc(sig[k], bad):.2f} | threshold (val p99) {thr:.3g} -> flags {flag[bad].mean():.0%} of bad cases, {flag[~bad].mean():.1%} of good ones")
print(f"  current input-range flag: flags {cur[bad].mean():.0%} of bad cases, {cur[~bad].mean():.1%} of good ones")
anyflag = np.zeros_like(bad)
for k in ("mass", "t0", "resid", "tail"):
    anyflag |= sig[k] > np.quantile(vs[k], 0.99)
print(f"  ANY of the four signals (each at its val p99): flags {anyflag[bad].mean():.0%} of bad, {anyflag[~bad].mean():.1%} of good;  plus the current flag: {(anyflag | cur)[bad].mean():.0%} / {(anyflag | cur)[~bad].mean():.1%}")

# ---------------- cheap post-hoc corrections (no retraining)
print("\nPost-hoc fixes on the FNO output (mean relative L2 error):")
print(f"{'input set':28s} {'as is':>8s} {'mass fixed':>11s} {'+ t0 anchored':>14s}")
for name, (u0, pr, tr, err, _, _) in rows.items():
    c1 = pr - pr.mean(-1, keepdims=True) + u0.mean(-1)[:, None, None]
    d0 = (u0 - c1[:, 0])[:, None, :] * np.exp(-t / 0.2)[None, :, None]
    c2 = c1 + d0
    print(f"{name:28s} {err.mean():8.4f} {rel(c1, tr).mean():11.4f} {rel(c2, tr).mean():14.4f}")

# ---------------- the DEPLOYED detector (src/inference.self_check, thresholds from model/ood_stats.json)
from src.inference import self_check  # noqa: E402
print("\nDEPLOYED self-check (validation-p99 thresholds) vs the current input-range flag")
allflag, allbad, allcur = [], [], []
for name, (u0, pr, tr, err, _, cur) in rows.items():
    fl = np.array([self_check(a, b, stats)["flagged"] for a, b in zip(u0, pr)])
    allflag.append(fl); allbad.append(err > 0.06); allcur.append(cur)
    print(f"  {name:28s} self-check flags {fl.mean():4.0%} | input-range flag {cur.mean():4.0%} | either {np.mean(fl | cur):4.0%} | actually bad {np.mean(err > .06):4.0%}")
fl, bd, cu = np.concatenate(allflag), np.concatenate(allbad), np.concatenate(allcur)
print(f"  pooled: bad cases caught: self-check {fl[bd].mean():.0%}, input-range {cu[bd].mean():.0%}, either {(fl | cu)[bd].mean():.0%};  "
      f"good cases wrongly flagged: self-check {fl[~bd].mean():.1%}, either {(fl | cu)[~bd].mean():.1%};  precision of 'either': {bd[fl | cu].mean():.0%}")
for nm in ("two waves / square / bump", "too tall / noise / wiggly"):
    u0, pr, tr, err, _, cur = rows[nm]
    print(f"  {nm}: self-check flags {[bool(self_check(a, b, stats)['flagged']) for a, b in zip(u0, pr)]}, errors {np.round(err, 3).tolist()}")
