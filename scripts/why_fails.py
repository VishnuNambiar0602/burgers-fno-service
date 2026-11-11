"""Why does the FNO fail on 'Two waves' and 'Square', and do free inference-time fixes help? (validation + synthetic inputs only)
A. how unusual are these inputs compared with the 4000 TRAINING initial conditions?
B. where in space-time is the error?
C. free fixes, no retraining: symmetry averaging (reflection), shift averaging, and averaging the 3 trained seeds.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference import NU_EFF  # noqa: E402
from src.models import FNO1d  # noqa: E402
from src.solver import solve  # noqa: E402

t = np.arange(201) * 0.01
x = (np.arange(256) + 0.5) / 256
dx = 1 / 256


def load(seed):
    ck = torch.load(f"checkpoints/fno_s{seed}/best.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    m = FNO1d(201, a["modes"], a["width"], a["layers"])
    m.load_state_dict(ck["model"])
    return m.eval()


M = [load(s) for s in (0, 1, 2)]


def pred(m, u0):
    with torch.no_grad():
        return m(torch.tensor(np.atleast_2d(u0), dtype=torch.float32)).numpy().astype(np.float64)


def rel(a, b):
    return np.sqrt(((a - b) ** 2).sum((-2, -1)) / (b ** 2).sum((-2, -1)))


# ------------------------------------------------------------- A. how unusual are the inputs?
tr = np.load("data/processed/train.npy", mmap_mode="r")[:, 0, :].astype(np.float64)


def feats(u):
    g = np.abs(np.roll(u, -1, -1) - np.roll(u, 1, -1)) / (2 * dx)
    P = np.abs(np.fft.rfft(u, axis=-1)) ** 2
    P[..., 0] = 0
    kk = np.arange(P.shape[-1])
    return {"max |u|": np.abs(u).max(-1), "max |du/dx| (steepness)": g.max(-1), "dominant wavenumber": (P * 1.0).argmax(-1),
            "energy above k=8": P[..., 9:].sum(-1) / P.sum(-1), "std": u.std(-1)}


two = 0.8 * np.sin(4 * np.pi * x + 0.5) + 0.5 * np.sin(10 * np.pi * x + 1.2)
sq = np.where((x > .25) & (x < .75), 1.0, -1.0)
bump = np.exp(-((x - .5) / .08) ** 2)
sine = np.sin(2 * np.pi * x)
ft = feats(tr)
print("A. Where each input sits among the 4000 training initial conditions (percentile = share of training inputs that are SMALLER)")
print(f"{'feature':28s} {'train median':>12s} {'train p99':>10s} | {'sine':>16s} {'two waves':>16s} {'square':>16s}")
for k, v in ft.items():
    row = []
    for u in (sine, two, sq):
        f = feats(u[None])[k][0]
        row.append(f"{f:9.3g} ({100 * np.mean(v < f):3.0f}%)")
    print(f"{k:28s} {np.median(v):12.3g} {np.quantile(v, .99):10.3g} | " + " ".join(f"{r:>16s}" for r in row))

# ------------------------------------------------------------- B. where is the error?
print("\nB. Where the FNO (seed 1) is wrong, for the two hard waves (error vs solver truth)")
for name, u in (("two waves", two), ("square", sq), ("smooth wave (reference)", sine)):
    truth = solve(u[None], NU_EFF, t)[0]
    p = pred(M[1], u)[0]
    per_t = np.sqrt(((p - truth) ** 2).sum(-1) / (truth ** 2).sum(-1))
    g = (np.roll(truth, -1, -1) - np.roll(truth, 1, -1)) / (2 * dx)
    shock = g < -20
    near = shock.copy()
    for s in range(1, 6):
        near |= np.roll(shock, s, -1) | np.roll(shock, -s, -1)
    err2 = (p - truth) ** 2
    share_near = err2[near].sum() / err2.sum()
    rms = lambda a: np.sqrt((a ** 2).mean())
    print(f"  {name:24s} absolute RMS error / RMS of truth at t=0.5: {rms((p - truth)[50]):.3f}/{rms(truth[50]):.3f}  t=1: {rms((p - truth)[100]):.3f}/{rms(truth[100]):.3f}  t=2: {rms((p - truth)[200]):.3f}/{rms(truth[200]):.3f}")
    print(f"  {name:24s} whole {np.sqrt(err2.sum() / (truth ** 2).sum()):.3f} | at t=0.1/0.5/1/2: {per_t[10]:.3f} {per_t[50]:.3f} {per_t[100]:.3f} {per_t[200]:.3f} "
          f"| points near a shock: {near.mean():.0%}, share of squared error there: {share_near:.0%} | max |truth| at t=2: {np.abs(truth[-1]).max():.2f}")

# ------------------------------------------------------------- C. free fixes (no retraining)
def reflect_avg(m, u0):                    # Burgers symmetry: if u(x,t) solves it, so does -u(1-x,t)
    p = pred(m, u0)
    pr = -pred(m, -u0[:, ::-1])[:, :, ::-1]
    return 0.5 * (p + pr)


def shift_avg(m, u0, shifts=(0, 32, 64, 96, 128, 160, 192, 224)):
    out = 0
    for s in shifts:
        out = out + np.roll(pred(m, np.roll(u0, s, -1)), -s, -1)
    return out / len(shifts)


def ens(u0):
    return np.mean([pred(m, u0) for m in M], axis=0)


rng = np.random.default_rng(11)
val = np.load("data/processed/val.npy").astype(np.float64)
drawn = []
for _ in range(60):
    n = rng.integers(6, 14); px = np.sort(rng.uniform(0, 1, n)); py = rng.uniform(-2, 2, n)
    u = np.interp(x, px, py, period=1.0) + 0.05 * rng.standard_normal(256)
    F = np.fft.rfft(u); F[9:] = 0; drawn.append(np.clip(np.fft.irfft(F, n=256), -3.35, 3.35))
mixes = []
for k in range(40):
    r = np.random.default_rng(k + 1); comps = [(0.3 + 0.9 * r.random(), 1 + int(r.integers(0, 8)), 6.28 * r.random()) for _ in range(1 + int(r.integers(0, 3)))]
    s = min(1.0, 2.4 / sum(a for a, _, _ in comps)); mixes.append(sum(a * s * np.sin(2 * np.pi * kk * x + p) for a, kk, p in comps) + (r.random() - 0.5) * 0.8)
sets = {"validation (n=300)": (val[:300, 0], val[:300]), "two waves": (two[None], None), "square": (sq[None], None), "bump": (bump[None], None),
        "hand-drawn-like (n=60)": (np.array(drawn), None), "random sine mixes (n=40)": (np.array(mixes), None)}
methods = {"seed 1 alone (deployed)": lambda u: pred(M[1], u), "reflection average": lambda u: reflect_avg(M[1], u),
           "shift average (8 shifts)": lambda u: shift_avg(M[1], u), "3-seed ensemble": ens,
           "3 seeds + reflection": lambda u: 0.5 * (ens(u) + (-np.mean([pred(m, -u[:, ::-1]) for m in M], axis=0)[:, :, ::-1]))}
print("\nC. Free fixes at inference time (mean relative L2 vs truth; lower is better)")
print(f"{'':26s}" + "".join(f"{k:>16s}" for k in methods))
spread = {}
for name, (u0, truth) in sets.items():
    truth = truth if truth is not None else solve(u0, NU_EFF, t)
    print(f"{name:26s}" + "".join(f"{rel(f(u0), truth).mean():16.4f}" for f in methods.values()))
    spread[name] = (u0, truth)

# seed disagreement as a trust signal
def spread_score(u0):
    P = np.stack([pred(m, u0) for m in M]); mu = P.mean(0)
    return np.sqrt((((P - mu) ** 2).sum((-2, -1))).mean(0) / (mu ** 2).sum((-2, -1)))


sc, er = [], []
for name, (u0, truth) in spread.items():
    sc.append(spread_score(u0)); er.append(rel(pred(M[1], u0), truth))
sc, er = np.concatenate(sc), np.concatenate(er); bad = er > 0.06
r = sc.argsort().argsort() + 1.0
auc = (r[bad].sum() - bad.sum() * (bad.sum() + 1) / 2) / (bad.sum() * (~bad).sum())
vs = spread_score(val[:300, 0]); thr = np.quantile(vs, 0.99)
print(f"\nSeed disagreement as a free warning: AUC {auc:.2f}; threshold at validation p99 = {thr:.3f} flags {np.mean(sc[bad] > thr):.0%} of bad cases, {np.mean(sc[~bad] > thr):.1%} of good ones")
