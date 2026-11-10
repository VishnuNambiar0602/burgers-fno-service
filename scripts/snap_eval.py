"""Choose the drawing 'snap' cutoff from evidence: synthetic hand-drawn-like curves (random control points + jitter),
low-passed at several maximum wavenumbers, FNO vs solver error + OOD-flag rate. Synthetic inputs only; no test data."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference import load_model, predict, ood_check, load_json, NU_EFF
from src.solver import solve

model, _ = load_model("model/fno_s1.pt"); stats = load_json("model/ood_stats.json")
x = (np.arange(256) + 0.5) / 256; t = np.arange(201) * 0.01
rng = np.random.default_rng(123)
curves = []
for _ in range(60):
    n = rng.integers(6, 14); px = np.sort(rng.uniform(0, 1, n)); py = rng.uniform(-2, 2, n)
    u = np.interp(x, px, py, period=1.0) + 0.05 * rng.standard_normal(256)   # piecewise-linear hand-drawn look + pen jitter
    curves.append(u)
curves = np.array(curves)

def lowpass(u, k):
    F = np.fft.rfft(u); F[k + 1:] = 0; return np.fft.irfft(F, n=len(u))

print(f"{'kmax':>5s} {'change vs drawing':>18s} {'FNO-vs-solver median':>21s} {'p90':>7s} {'max':>7s} {'OOD flagged':>12s}")
for k in (3, 4, 6, 8, 12, 16, 128):
    snapped = np.array([np.clip(lowpass(u, k), -3.35, 3.35) for u in curves])
    truth = solve(snapped, NU_EFF, t)                       # (60, 201, 256)
    pred = predict(model, snapped).astype(float)
    e = np.sqrt(((pred - truth) ** 2).sum((1, 2)) / (truth ** 2).sum((1, 2)))
    ch = np.sqrt(((snapped - curves) ** 2).sum(1) / (curves ** 2).sum(1))
    fl = np.mean([ood_check(u, stats)["is_ood"] for u in snapped])
    print(f"{k:5d} {np.median(ch):18.3f} {np.median(e):21.4f} {np.quantile(e, .9):7.4f} {e.max():7.4f} {fl:12.0%}")
