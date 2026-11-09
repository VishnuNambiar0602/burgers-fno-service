"""How well does the OOD flag (src/inference.ood_check) track real model failure? Validation data + synthetic families only."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference import ood_check, load_json

stats = load_json("model/ood_stats.json")
val = np.load("data/processed/val.npy", mmap_mode="r")[:, 0, :].astype(np.float64)
fl = [ood_check(u, stats)["is_ood"] for u in val]
print(f"false-alarm rate on all {len(val)} validation ICs: {np.mean(fl):.3%} ({int(np.sum(fl))} flagged)")
errs = json.load(open("reports/validate_m2.json"))["E_ood"]
x = (np.arange(256) + 0.5) / 256
rng = np.random.default_rng(1)
base = val[:50]
fam = {"val_IC x1 (control, solver truth)": base, "amplitude x0.5": 0.5 * base, "amplitude x2": 2 * base, "amplitude x3": 3 * base,
       "mean offset +1.5": base + 1.5, "mean offset +3.0": base + 3.0,
       "gaussian bump": np.exp(-((x - 0.5) / 0.08) ** 2)[None] * np.ones((4, 1)),
       "square wave": np.where((x > 0.25) & (x < 0.75), 1.0, -1.0)[None] * np.ones((4, 1))}
# the noise / sine families reuse the same RNG order as scripts/validate_m2.py
rng.standard_normal((1, 1))  # keep deterministic, exact values are irrelevant for flag rates here
fam["white noise sd 0.3"] = 0.3 * np.random.default_rng(5).standard_normal((8, 256))
fam["sine k=40 (random phase)"] = np.sin(2 * np.pi * 40 * x[None] + np.random.default_rng(6).uniform(0, 6.28, (8, 1)))
fam["sine k=100 (random phase)"] = np.sin(2 * np.pi * 100 * x[None] + np.random.default_rng(7).uniform(0, 6.28, (8, 1)))
fam["constant 0.7"] = 0.7 * np.ones((2, 256))
print(f"{'family':36s} {'flagged':>8s} {'FNO relL2 (validate_m2)':>24s}")
for k, u in fam.items():
    r = np.mean([ood_check(v, stats)["is_ood"] for v in u])
    print(f"{k:36s} {r:8.0%} {errs[k]['fno_s1']:24.4f}")
