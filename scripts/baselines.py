"""Trivial baselines on the VALIDATION set only (data/processed/val.npy). The test set is never opened here.

Metric ("relative L2"): for each sample s and time index k,
    rel[s, k] = ||pred[s, k, :] - true[s, k, :]||_2 / ||true[s, k, :]||_2        (norm over x)
Reported: mean over samples of rel[s, k] for chosen k, and the mean over samples and over k = 1..200
(k=0 is excluded from the trajectory mean because every baseline that copies the IC is trivially exact there).
Also reported: the whole-trajectory version  ||pred - true||_F / ||true||_F per sample, averaged over samples.

Baselines
  copy_ic        : pred[s, k] = u[s, 0]   for all k
  const_trainmean: pred[s, k] = scalar mean of all training values (computed on train.npy)
  const_icmean   : pred[s, k] = mean_x u[s, 0]   (per-sample constant; mass is conserved, so this is the
                   long-time limit if viscosity flattened everything)
"""
import json
import numpy as np

val = np.load("data/processed/val.npy").astype(np.float64)       # (500, 201, 256)
train_mean = float(np.load("data/processed/train.npy", mmap_mode="r")[:, :, :].mean())
print("val", val.shape, "| train scalar mean", train_mean)

preds = {
    "copy_ic": np.repeat(val[:, :1], val.shape[1], axis=1),
    "const_trainmean": np.full_like(val, train_mean),
    "const_icmean": np.repeat(val[:, :1].mean(axis=2, keepdims=True), val.shape[1], axis=1)
                    * np.ones_like(val),
}
ks = [0, 1, 5, 10, 25, 50, 100, 150, 200]
out = {"method": "per-sample relative L2 over x at each time index, averaged over validation samples",
       "n_val": int(val.shape[0]), "t_index_to_time": "t = 0.01 * k"}
print(f"{'baseline':16s} " + " ".join(f"k={k:<4d}" for k in ks) + " | mean k=1..200 | whole-traj")
for name, p in preds.items():
    rel = np.sqrt(((p - val) ** 2).sum(axis=2) / (val ** 2).sum(axis=2))        # (500, 201)
    whole = np.sqrt(((p - val) ** 2).sum(axis=(1, 2)) / (val ** 2).sum(axis=(1, 2)))
    row = {"per_k": {str(k): float(rel[:, k].mean()) for k in ks},
           "mean_k1_200": float(rel[:, 1:].mean()), "whole_traj": float(whole.mean())}
    out[name] = row
    print(f"{name:16s} " + " ".join(f"{rel[:, k].mean():.3f} " for k in ks) +
          f"| {row['mean_k1_200']:.4f}       | {row['whole_traj']:.4f}")
json.dump(out, open("reports/baselines_val.json", "w"), indent=1)
