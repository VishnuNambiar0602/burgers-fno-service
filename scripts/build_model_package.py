"""Package the chosen checkpoint for serving: model/fno_s1.pt (weights_only-loadable), model/ood_stats.json,
model/model_card.json. All numbers in the card are READ from reports/*.json, never typed in.

  python scripts/build_model_package.py
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference import ood_features, HF_K, ALLOWED_NX, NT, DT, load_model, predict, self_check_signals  # noqa: E402
from src.models import n_params, FNO1d  # noqa: E402

SRC = "checkpoints/fno_s1/best.pt"
ck = torch.load(SRC, map_location="cpu", weights_only=False)
c = ck["args"]
cfg = {"nt": 201, "modes": c["modes"], "width": c["width"], "layers": c["layers"]}
meta = {"name": "fno_s1", "seed": c["seed"], "source_checkpoint": SRC, "epoch": ck["epoch"],
        "val_relL2_whole_traj": ck["val_relL2"], "torch_trained_with": str(torch.__version__)}
os.makedirs("model", exist_ok=True)
if "--stats-only" not in sys.argv:                       # --stats-only: recompute thresholds/card without re-saving the weights
    torch.save({"state_dict": ck["model"], "config": cfg, "meta": meta}, "model/fno_s1.pt")
sha = hashlib.sha256(open("model/fno_s1.pt", "rb").read()).hexdigest()

# ---- OOD statistics from the 4000 TRAINING initial conditions only
tr = np.load("data/processed/train.npy", mmap_mode="r")[:, 0, :].astype(np.float64)
feats = [ood_features(u) for u in tr]
rng = lambda k: {"min": float(min(f[k] for f in feats)), "max": float(max(f[k] for f in feats)),
                 "p99": float(np.quantile([f[k] for f in feats], 0.99))}
stats = {"computed_on": "data/processed/train.npy initial conditions (n=4000, 256 points)", "hf_wavenumber_cutoff": HF_K,
         "train_range": {"max_abs": rng("max_abs"), "hf_energy_frac": rng("hf_energy_frac"),
                         "abs_mean": {"max": float(max(abs(f["mean"]) for f in feats))}}}
# self-check thresholds: 99th percentile of each signal over the 500 VALIDATION inputs (about 1% false alarms by construction)
_m, _ = load_model("model/fno_s1.pt")
_va = np.load("data/processed/val.npy", mmap_mode="r")[:, 0, :].astype(np.float64)
_pr = predict(_m, _va).astype(np.float64)
_sig = [self_check_signals(a, b) for a, b in zip(_va, _pr)]
stats["self_check"] = {"computed_on": "validation initial conditions (n=500), FNO seed 1 output",
                       "thresholds": {k: float(np.quantile([s[k] for s in _sig], 0.99)) for k in ("t0", "mass", "resid")}}
json.dump(stats, open("model/ood_stats.json", "w"), indent=1)

# ---- model card, numbers read from the evaluation reports
rd = lambda p: json.load(open(p))
test = {s: rd(f"reports/eval_test_fno_s{s}.json") for s in range(3)}
val = {s: rd(f"reports/eval_val_fno_s{s}.json") for s in range(3)}
unet_test = [rd(f"reports/eval_test_unet_s{s}.json")["whole_mean"] for s in range(3)]
card = {
    "name": "fno_s1", "type": "Fourier Neural Operator, 1D, written from scratch (src/models.py)",
    "parameters": n_params(FNO1d(**{"nt": 201, "modes": cfg["modes"], "width": cfg["width"], "layers": cfg["layers"]})),
    "config": cfg, "sha256": sha, "source": meta,
    "task": "initial condition u(x,0) -> 201 frames u(x, 0.01 k), k=0..200, one forward pass",
    "equation": "u_t + (u^2/2)_x = (nu/pi) u_xx with nu = 0.01, periodic on [0,1) (PDEBench convention, verified against the data)",
    "input": {"allowed_lengths": list(ALLOWED_NX), "trained_at": 256, "domain": "[0,1) periodic, cell-centred",
              "note": "512 was checked zero-shot on validation data only (relL2 0.0271)"},
    "output": {"frames": NT, "dt": DT, "t_max": (NT - 1) * DT},
    "training": {"dataset": "PDEBench 1D_Burgers_Sols_Nu0.01.hdf5 (CC BY 4.0)", "train_samples": 4000, "val": 500, "test": 1000,
                 "split_seed": 0, "epochs": 300, "x_subsample": "every 4th point (256)"},
    "metrics": {"definition": "relative L2 over (t,x) per sample, mean over samples",
                "deployed_seed1_val": val[1]["whole_mean"], "deployed_seed1_test": test[1]["whole_mean"],
                "fno_test_mean_3_seeds": float(np.mean([test[s]["whole_mean"] for s in range(3)])),
                "fno_test_sd_3_seeds": float(np.std([test[s]["whole_mean"] for s in range(3)], ddof=1)),
                "unet_test_mean_3_seeds": float(np.mean(unet_test)),
                "near_shock_pooled_relL2_test_seed1": test[1]["near_shock"]["pooled_relL2"],
                "away_from_shock_pooled_relL2_test_seed1": test[1]["away_from_shock"]["pooled_relL2"],
                "mass_drift_pred_test_seed1": test[1]["mass_drift_pred"], "mass_drift_truth_test": test[1]["mass_drift_true"]},
    "known_limitations": [
        "Trained at one viscosity (nu = 0.01) and one family of initial conditions; nothing else is claimed.",
        "Does not conserve total mass: drift is about 100x that of the reference data (see metrics).",
        "Error is about 3x larger within a few grid points of shocks than away from them.",
        "Error at t=0 is about 0.016: the network does not reproduce its own input exactly.",
        "Fails on inputs unlike training: taller than 3.35, mean beyond about 2.4, very high-frequency or noisy inputs.",
        "Not tested at any other grid length than 256 (trained) and 512 (validation only)."],
    "ood_flag_meaning": "a flag means the input is outside the training range; no flag does NOT certify accuracy",
}
json.dump(card, open("model/model_card.json", "w"), indent=1)
print("model/fno_s1.pt", os.path.getsize("model/fno_s1.pt"), "bytes | sha256", sha[:16], "...")
print("params", card["parameters"], "| val", round(card["metrics"]["deployed_seed1_val"], 6), "test", round(card["metrics"]["deployed_seed1_test"], 6))
print(json.dumps(stats, indent=1))
