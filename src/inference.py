"""Inference helpers shared by the web app and the offline checks, so both run exactly the same code.

Conventions (found and verified in M1/M2): domain [0,1) periodic, cell-centred grid, u_t + (u^2/2)_x = (0.01/pi) u_xx,
output frames t_k = 0.01 k for k = 0..200. The model was trained at 256 points; 512 points was checked zero-shot on
validation data (relL2 0.0271). Any other length is rejected, not silently resampled.
"""
import json

import numpy as np
import torch

from .models import FNO1d

NT = 201
DT = 0.01
NU_EFF = 0.01 / np.pi
ALLOWED_NX = (256, 512)
HF_K = 32  # wavenumber above which energy counts as "high frequency" in the OOD check


def load_model(path):
    """Load the packaged FNO (weights_only: no pickled code is executed). Returns (model, meta)."""
    pkg = torch.load(path, map_location="cpu", weights_only=True)
    c = pkg["config"]
    model = FNO1d(c["nt"], c["modes"], c["width"], c["layers"])
    model.load_state_dict(pkg["state_dict"])
    return model.eval(), pkg["meta"]


def predict(model, u0):
    """u0: array (nx,) or (B, nx) -> float32 array (nt, nx) or (B, nt, nx)."""
    x = torch.as_tensor(np.asarray(u0, dtype=np.float32))
    single = x.ndim == 1
    with torch.inference_mode():
        y = model(x[None] if single else x)
    y = y.numpy()
    return y[0] if single else y


def ood_features(u0):
    """Three scalar features of an initial condition (computed on the values as given)."""
    u0 = np.asarray(u0, dtype=np.float64)
    power = np.abs(np.fft.rfft(u0)) ** 2
    power[0] = 0.0
    total = power.sum()
    hf = float(power[HF_K + 1:].sum() / total) if total > 0 else 0.0
    return {"max_abs": float(np.abs(u0).max()), "mean": float(u0.mean()), "hf_energy_frac": hf}


def ood_check(u0, stats):
    """Flag inputs outside the range seen in training. `stats` is model/ood_stats.json.

    A flag means "the model has not seen inputs like this; its answer may be wrong without looking wrong".
    The absence of a flag does NOT certify accuracy.
    """
    f = ood_features(u0)
    tr = stats["train_range"]
    reasons = []
    if f["max_abs"] > tr["max_abs"]["max"]:
        reasons.append(f"amplitude max|u0| = {f['max_abs']:.3g} exceeds the largest seen in training ({tr['max_abs']['max']:.3g})")
    if abs(f["mean"]) > tr["abs_mean"]["max"]:
        reasons.append(f"mean(u0) = {f['mean']:.3g} is outside the training range (|mean| <= {tr['abs_mean']['max']:.3g})")
    if f["hf_energy_frac"] > tr["hf_energy_frac"]["max"]:
        reasons.append(f"{100 * f['hf_energy_frac']:.1f}% of the energy is at wavenumbers above {HF_K}; "
                       f"the most seen in training was {100 * tr['hf_energy_frac']['max']:.1f}%")
    return {"is_ood": bool(reasons), "reasons": reasons, "features": f}


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def self_check_signals(u0, u):
    """Three cheap sanity signals of ONE prediction, computed from the network's output alone (no solver):
      t0    relative error between frame 0 of the output and the input wave (it should reproduce its input)
      mass  max_t |M(t) - M(0)| / max(mean|u0|, 0.05)   (the equation conserves M = sum(u) dx exactly)
      resid relative PDE residual of the output (finite differences), size of the residual vs size of the terms
    """
    u0 = np.asarray(u0, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    nx = u.shape[-1]
    dx = 1.0 / nx
    mass = np.abs(u.sum(-1) * dx - u0.sum() * dx).max() / max(np.abs(u0).mean(), 0.05)
    t0 = np.sqrt(((u[0] - u0) ** 2).sum() / (u0 ** 2).sum())
    ut = (u[2:] - u[:-2]) / (2 * DT)
    um = u[1:-1]
    fx = (np.roll(um ** 2 / 2, -1, -1) - np.roll(um ** 2 / 2, 1, -1)) / (2 * dx)
    uxx = NU_EFF * (np.roll(um, -1, -1) - 2 * um + np.roll(um, 1, -1)) / dx ** 2
    R = ut + fx - uxx
    resid = np.sqrt((R ** 2).sum()) / (np.sqrt((ut ** 2).sum()) + np.sqrt((fx ** 2).sum()) + np.sqrt((uxx ** 2).sum()) + 1e-9)
    return {"t0": float(t0), "mass": float(mass), "resid": float(resid)}


def self_check(u0, u, stats):
    """Flag a prediction that fails basic sanity checks. Thresholds are validation-set 99th percentiles (stats['self_check']).
    A flag means the forecast is probably wrong; no flag does NOT certify it (about 30% of bad cases pass in our tests)."""
    sig = self_check_signals(u0, u)
    th = stats["self_check"]["thresholds"]
    why = {"t0": "the forecast does not even reproduce the starting wave you gave it (frame 0 is off by {v:.1%}; typical {t:.1%} or less)",
           "mass": "the forecast does not conserve the total amount of 'stuff' as the equation requires (drift score {v:.3g}; typical {t:.3g} or less)",
           "resid": "the forecast does not satisfy the equation itself (residual score {v:.3g}; typical {t:.3g} or less)"}
    reasons = [why[k].format(v=sig[k], t=th[k]) for k in ("t0", "mass", "resid") if sig[k] > th[k]]
    return {"flagged": bool(reasons), "reasons": reasons, "signals": sig}
