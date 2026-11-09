"""Train an FNO or the U-Net baseline on data/processed/train.npy, validating on val.npy.

The TEST set is never loaded by this script. Loss: relative L2 over the whole trajectory (norm over t and x,
per sample, averaged over the batch). Resumable: every epoch writes checkpoints/<name>/last.pt atomically
(model, optimizer, scheduler, epoch, RNG state, best val); rerunning the same command continues from there.
Usage (repo root): python scripts/train.py --model fno --name fno_main --epochs 200 --n-train 4000
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import FNO1d, UNet1d, n_params, rel_l2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["fno", "unet"], required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--n-train", type=int, default=4000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-5)
ap.add_argument("--modes", type=int, default=24)
ap.add_argument("--width", type=int, default=64)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--base", type=int, default=32, help="U-Net base channels")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-minutes", type=float, default=0, help="stop cleanly after this long (smoke tests); 0 = no limit")
ap.add_argument("--label", default="", help="free-text label written into the log header, e.g. SMOKE TEST")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)
np.random.seed(a.seed)
ck_dir = os.path.join("checkpoints", a.name)
os.makedirs(ck_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", f"train_{a.name}.log")
csv_path = os.path.join("logs", f"train_{a.name}.csv")


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


tr = torch.from_numpy(np.load("data/processed/train.npy")[:a.n_train]).to(dev)   # (N, nt, nx) on the GPU
va = torch.from_numpy(np.load("data/processed/val.npy")).to(dev)
nt, nx = tr.shape[1], tr.shape[2]
model = (FNO1d(nt, a.modes, a.width, a.layers) if a.model == "fno" else UNet1d(nt, a.base)).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
start, best = 0, float("inf")
last = os.path.join(ck_dir, "last.pt")
if os.path.exists(last):
    ck = torch.load(last, map_location="cpu", weights_only=False)
    skip = ("epochs", "max_minutes", "label")
    if {k: v for k, v in ck["args"].items() if k not in skip} != {k: v for k, v in vars(a).items() if k not in skip}:
        sys.exit(f"{last} was written with a different config; use a new --name. Not overwriting.")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start, best = ck["epoch"], ck["best"]
    torch.set_rng_state(ck["rng_cpu"])
    if dev == "cuda":
        torch.cuda.set_rng_state(ck["rng_cuda"])
    log(f"RESUMED from epoch {start} (best val so far {best:.5f})")
else:
    log(f"START {a.label} | args={json.dumps(vars(a))}")
    log(f"model={a.model} params={n_params(model):,} device={dev} train={tuple(tr.shape)} val={tuple(va.shape)} "
        f"torch={torch.__version__}")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_relL2", "val_relL2", "lr", "epoch_s", "peak_vram_GB"])


def evaluate(x):
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(x), 100):
            b = x[i:i + 100]
            errs.append(rel_l2(model(b[:, 0]), b, (1, 2)))
    return torch.cat(errs).mean().item()


t_start = time.time()
for ep in range(start, a.epochs):
    t0 = time.time()
    model.train()
    perm = torch.randperm(len(tr), device=dev)
    tot, n = 0.0, 0
    for i in range(0, len(tr), a.batch):
        b = tr[perm[i:i + a.batch]]
        loss = rel_l2(model(b[:, 0]), b, (1, 2)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += loss.item() * len(b); n += len(b)
    sched.step()
    v = evaluate(va)
    dt = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
    log(f"epoch {ep + 1}/{a.epochs} train_relL2={tot / n:.5f} val_relL2={v:.5f} lr={sched.get_last_lr()[0]:.2e} "
        f"epoch_s={dt:.1f} peak_vram_GB={vram:.2f}")
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([ep + 1, tot / n, v, sched.get_last_lr()[0], dt, vram])
    if v < best:
        best = v
        torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep + 1, "val_relL2": v},
                   os.path.join(ck_dir, "best.pt"))
    ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1,
          "best": best, "args": vars(a), "rng_cpu": torch.get_rng_state(),
          "rng_cuda": torch.cuda.get_rng_state() if dev == "cuda" else None}
    torch.save(ck, last + ".tmp")
    os.replace(last + ".tmp", last)                   # atomic: a crash never leaves a half-written last.pt
    if a.max_minutes and (time.time() - t_start) / 60 >= a.max_minutes:
        log(f"STOPPED early after {ep + 1} epochs because --max-minutes={a.max_minutes} was reached")
        break
else:
    log(f"FINISHED {a.epochs} epochs. best val_relL2={best:.5f}")
