"""CONTROL for M7: the same one-shot FNO as the deployed model, retrained on the SAME combined data M7 uses (PDEBench train + solver-generated waves).
If it generalises to the UNSEEN styles as well as M7, the gain comes from the broader data, not from M7's architecture.

  python scripts/m7_fno_aug.py --name m7_fno_aug --epochs 30      (resumable: rerun the same command)
Same loss/optimiser/settings as scripts/train.py (relative L2 over the whole trajectory, Adam 1e-3 cosine, batch 32). Selection on
PDEBench validation + held-out waves of the TRAIN styles only; UNSEEN styles and the PDEBench test set are never loaded.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import FNO1d, n_params, rel_l2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--modes", type=int, default=24)
ap.add_argument("--width", type=int, default=64)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)
ck_dir = os.path.join("checkpoints", a.name)
os.makedirs(ck_dir, exist_ok=True)
log_path = os.path.join("logs", f"train_{a.name}.log")


def log(m):
    line = f"{time.strftime('%F %T')} {m}"
    print(line, flush=True)
    open(log_path, "a").write(line + "\n")


pde = np.load("data/processed/train.npy", mmap_mode="r")
files = sorted(glob.glob("data/m7/train/chunk_*.npy"))
n_syn = sum(np.load(f, mmap_mode="r").shape[0] for f in files)
N = len(pde) + n_syn
data = torch.empty((N, 201, 256), dtype=torch.float32, device=dev)
data[:len(pde)] = torch.from_numpy(np.ascontiguousarray(pde)).to(dev)
pos = len(pde)
for f in files:
    c = np.load(f)
    data[pos:pos + len(c)] = torch.from_numpy(c).to(dev)
    pos += len(c)
val_pde = torch.from_numpy(np.load("data/processed/val.npy")).to(dev)
val_style = torch.from_numpy(np.load("data/m7/heldout_style.npy")).to(dev)

model = FNO1d(201, a.modes, a.width, a.layers).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
ep0, best = 0, float("inf")
last = os.path.join(ck_dir, "last.pt")
if os.path.exists(last):
    ck = torch.load(last, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); ep0, best = ck["epoch"], ck["best"]
    log(f"RESUMED at epoch {ep0}")
else:
    log(f"START args={json.dumps(vars(a))} params={n_params(model):,} device={dev} data={N} trajectories (pdebench {len(pde)} + synthetic {n_syn})")


def evaluate(v):
    model.eval()
    with torch.no_grad():
        e = torch.cat([rel_l2(model(v[i:i + 100, 0]), v[i:i + 100], (1, 2)) for i in range(0, len(v), 100)]).mean().item()
    model.train()
    return e


for ep in range(ep0, a.epochs):
    t0 = time.time()
    perm = torch.randperm(N, device=dev)
    tot = 0.0
    for i in range(0, N, a.batch):
        b = data[perm[i:i + a.batch]]
        loss = rel_l2(model(b[:, 0]), b, (1, 2)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += loss.item() * len(b)
    sched.step()
    ep_, es_ = evaluate(val_pde), evaluate(val_style)
    sel = 0.5 * (ep_ + es_)
    log(f"epoch {ep + 1}/{a.epochs} train {tot / N:.4f} | PDEBench-val {ep_:.4f} | held-out train-style {es_:.4f} | {time.time() - t0:.0f}s")
    if sel < best:
        best = sel
        torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep + 1}, os.path.join(ck_dir, "best.pt"))
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1, "best": best, "args": vars(a)}, last + ".tmp")
    os.replace(last + ".tmp", last)
log(f"FINISHED {a.epochs} epochs; best selection error {best:.4f}")
