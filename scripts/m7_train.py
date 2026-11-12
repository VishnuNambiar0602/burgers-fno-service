"""Train the M7 FluxStepper with rollout (unrolled) training on PDEBench train + solver-generated waves. Resumable.

  python scripts/m7_train.py --name m7_s0 --stages 1:3000,4:6000,8:10000

--stages "K:end,..." = unroll K steps until iteration `end` (a curriculum: 1 step, then 4, then 8). Checkpoints every 500 iterations
(atomic); rerun the same command to resume. Selection uses PDEBench validation + held-out waves of the TRAIN styles only; the UNSEEN styles
and the PDEBench test set are never loaded here.
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
from src.m7_model import FluxStepper, n_params  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--stages", default="1:3000,4:6000,8:10000")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--ch", type=int, default=32)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-pdebench", type=int, default=4000)
ap.add_argument("--n-synth", type=int, default=10000)
ap.add_argument("--no-synth", action="store_true", help="PDEBench data only (ablation)")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)
np.random.seed(a.seed)
stages = [(int(s.split(":")[0]), int(s.split(":")[1])) for s in a.stages.split(",")]
total = stages[-1][1]
ck_dir = os.path.join("checkpoints", a.name)
os.makedirs(ck_dir, exist_ok=True)
log_path = os.path.join("logs", f"train_{a.name}.log")


def log(m):
    line = f"{time.strftime('%F %T')} {m}"
    print(line, flush=True)
    open(log_path, "a").write(line + "\n")


# ---- data on the GPU (loaded chunk by chunk so CPU RAM stays small)
pde = np.load("data/processed/train.npy", mmap_mode="r")[:a.n_pdebench]
files = [] if a.no_synth else sorted(glob.glob("data/m7/train/chunk_*.npy"))
n_synth = 0
if files:
    n_synth = sum(np.load(f, mmap_mode="r").shape[0] for f in files)
    n_synth = min(n_synth, a.n_synth)
N = len(pde) + n_synth
data = torch.empty((N, 201, 256), device=dev, dtype=torch.float32)
data[:len(pde)] = torch.from_numpy(np.ascontiguousarray(pde)).to(dev)
pos = len(pde)
for f in files:
    if pos >= N:
        break
    c = np.load(f)[: N - pos]
    data[pos:pos + len(c)] = torch.from_numpy(c).to(dev)
    pos += len(c)
assert pos == N and torch.isfinite(data).all()
val_pde = torch.from_numpy(np.load("data/processed/val.npy")[:200]).to(dev)
hs = "data/m7/heldout_style.npy"
val_style = torch.from_numpy(np.load(hs)[:200]).to(dev) if os.path.exists(hs) else None

model = FluxStepper(a.ch, a.layers).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
it = 0
best = float("inf")
last = os.path.join(ck_dir, "last.pt")
if os.path.exists(last):
    ck = torch.load(last, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); it, best = ck["iter"], ck["best"]
    log(f"RESUMED at iteration {it} (best selection error so far {best:.5f})")
else:
    log(f"START args={json.dumps(vars(a))} params={n_params(model):,} receptive_field={model.receptive} device={dev} "
        f"data: pdebench {len(pde)} + synthetic {n_synth} = {N} trajectories ({data.numel() * 4 / 1e9:.2f} GB on {dev})")


def rollout_err(val):
    model.eval()
    with torch.no_grad():
        errs = []
        for i in range(0, len(val), 100):
            v = val[i:i + 100]
            p = model(v[:, 0], 200, clamp=True)
            errs.append(torch.sqrt(((p - v) ** 2).sum((1, 2)) / (v ** 2).sum((1, 2))))
    model.train()
    return torch.cat(errs).mean().item()


def unroll_for(i):
    for k, end in stages:
        if i < end:
            return k
    return stages[-1][0]


t0 = time.time()
model.train()
while it < total:
    K = unroll_for(it)
    lr = 1e-5 + 0.5 * (a.lr - 1e-5) * (1 + np.cos(np.pi * it / total))
    for g in opt.param_groups:
        g["lr"] = lr
    idx = torch.randint(0, N, (a.batch,), device=dev)
    k0 = torch.randint(0, 201 - K, (a.batch,), device=dev)
    ar = torch.arange(K + 1, device=dev)
    seq = data[idx[:, None], k0[:, None] + ar[None]]                    # (B, K+1, 256)
    flip = torch.rand(a.batch, device=dev) < 0.5                         # Burgers symmetry: u(x,t) -> -u(1-x,t)
    seq = torch.where(flip[:, None, None], -seq.flip(-1), seq)
    pred = model(seq[:, 0], K)
    tgt = seq[:, 1:]
    e = ((pred[:, 1:] - tgt) ** 2).sum(-1) / ((tgt ** 2).sum(-1) + 0.05 * 256)     # relative, with a floor for tiny frames
    loss = e.mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    it += 1
    if it % 250 == 0:
        log(f"it {it}/{total} unroll={K} lr={lr:.2e} loss={loss.item():.5f} ({(time.time() - t0) / 60:.1f} min)")
    if it % 500 == 0 or it == total:
        ep, es = rollout_err(val_pde), (rollout_err(val_style) if val_style is not None else float("nan"))
        sel = ep if val_style is None else 0.5 * (ep + es)
        log(f"VAL it {it}: full 200-step rollout relL2  PDEBench-val {ep:.4f} | held-out train-style {es:.4f} | selection {sel:.4f}")
        if np.isfinite(sel) and sel < best:
            best = sel
            torch.save({"model": model.state_dict(), "args": vars(a), "iter": it, "val_pde": ep, "val_style": es}, os.path.join(ck_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it, "best": best, "args": vars(a)}, last + ".tmp")
        os.replace(last + ".tmp", last)
log(f"FINISHED {total} iterations in {(time.time() - t0) / 60:.1f} min; best selection error {best:.5f}")
