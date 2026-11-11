# M7 plan (deferred): a local-rule time-stepper that generalises to unseen waves

Status: **PLANNED, NOT STARTED.** Agreed with the user on 2026-11-11 ~10:25 IST; the user will schedule it later.
Nothing in the deployed system changes until this experiment earns its place. Gates still to pass when it starts:
(b) before any training run, (d) architecture change. No tests are run unless the user says so.

## Problem it addresses
The deployed FNO maps a whole starting wave to the whole 2-second solution in one shot, trained on 4,000 PDEBench waves.
Unseen waves break it (measured: hand-drawn-like curves 75% of cases more than 6% off; random sine mixes 52%; Square 0.110; Two waves 0.111).
The user needs something viable for waves it has never seen, not a patch per wave type.

## Why not reinforcement learning
The target is known exactly (our solver) and the error is differentiable, so gradient descent on the error is the same objective with less noise.
RL fits non-differentiable decisions (for example "when should the app call the slow solver"), which is a possible side feature, not the fix.
What survives from the idea (break the problem into small pieces, score each piece, optimise the total) is *rollout training* of a *local* model.

## Design
1. **Local flux model.** A small periodic 1D CNN (about 30k parameters, receptive field about 9 grid points) predicts the flux between cells from the local values.
   The update is `u_new = u - dt/dx * (F[i+1/2] - F[i-1/2]) + diffusion`, so total mass is conserved by construction. Translation- and resolution-independent.
2. **Time stepping.** One step of 0.01 (200 steps for t in [0, 2]); each output is fed back in.
3. **Rollout (pushforward) training.** Loss over 4 to 8 consecutive steps so errors do not compound. (PDEBench's own baselines use the pushforward trick: seen in the web search of 2026-11-09.)
4. **Broad data.** Existing 4,000 PDEBench trajectories (every frame is a training pair) plus about 10,000 solver-generated waves of varied style: random band-limited Fourier series,
   piecewise-linear and hand-drawn-like curves, steps and squares, bumps, varied mean and amplitude up to the training limit and a little beyond.
   Our solver matched the stored PDEBench data to 0.02-0.05% relative L2 at 1024 cells and 0.2-0.3% at 256 cells, so it is a valid label source (to be disclosed: part of the data is self-generated).
5. **Runtime verification.** The PDE residual of any forecast can be checked without labels; the app keeps the self-check and falls back to the solver when it fails.

## Evaluation (sets and criteria fixed BEFORE running)
- Compare against the deployed FNO (seed 1) on: the PDEBench validation set; Two waves; Square; hand-drawn-like curves; random sine mixes; a NEW held-out set of unfamiliar waves (generated with different seeds and styles than the training mix); mass drift.
- Same metric as before: relative L2 over (t, x), mean over samples, with 3 seeds if time allows.
- **Success:** lower mean error than the FNO on the unfamiliar-wave set AND within about 1.5x of the FNO on the PDEBench validation set. If not met, M7 is reported as a negative result.
- The PDEBench test set is touched only for a final single evaluation, and its second use (for a new model version) is disclosed.

## Estimates (unmeasured)
| step | estimate |
|---|---|
| implement model, rollout training, evaluation scripts | about 45 min |
| generate about 10,000 solver waves (batched numpy; 8 waves took 0.5 s earlier) | about 15-30 min |
| train, one seed, about 8,000 steps | about 20-30 min |
| evaluate and write up | about 20 min |
| inference speed (200 sequential steps on CPU) | about 60 ms (a guess); the solver takes 0.4-0.8 s, the FNO 2 ms |

## Known risks
No learned model can guarantee correctness on arbitrary inputs; errors can accumulate over 200 steps; shocks are hard for a small network and may smear or oscillate;
accuracy on familiar waves may be worse than the FNO's 2.7%; inference is slower than the FNO.

## Options evaluated earlier and kept on the shelf (not part of M7)
- 3-seed ensemble + mirror-image averaging: 11-23% lower error on most families (validation 0.0269 to 0.0224), no help for Two waves; seed spread is also a warning signal (AUC 0.99). Needs a new test evaluation to be reported. `reports/why_fails.txt`.
- Loss that weights every time step equally (targets Two waves' constant absolute error floor); more Fourier modes (24 to 48) for sharp fronts. Both need a training run.
- Broader-family retraining of the FNO itself (about 30 min data + 45 min per run, estimated).
