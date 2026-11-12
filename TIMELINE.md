# TIMELINE

**T0 = 2026-11-09 15:03:49 +05:30 (IST)**, from `date` run in this session (UTC 09:33:49).
**Final commit due: T0+72h = 2026-11-12 15:03 IST.**

Absolute clock times are provisional: they are pure elapsed-time conversions and do not yet account for the user's availability blocks (to be supplied).

| Milestone | T+ deadline | Absolute deadline (IST) | Effective h (budget) | Actual completion | Status |
|---|---|---|---|---|---|
| M0 Discovery, environment, plan | T+6 | 2026-11-09 21:03 | ~1.5 | 2026-11-09 18:52 (T+3h48m) | DONE |
| M1 Data inspected, split, validated | T+14 | 2026-11-10 05:03 | ~3 | 2026-11-09 20:07 (T+5h03m) | DONE |
| M2 Baselines, FNO trained, evaluated, physics checks | T+30 | 2026-11-10 21:03 | ~8 | 2026-11-09 21:13 (T+6h09m) | DONE (kept models; extra validation 23:03) |
| M3 FastAPI backend working locally | T+38 | 2026-11-11 05:03 | ~3 | 2026-11-09 23:09 (T+8h05m) | DONE |
| M4 Skeleton deploy, live URL with real inference | T+44 | 2026-11-11 11:03 | ~2 | 2026-11-12 10:31 (T+67h27m, 23h28m late) | DONE: https://burgers-fno-service.onrender.com, live-vs-offline PASS (7.94e-07) |
| M5 Front end polish, solver comparison, OOD warnings | T+54 | 2026-11-11 21:03 | ~4 | 2026-11-10 09:03 (T+18h) | DONE locally (not deployed) |
| M6 Final verification, README, cleanup, FINAL COMMIT | T+66 | 2026-11-12 09:03 | ~3 | 2026-11-12 08:19 (T+65h15m) | DONE locally: 62/62 tests, live-vs-offline PASS (local, 7.32e-07), README final, pushed. Live public URL not met (M4). |
| Buffer (no new features) | T+72 | 2026-11-12 15:03 | - | | - |
| M7 (plan, deferred by user) Local-rule time-stepper for unseen waves | - | - | ~2.5 (est.) | | EXPLORATORY RESULT: mixed, pre-registered criterion not met; not deployed; see README 11b and docs/M7_plan.md |

Cut list, in order (propose, user approves): (1) freehand drawing; (2) multiple viscosities; (3) UI polish; (4) extra architectures; (5) reduced training data / fewer epochs, documented honestly.
Never cut: baseline, held-out evaluation, physics checks, live inference check, honest README.

## Log
- 2026-11-09 15:03:49 IST: T0 recorded. Hardware/environment detected. No code written, nothing downloaded.
- 2026-11-09 15:15 IST: Python 3.13.3 venv created at .venv (git-ignored, no packages installed). .gitignore added.
- 2026-11-09 15:20 IST: Found that the pre-existing git repo was rooted at D:\Projects (parent of all the user's projects), not Pre-doc. My branch rename and remote add had briefly touched it; both were reverted (parent is back on `master`, no remote). Created a separate repo in Pre-doc (branch `main`, origin set, NOT pushed). Parent repo's index/HEAD tree is already damaged ("bad tree object HEAD"); left untouched. User stepping away ~30 min; push awaits go-ahead.
- 2026-11-09 15:28 IST: Install (torch 2.14.1+cu126 chosen by pip resolver) and 2.6 GB/8.2 GB downloads running in background (logs/install.log, logs/download.log). Range request tested on 1000 bytes: HTTP 206, bytes identical to on-disk bytes. Origin removed at user request (URL had come from the user first message); user adds the real remote at gate (c). scripts/chain.sh runs GPU check, commit (no push), size+md5 verify, inspect_h5 unattended. No training or scope change until M1 proposal is approved.
- 2026-11-09 18:40 IST: User returned. All background jobs had died at 15:53 (no log lines after that; Windows boot time shows 18:35, so the laptop slept/restarted - exact cause not confirmed). Resumed: torch wheel curl resumes from 910 MB, dataset download resumes from 1.478 GB after the wheel completes, chain restarted. User says laptop stays on for the next 10-12 h.
- 2026-11-09 18:52 IST: Install finished: torch 2.14.1+cu126, numpy 2.5.4, h5py 3.16.0, scipy 1.18.1, matplotlib 3.11.2, fastapi 0.162.2, uvicorn 0.54.0, pydantic 2.14.3, pytest 9.2.0, httpx 0.28.1 (44-line lockfile). GPU check passed (RTX 4050, torch reports 6.44 GB VRAM, not the 4 GB that Windows WMI reported). pytest: 5 passed. Lockfile torch line fixed (pip wrote a local file path). Dataset at ~24%, single downloader.
- 2026-11-09 19:38 IST: Dataset downloaded (6 resumes), size 8232968312 and md5 e6d9a4f6...2770 verified (chain + download script). HDF5: tensor (10000,201,1024) float32, Nu=0.01 attr, x cell-centred dx=1/1024 on [0,1), t-coordinate has 202 entries (0..2.01) vs 201 frames (frame k = 0.01*k confirmed by solver).
- 2026-11-09 19:49 IST: Split + processed arrays (train 4000 / val 500 / test 1000, x stride 4 -> 256 pts, 201 frames), seed 0. 6 tests pass. Baselines (val only) and IC leakage check run. Solver check: data follows u_t+(u^2/2)_x=(nu/pi)u_xx; 1024-cell solver rel L2 0.02-0.05% on 3 val samples.
- 2026-11-09 19:52 IST: M2 prep done, NOTHING TRAINED. src/models.py (FNO 837,513 params, U-Net 1,067,241), scripts/train.py (resumable, val-only), scripts/evaluate.py (test needs --final), scripts/smoke.sh. Smoke runs (256 samples, 15 epochs) and 2-epoch timing runs at 4000 samples: FNO ~2.5-3.0 s/epoch, U-Net ~3.7-5.1 s/epoch, peak VRAM ~1.0 GB. Resume path tested (RESUMED from epoch 15). M1 closes when user approves subset/split. Gate (b) pending.
- 2026-11-09 20:07 IST: User approved gate (b) and the subset/split (train 4000 / val 500 / test 1000, x stride 4). M1 closed. Full training started: FNO then U-Net, seeds 0-2, 300 epochs each, scripts/train_all.sh, logs/train_*.log. Estimate ~1h50m.
- 2026-11-09 21:12 IST: Six main runs finished (FNO s0-2: 300 epochs, ~7 min each; U-Net s0-2: ~13.5 min each); no failures. Val-set evaluation and 3-case solver spot check done (reports/eval_val_*.json, spotcheck.png). Models FROZEN here, checkpoints chosen by val; test evaluation run once next.
- 2026-11-09 21:13 IST: Test evaluation run ONCE on frozen models (reports/eval_test_*.json). FNO whole-trajectory rel L2 0.0264-0.0271 (3 seeds), U-Net 0.0843-0.0941, copy-IC 1.158, IC-mean constant 0.456. Test-vs-train IC duplicate check run. M2 pending user review (physics checks done: mass drift + 3-case solver spot check).
- 2026-11-09 23:03 IST: User confirmed keeping the models; M2 CLOSED (work finished 21:13, T+6h09m). Extra validation (scripts/validate_m2.py): independent metric check matches (0.027654), raw-row mapping exact, time alignment min at offset 0, FNO translation-invariant, OOD behaviour measured. Latency/memory benchmark (scripts/bench_latency.py). Published comparison searched: no reliable apples-to-apples numbers (reports/published_comparison.md).
- 2026-11-09 23:09 IST: M3 DONE (T+8h05m, due 11 Nov 05:03). FastAPI app (app/main.py): /health, /model-info, /predict (+optional solver), OOD flag, input validation. 18 tests pass. Real local server check (scripts/live_check.py): live vs offline max abs diff 7.3e-07, same sha256, relL2 vs truth 0.01179. HTTP /predict median 74 ms (463 kB body), solver adds ~0.8 s. OOD flag: 0.2% false alarms on val, misses moderate degradations (amp x2 28% flagged, square wave 0%). Bugs found+fixed: torch version type in package, NaN input crashed the 422 handler.
- 2026-11-09 23:45 IST: M4 prep (no deploy, no token read): requirements-serve.txt (CPU torch), deploy/Dockerfile, deploy/SPACE_README.md, scripts/deploy_hf.py (--dry-run stages 12 files, 3.38 MB), .env.example + blank .env (git-ignored), root route. huggingface_hub 2.4.0 installed (small, lockfile updated). Waiting on gate (c) and the user filling .env.
- 2026-11-10 00:02 IST: Docker local test PASSED: image burgers-fno-local 319 MB built from the staged 12 files (python:3.13-slim + CPU torch 2.14.1); container up in <1 s, live_check: max abs diff vs offline 7.77e-07, same sha256, HTTP /predict median 64 ms, container memory 183 MiB. Container removed, image kept. Docker Desktop was started by Claude at 23:57 on user request and is still running. Waiting for user: .env filled + run deploy_hf.py (public Space, name burgers-fno-service).
- 2026-11-10 08:41 IST: deploy_hf.py run (user pre-authorised): HF 402 Payment Required for Docker Space on free tier. M4 BLOCKED (external). See WORKLOG.md.
- 2026-11-10 09:03 IST: M5 (front end) DONE locally: static HTML/JS served by the app, 19/19 real-browser checks, 32 tests pass. M6 README drafted (final live verification blocked with M4). See WORKLOG.md.
- 2026-11-11 09:25 IST: M4 re-scoped by user to a LOCAL deployment (127.0.0.1:8000, CPU settings). Verified: live_check PASS, 19/19 browser checks. Public live URL still not achieved (HF 402).
- 2026-11-11 09:58 IST: UI rebuilt as a single-screen dashboard with explanations in dialogs; 33 tests and 47/47 browser checks pass. Local server on 127.0.0.1:8000 serves it.
- 2026-11-11 10:15 IST: Self-check added to /predict and the page (54 percent of bad cases caught, 0.3 percent false alarms). Tests deliberately not re-run at the user request; 2 failures from the last run are unresolved or unconfirmed (see WORKLOG).
- 2026-11-11 10:30 IST: M7 recorded as a deferred PLAN (docs/M7_plan.md) at the user's request: local flux-form time-stepper with rollout training on broad solver-generated data; success criteria fixed in advance. Not started. Tests are not being run until the user says so.
- 2026-11-11 10:46 IST: Added PDF report, MCP server (/mcp, /agent.json) and the Agent-mode toggle. Syntax/import checks only; tests written but not run (user's instruction).
- 2026-11-12 00:29 IST: Tests 61/62 pass (see WORKLOG). M4 BEHIND (no public URL; local only). M6 due 09:03.
- 2026-11-12 00:38 IST: 62/62 tests pass; main pushed to GitHub. M4 (public live URL) still not met; M6 open.
- 2026-11-12 08:16 IST: M7 evaluated: mixed result, criterion not met, not deployed (README 11b). M6 finalisation (README, full tests, commit, push) in progress.
- 2026-11-12 08:19 IST: M6 DONE (local verification only): 62/62 tests pass, live_check PASS against the local server (max abs diff 7.32e-07), README final, main pushed (2e9a219). Public URL (M4) still not met.
- 2026-11-12 10:31 IST: M4 DONE (late): live on Render, https://burgers-fno-service.onrender.com. live_check PASS (max abs diff 7.94e-07, same weights hash), MCP + agent.json + origin check work publicly. /predict about 2.2 s, with solver about 9.2 s on the free tier.
