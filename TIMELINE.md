# TIMELINE

**T0 = 2026-11-09 15:03:49 +05:30 (IST)**, from `date` run in this session (UTC 09:33:49).
**Final commit due: T0+72h = 2026-11-12 15:03 IST.**

Absolute clock times are provisional: they are pure elapsed-time conversions and do not yet account for the user's availability blocks (to be supplied).

| Milestone | T+ deadline | Absolute deadline (IST) | Effective h (budget) | Actual completion | Status |
|---|---|---|---|---|---|
| M0 Discovery, environment, plan | T+6 | 2026-11-09 21:03 | ~1.5 | 2026-11-09 18:52 (T+3h48m) | DONE |
| M1 Data inspected, split, validated | T+14 | 2026-11-10 05:03 | ~3 | | IN PROGRESS (download running) |
| M2 Baselines, FNO trained, evaluated, physics checks | T+30 | 2026-11-10 21:03 | ~8 | | not started |
| M3 FastAPI backend working locally | T+38 | 2026-11-11 05:03 | ~3 | | not started |
| M4 Skeleton deploy, live URL with real inference | T+44 | 2026-11-11 11:03 | ~2 | | not started |
| M5 Front end polish, solver comparison, OOD warnings | T+54 | 2026-11-11 21:03 | ~4 | | not started |
| M6 Final verification, README, cleanup, FINAL COMMIT | T+66 | 2026-11-12 09:03 | ~3 | | not started |
| Buffer (no new features) | T+72 | 2026-11-12 15:03 | - | | - |

Cut list, in order (propose, user approves): (1) freehand drawing; (2) multiple viscosities; (3) UI polish; (4) extra architectures; (5) reduced training data / fewer epochs, documented honestly.
Never cut: baseline, held-out evaluation, physics checks, live inference check, honest README.

## Log
- 2026-11-09 15:03:49 IST: T0 recorded. Hardware/environment detected. No code written, nothing downloaded.
- 2026-11-09 15:15 IST: Python 3.13.3 venv created at .venv (git-ignored, no packages installed). .gitignore added.
- 2026-11-09 15:20 IST: Found that the pre-existing git repo was rooted at D:\Projects (parent of all the user's projects), not Pre-doc. My branch rename and remote add had briefly touched it; both were reverted (parent is back on `master`, no remote). Created a separate repo in Pre-doc (branch `main`, origin set, NOT pushed). Parent repo's index/HEAD tree is already damaged ("bad tree object HEAD"); left untouched. User stepping away ~30 min; push awaits go-ahead.
- 2026-11-09 15:28 IST: Install (torch 2.14.1+cu126 chosen by pip resolver) and 2.6 GB/8.2 GB downloads running in background (logs/install.log, logs/download.log). Range request tested on 1000 bytes: HTTP 206, bytes identical to on-disk bytes. Origin removed at user request (URL had come from the user first message); user adds the real remote at gate (c). scripts/chain.sh runs GPU check, commit (no push), size+md5 verify, inspect_h5 unattended. No training or scope change until M1 proposal is approved.
- 2026-11-09 18:40 IST: User returned. All background jobs had died at 15:53 (no log lines after that; Windows boot time shows 18:35, so the laptop slept/restarted - exact cause not confirmed). Resumed: torch wheel curl resumes from 910 MB, dataset download resumes from 1.478 GB after the wheel completes, chain restarted. User says laptop stays on for the next 10-12 h.
- 2026-11-09 18:52 IST: Install finished: torch 2.14.1+cu126, numpy 2.5.4, h5py 3.16.0, scipy 1.18.1, matplotlib 3.11.2, fastapi 0.162.2, uvicorn 0.54.0, pydantic 2.14.3, pytest 9.2.0, httpx 0.28.1 (44-line lockfile). GPU check passed (RTX 4050, torch reports 6.44 GB VRAM, not the 4 GB that Windows WMI reported). pytest: 5 passed. Lockfile torch line fixed (pip wrote a local file path). Dataset at ~24%, single downloader.
