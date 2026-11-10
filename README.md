# burgers-fno-service

PDE solutions as a service: a Fourier Neural Operator (FNO) trained on a PDEBench 1D Burgers dataset, served by a
FastAPI app with a built-in web page. You give it an initial condition; it returns the solution over time, next to a
numerical solver's solution, an error map, and a warning when the input is outside what the model was trained on.

> **Status (10 Nov 2026):** the model, API and web page work **locally and in a local Docker container**. It is
> **not deployed**: Hugging Face now requires a paid PRO subscription for Docker Spaces on the free CPU tier (the API
> returned `402 Payment Required`), and I did not pay or switch hosts without you. There is **no live URL** yet.
> See [Deployment](#deployment).

## 1. The problem and the equation

The 1D viscous Burgers equation, on the periodic domain x in [0, 1), t in [0, 2]:

```
du/dt + d/dx( u^2 / 2 )  =  (nu / pi) * d^2u/dx^2        with nu = 0.01
```

Note the factor **nu/pi**, not plain nu. That is the convention in the PDEBench data. It was **checked against the
data**, not assumed: our solver, started from stored initial conditions, was compared with the stored solutions under
four conventions on 3 validation samples (256 cells, relative L2 over the whole trajectory):

| convention | relative L2 |
|---|---|
| `a=1`, nu_eff = nu/pi (used) | 0.0022 to 0.0026 |
| `a=1`, nu_eff = nu | 0.066 to 0.164 |
| `a=2`, nu_eff = nu | 0.157 to 0.634 |
| `a=2`, nu_eff = nu/pi | 0.173 to 0.676 |

(`a` multiplies the flux, i.e. `a=2` is the `d/dx(u^2)` form. Numbers: `reports/solver_check_A.txt`.)

Task: initial condition `u(x,0)` (256 points) -> all 201 frames `u(x, 0.01 k)`, `k = 0..200`, in one forward pass.

## 2. Data

- **PDEBench** (Takamoto et al., NeurIPS 2022 Datasets & Benchmarks), DaRUS DOI
  [10.18419/darus-2986](https://doi.org/10.18419/darus-2986), **licence CC BY 4.0** (read from the DaRUS metadata).
  Attribution is required; please cite the PDEBench paper if you reuse the data.
- File used: `1D_Burgers_Sols_Nu0.01.hdf5`, 8,232,968,312 bytes, md5 `e6d9a4f62baf9a29121a816b919e2770`. Size and md5 were
  checked against the DaRUS API. Layout: `tensor` (10000, 201, 1024) float32, `x-coordinate` (1024), `t-coordinate` (202, one
  more than the frames; frame k is t = 0.01 k, confirmed by the solver), attribute `Nu = 0.01`. No NaN or inf. Values in
  [-3.35, 3.41].
- Not committed (`data/` is git-ignored). `scripts/download_data.py` downloads it with resume and verifies size and md5.
- **Split** (by sample, seed 0, `src/data.py`): 1,000 test, 500 validation, the rest a training pool of which the **first 4,000**
  are used. The split file is `splits/burgers_nu0.01_seed0.json`; a test checks it is disjoint and reproducible.
  Every 4th grid point is kept (256 points, point samples, no anti-aliasing filter).
- **Leakage check:** no validation or test initial condition is within 0.1% (relative L2) of any training one (nearest: 2.4% for test).
- **Test-set discipline:** the test set was written to disk with the other splits, then not used for any decision. It was
  evaluated **once**, after the models and checkpoints were frozen. The validation-based checkpoint choice makes
  *validation* numbers slightly optimistic; test numbers are not affected by it. Full disclosure: before that, the subset script printed the test array's min, max and NaN count, and
  the index-mapping and duplicate checks read test *initial conditions* (never test solutions for any model decision). A later "t = 0 skip connection"
  idea was **not** tried so the test run stays valid.

## 3. Models and training

- **FNO** (`src/models.py`, written from scratch, follows the FNO paper's structure; not compared line by line with the
  reference implementation): lift (u, x) -> 64 channels, 4 spectral layers keeping 24 Fourier modes + 1x1 skip convolutions,
  GELU, head 64 -> 128 -> 201 (time as output channels). **837,513** parameters.
- **U-Net baseline** (same file): 3 down/up levels, base 32 channels, circular padding. **1,067,241** parameters.
- Training (`scripts/train.py`): relative L2 loss over the whole trajectory, Adam lr 1e-3 with cosine decay, weight decay
  1e-5, batch 32, **300 epochs**, 4,000 training samples, 3 seeds (0, 1, 2) each, checkpoint with the lowest validation error.
  FNO ~1.4 s/epoch (about 7 min per run), U-Net ~2.8 s/epoch (about 13.5 min per run), about 1 GB GPU memory. The runs are resumable.
  Logs: `logs/train_*.log` (git-ignored; the summarised results are in `reports/`).
- Both models were still improving at the end (best epoch 297 or 298 of 300), so they are probably not fully trained.

## 4. Results (test set, 1,000 held-out simulations)

Metric: **relative L2** per sample, `||pred - true||_2 / ||true||_2`, norm over (t, x), averaged over samples; spread is the
sample standard deviation over the 3 training seeds (`src/models.py::rel_l2`, `scripts/evaluate.py`).

| model | mean | median | 95th percentile |
|---|---|---|---|
| copy the initial condition | 1.158 | 1.033 | 2.617 |
| per-sample constant = initial-condition mean | 0.456 | 0.382 | 0.985 |
| U-Net | 0.0904 +- 0.0053 | 0.0690 | 0.2710 |
| **FNO** (mean +- sd over 3 seeds) | **0.0267 +- 0.0004** | 0.0211 | 0.0654 |

The FNO is about **3.4x** more accurate than the U-Net, with a spread far smaller than the gap. Validation shows the same
(FNO 0.0279 +- 0.0004, U-Net 0.0966 +- 0.0053). The deployed model is **seed 1**, listed separately and *not* a best-of choice:
test 0.0264, validation 0.0277 (seeds 0 and 2 test 0.0271 and 0.0265).

**Error by time** (mean over seeds, test; the time is t = 0.01 k):

| k (t) | 1 (0.01) | 10 (0.1) | 50 (0.5) | 100 (1.0) | 200 (2.0) |
|---|---|---|---|---|---|
| copy the initial condition | 0.084 | 0.589 | 1.281 | 1.604 | 2.013 |
| IC-mean constant | 0.680 | 0.628 | 0.441 | 0.336 | 0.236 |
| U-Net | 0.0279 | 0.0495 | 0.0745 | 0.1020 | 0.1399 |
| FNO | 0.0137 | 0.0222 | 0.0279 | 0.0268 | 0.0392 |

**Near vs away from shocks** (pooled relative L2; "near" = within 5 grid points of a point where the true slope is below -20,
about 2.2% of points; this rule was chosen by me and not tuned):

| | near shocks | away |
|---|---|---|
| FNO | 0.1342 +- 0.0005 | 0.0421 +- 0.0005 |
| U-Net | 0.2363 +- 0.0050 | 0.0956 +- 0.0041 |

**Mass conservation** (max over time of |M(t) - M(0)|, M = sum(u) dx, mean over samples): FNO 0.0080 +- 0.0001, U-Net
0.0285 +- 0.0023, reference data 7.9e-5, our solver about 1e-13. **The networks do not conserve mass** (about 100x the data's drift for the FNO).

**Solver as a reference:** on 3 validation samples at the data's own 1024 cells the solver matches the stored solution to
relative L2 0.0002 to 0.0005 (`reports/solver_check_B.txt`). That supports the equation and the time axis; it is partly shared numerics
with the data generator, so it is not independent proof. A 3-case FNO-vs-solver spot check is in `reports/spotcheck.png`.

## 5. Where it fails (measured)

Details in `reports/validate_m2.txt` and `reports/ood_eval.txt`; FNO error against solver truth on synthetic inputs:

| input | FNO relative L2 | OOD flag raised |
|---|---|---|
| in-family control | 0.027 | 0.2% (false alarms on 500 validation inputs) |
| smooth bump | 0.017 | 0% |
| square wave | 0.110 | 0% |
| amplitude x2 / x3 | 0.100 / 0.195 | 28% / 64% |
| mean offset +1.5 / +3 | 0.055 / 0.089 | 12% / 86% |
| white noise | 0.708 | 100% |
| 40 or 100 waves | about 1.5 (worse than predicting zero) | 100% |
| hand-drawn-like curves, snapped to <= 8 waves | median 0.078, p90 0.130 | 0% |

- **Shocks:** error about 3x larger near them (section 4).
- **t = 0:** the output reproduces its own input to only about 1.6% (relative), i.e. the first frame is not exact.
- **Mass:** not conserved (section 4).
- **Scope of training:** one viscosity (0.01), one family of smooth sine-mixture initial conditions, 256 points, t in [0, 2]. Anything else is untested.
- **The out-of-distribution flag is weak:** it compares amplitude, mean and high-frequency energy with the training range. It catches noise and very wiggly inputs, but it
  *misses* moderate failures (square wave, hand-drawn curves). **No flag does not mean accurate.** The web page adds a second warning when FNO and solver disagree by more than 6%.
- **Resolution:** works at 256 (trained) and, zero-shot, at 512 points (validation relative L2 0.0271); other lengths are rejected. The U-Net fails at 512 (0.56).
- **Robustness checks that passed:** an independent numpy recomputation of the metric matches (0.027654); processed arrays equal the raw rows;
  feeding another sample's initial condition gives error 1.92 (the model uses its input); 1% input noise changes the output by 0.24%;
  the error is smallest at zero time offset; shifting the wave around the ring leaves the error unchanged.

## 6. What is verified, and what is not

| Verified (evidence in this repo) | Not verified |
|---|---|
| Dataset size and md5; split disjointness; no near-duplicate initial conditions | **Any live deployment** (none exists) |
| Equation convention (nu/pi) against the data | **Published comparison**: no reliable like-for-like numbers found. A garbled table suggested PDEBench FNO nRMSE about 7.8e-3 and U-Net about 0.22 for nu = 0.01, but the labelling is doubtful and the protocols differ (autoregressive, more data, 1024 points). See `reports/published_comparison.md`. **Check the PDEBench appendix yourself before quoting any comparison.** |
| Test-set errors, 3 seeds, one evaluation | Behaviour at other viscosities or other initial-condition families |
| API matches the offline checkpoint (max abs difference 7e-7 locally and in Docker) | Latency and memory on a free cloud host (laptop numbers only) |
| Web page behaviour in a real Chromium browser (Edge, headless): run, OOD banner, 375 px width with no overflow, keyboard skip link, mouse drawing, paste errors, CSV/JSON downloads, error state | Firefox, Safari, touch input, screen readers |
| WCAG AA contrast of the colour tokens (computed) | Colour-blind simulation of the heatmaps |
| Docker image builds and runs locally (319 MB, 183 MiB memory in use) | `deploy/Dockerfile.render` and `render.yaml` (not built or tested) |
| 32 automated tests | Bitwise reproducibility across GPUs |

## 7. Speed and memory (this laptop: i7-13650HX, RTX 4050 Laptop 6 GB, 15.6 GB RAM)

- FNO inference, one input, CPU, 1 or 2 threads: 1.8 ms typical (p95 2.1 ms); 4.5 ms for 8 inputs. The GPU is no faster for one input (4.2 ms). 14 threads were *slower* (3.0 ms), so the app defaults to 2 threads.
- Numerical solver (numpy, 256 cells, one input): about 0.4 to 0.8 s; the FNO is roughly 200x faster, but the solver is not optimised.
- Over HTTP locally: `/predict` about 64 to 74 ms median, mostly JSON (463 kB); with the solver about 0.9 s.
- Memory: about 530 MB with the large CUDA build of torch on the laptop; 183 MiB for the CPU-only Docker container.
- Cold start of the process including `import torch`: about 1.5 s of import. Free-host cold starts are unmeasured.

## 8. Using it

Web page at `/`, API docs at `/docs`.

```
POST /predict   {"u0": [256 numbers (or 512)], "include_solver": true}
GET  /model-info    GET /health    GET /api
```

`/predict` returns `x`, `t`, `fno.u` (201 x nx), optionally `solver.u`, `ood` (`is_ood`, `reasons`, features), and model identification
(weights sha256). Inputs must be finite, |u0| <= 100, length 256 or 512; otherwise HTTP 422.

## 9. Setup (Windows, Python 3.13) and reproduction

```bash
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock.txt   # CUDA 12.6 torch 2.14.1; see scripts/install_env.sh for the resumable variant
.venv\Scripts\python scripts\download_data.py                  # 8.23 GB, resumable, md5-verified
.venv\Scripts\python scripts\make_subset.py --n-train 4000 --x-stride 4
bash scripts/train_all.sh                                      # 6 runs, about 1h50m on the laptop above
.venv\Scripts\python scripts\build_model_package.py            # packs model/fno_s1.pt and OOD statistics
.venv\Scripts\python -m uvicorn app.main:app --port 8000       # then open http://127.0.0.1:8000
.venv\Scripts\python -m pytest                                 # tests (the browser tests need Node >= 22 and Edge or Chrome)
```

Key versions (full list in `requirements.lock.txt`; serving subset in `requirements-serve.txt`): torch 2.14.1 (cu126 for training,
CPU for serving), numpy 2.5.4, h5py 3.16.0, scipy 1.18.1, matplotlib 3.11.2, fastapi 0.162.2, starlette 1.10.0, uvicorn 0.54.0,
pydantic 2.14.3, pytest 9.2.0, huggingface_hub 2.4.0. The serving pins are not hash-locked.

`scripts/evaluate.py` refuses to open the test split without `--final`. `scripts/validate_m2.py`, `scripts/bench_latency.py`,
`scripts/ood_eval.py`, `scripts/snap_eval.py`, `scripts/ui_check.mjs` reproduce the other reports.

## 10. Deployment

- **Hugging Face Spaces (Docker, free CPU):** attempted on 2026-11-10 with `scripts/deploy_hf.py`. The API answered
  `402 Payment Required` ("hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription"). Nothing was created. The script
  and `deploy/Dockerfile` are ready; the Docker image was built and passed a live check locally (live-vs-offline max abs difference 7.77e-7). A Static
  Space is free but cannot run this API.
- **Render (free web service), untested:** `render.yaml` and `deploy/Dockerfile.render` are prepared. It needs this repository on GitHub (not pushed). Reported free-tier
  limits from web searches, **unverified**: 512 MB RAM (one source says 256 MB), 0.1 CPU, spin-down after 15 minutes idle, about a minute to wake. The container used 183 MiB.
- **Local Docker:** `docker build -f deploy/Dockerfile -t burgers-fno <staged folder from scripts/deploy_hf.py --dry-run>`, then `python scripts/live_check.py http://127.0.0.1:<port>`.
- `scripts/live_check.py <url>` is the end-to-end check to run against any deployment: it compares the served output with the original training checkpoint's output.

## 11. Limits of this work

- **Time:** a 72-hour window (start 9 Nov 15:03 IST) with limited hands-on time; M0-M3 and the front end were finished ahead of the plan, deployment was blocked by hosting cost.
- **Compute:** one laptop GPU; one data file; 4,000 of the 10,000 training simulations used; 256 of 1,024 grid points; 300 epochs, not converged; 3 seeds.
- **Scope kept fixed:** no retraining after the first test evaluation, no architecture change, one viscosity.
- **Code licence:** not chosen yet (data: CC BY 4.0, PDEBench).

## 12. Layout

```
app/            FastAPI app and static web page (index.html, app.js, lib.js, style.css)
src/            models.py, solver.py, data.py, inference.py
scripts/        download, subset, train, evaluate, validation, benchmarks, deploy, browser checks
tests/          pytest (API, front end, contrast, solver/data) and tests/js (Node unit tests)
model/          packaged FNO (seed 1), OOD statistics, model card (all numbers read from reports/)
reports/        all measured results, screenshots; splits/ the split file; docs/ a plain-English explainer
deploy/         Dockerfiles and the Space README; TIMELINE.md and WORKLOG.md log the schedule and decisions
```

## References

- Takamoto et al., *PDEBench: An Extensive Benchmark for Scientific Machine Learning*, NeurIPS 2022 (Datasets & Benchmarks), arXiv:2210.07182.
- Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021.
