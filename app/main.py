"""FastAPI service: initial condition in, FNO solution of 1D Burgers (nu = 0.01) out.

Run (repo root):  uvicorn app.main:app --host 127.0.0.1 --port 8000
Endpoints: GET /health, GET /model-info, POST /predict (see PredictRequest). Interactive docs at /docs.
"""
import hashlib
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator

from src.inference import ALLOWED_NX, DT, NT, NU_EFF, load_json, load_model, ood_check, predict, self_check
from src.solver import mass, solve

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "model", "fno_s1.pt"))
STATS_PATH = os.environ.get("OOD_STATS_PATH", os.path.join(ROOT, "model", "ood_stats.json"))
CARD_PATH = os.environ.get("MODEL_CARD_PATH", os.path.join(ROOT, "model", "model_card.json"))
MAX_ABS_INPUT = 100.0   # hard input sanity bound, far above anything physical here; beyond it we refuse


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))   # more threads were slower in the benchmark
    app.state.started = time.time()
    app.state.model = app.state.error = None
    try:
        app.state.model, app.state.meta = load_model(MODEL_PATH)
        app.state.stats = load_json(STATS_PATH)
        app.state.card = load_json(CARD_PATH)
        app.state.sha = hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest()
    except Exception as e:   # keep the process up so /health can say what is wrong
        app.state.error = repr(e)
    yield


app = FastAPI(title="burgers-fno-service", version="0.1.0", lifespan=lifespan,
              description="1D viscous Burgers (nu=0.01, periodic) solved by a Fourier Neural Operator.")
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    # The default handler echoes the offending input back; for NaN/inf that is not even valid JSON (500).
    return JSONResponse(status_code=422, content={"detail": [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]})


class PredictRequest(BaseModel):
    u0: list[float] = Field(..., description="Initial condition u(x,0) on a uniform periodic cell-centred grid over [0,1). "
                                              f"Length must be one of {list(ALLOWED_NX)}.")
    include_solver: bool = Field(False, description="Also run the reference numerical solver (slower, about 0.4 s at 256 points).")

    @field_validator("u0")
    @classmethod
    def _check(cls, v):
        if len(v) not in ALLOWED_NX:
            raise ValueError(f"u0 must have length {list(ALLOWED_NX)}, got {len(v)}")
        a = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            raise ValueError("u0 contains NaN or infinity")
        if np.abs(a).max() > MAX_ABS_INPUT:
            raise ValueError(f"|u0| must be <= {MAX_ABS_INPUT}")
        return v


def _require_model():
    if app.state.model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {app.state.error}")


def _round(a):
    return np.round(np.asarray(a, dtype=np.float64), 6).tolist()


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; "
       "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    if request.url.path == "/" or request.url.path.startswith("/static/"):   # /docs (Swagger) needs its own CDN assets
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["Cache-Control"] = "no-cache"                           # always revalidate: no stale stylesheet next to a new page
    return resp


def _index_html():
    """index.html uses relative asset links so it also works from file:// or an editor preview; here they become absolute and versioned."""
    html = open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8").read()
    v = int(max(os.path.getmtime(os.path.join(STATIC_DIR, n)) for n in ("index.html", "style.css", "lib.js", "app.js")))
    for name in ("style.css", "lib.js", "app.js"):
        html = html.replace(f'"{name}"', f'"/static/{name}?v={v}"')
    return html


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def index():
    return HTMLResponse(_index_html())


@app.get("/api")
def api_root():
    return {"service": "burgers-fno-service", "docs": "/docs", "ui": "/",
            "endpoints": ["GET /health", "GET /model-info", "POST /predict"]}


@app.get("/health")
def health():
    ok = app.state.model is not None
    body = {"status": "ok" if ok else "degraded", "model_loaded": ok, "uptime_s": round(time.time() - app.state.started, 1)}
    if not ok:
        raise HTTPException(status_code=503, detail={**body, "error": app.state.error})
    return body


@app.get("/model-info")
def model_info():
    _require_model()
    return {**app.state.card, "runtime": {"torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads(),
                                         "served_sha256": app.state.sha, "ood_train_range": app.state.stats["train_range"]}}


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    _require_model()
    u0 = np.asarray(req.u0, dtype=np.float64)
    nx = len(u0)
    t0 = time.perf_counter()
    u = predict(app.state.model, u0)                      # (201, nx) float32
    fno_ms = (time.perf_counter() - t0) * 1000
    t = np.arange(NT) * DT
    out = {
        "x": _round((np.arange(nx) + 0.5) / nx), "t": _round(t),
        "fno": {"u": _round(u), "seconds": fno_ms / 1000, "mass_drift": float(np.abs(mass(u.astype(np.float64)) - mass(u0[None])[0]).max())},
        "ood": ood_check(u0, app.state.stats),
        "self_check": self_check(u0, u, app.state.stats),
        "model": {"name": app.state.card["name"], "sha256_12": app.state.sha[:12], "trained_at_nx": 256, "input_nx": nx},
        "equation": app.state.card["equation"],
    }
    if req.include_solver:
        t0 = time.perf_counter()
        s = solve(u0, NU_EFF, t, L=1.0, a=1.0)            # (201, nx)
        sec = time.perf_counter() - t0
        err_t = np.sqrt(((u - s) ** 2).sum(axis=1) / np.maximum((s ** 2).sum(axis=1), 1e-30))
        out["solver"] = {"u": _round(s), "seconds": sec, "mass_drift": float(np.abs(mass(s) - mass(u0[None])[0]).max()),
                         "scheme": "finite volume, MUSCL-minmod + Rusanov flux, SSP-RK3 (src/solver.py)",
                         "rel_l2_fno_vs_solver_whole": float(np.sqrt(((u - s) ** 2).sum() / (s ** 2).sum())),
                         "rel_l2_fno_vs_solver_per_t": _round(err_t)}
    return out
