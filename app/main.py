"""FastAPI service: initial condition in, FNO solution of 1D Burgers (nu = 0.01) out.

Run (repo root):  uvicorn app.main:app --host 127.0.0.1 --port 8000
Endpoints: GET /health, GET /model-info, POST /predict (see PredictRequest), POST /mcp (Model Context Protocol), GET /agent.json,
GET /runs/{id}.csv. Interactive docs at /docs. The web page is served at /.
"""
import hashlib
import os
import time
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from src.inference import ALLOWED_NX, DT, NT, NU_EFF, load_json, load_model, ood_check, predict, self_check
from src.solver import mass, solve
from . import mcp as mcp_server

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "model", "fno_s1.pt"))
STATS_PATH = os.environ.get("OOD_STATS_PATH", os.path.join(ROOT, "model", "ood_stats.json"))
CARD_PATH = os.environ.get("MODEL_CARD_PATH", os.path.join(ROOT, "model", "model_card.json"))
PRESETS_PATH = os.environ.get("PRESET_ERRORS_PATH", os.path.join(ROOT, "model", "preset_errors.json"))
MAX_ABS_INPUT = 100.0   # hard input sanity bound, far above anything physical here; beyond it we refuse


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))   # more threads were slower in the benchmark
    app.state.started = time.time()
    app.state.model = app.state.error = None
    app.state.measured = {}
    try:
        app.state.model, app.state.meta = load_model(MODEL_PATH)
        app.state.stats = load_json(STATS_PATH)
        app.state.card = load_json(CARD_PATH)
        app.state.sha = hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest()
    except Exception as e:   # keep the process up so /health can say what is wrong
        app.state.error = repr(e)
    try:
        app.state.measured = load_json(PRESETS_PATH)          # optional: measured error of each preset
    except Exception:
        app.state.measured = {}
    yield


app = FastAPI(title="burgers-fno-service", version="0.2.0", lifespan=lifespan,
              description="1D viscous Burgers (nu=0.01, periodic) solved by a Fourier Neural Operator. Also an MCP server at /mcp.")
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


def _model_info_dict():
    return {**app.state.card, "runtime": {"torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads(),
                                         "served_sha256": app.state.sha, "ood_train_range": app.state.stats["train_range"]}}


def _compute(u0, include_solver):
    """The one place a forecast is computed; /predict and the MCP tools both use it, so they cannot disagree."""
    u0 = np.asarray(u0, dtype=np.float64)
    nx = len(u0)
    t0 = time.perf_counter()
    u = predict(app.state.model, u0)                      # (201, nx) float32
    fno_s = time.perf_counter() - t0
    t = np.arange(NT) * DT
    res = {"u0": u0, "x": (np.arange(nx) + 0.5) / nx, "t": t, "u": u, "fno_s": fno_s, "s": None, "solver_s": None, "rel": None, "per_t": None,
           "mass_fno": float(np.abs(mass(u.astype(np.float64)) - mass(u0[None])[0]).max()), "mass_solver": None,
           "ood": ood_check(u0, app.state.stats), "self_check": self_check(u0, u, app.state.stats),
           "model": {"name": app.state.card["name"], "sha256_12": app.state.sha[:12], "trained_at_nx": 256, "input_nx": nx},
           "equation": app.state.card["equation"]}
    if include_solver:
        t0 = time.perf_counter()
        s = solve(u0, NU_EFF, t, L=1.0, a=1.0)            # (201, nx)
        res["solver_s"] = time.perf_counter() - t0
        res["s"] = s
        res["mass_solver"] = float(np.abs(mass(s) - mass(u0[None])[0]).max())
        res["per_t"] = np.sqrt(((u - s) ** 2).sum(axis=1) / np.maximum((s ** 2).sum(axis=1), 1e-30))
        res["rel"] = float(np.sqrt(((u - s) ** 2).sum() / (s ** 2).sum()))
    return res


class _Svc:
    """What the MCP tools may use (kept small on purpose)."""
    @property
    def measured(self):
        return app.state.measured

    def ready(self):
        return app.state.model is not None

    def compute(self, u0, include_solver):
        return _compute(u0, include_solver)

    def model_info(self):
        return _model_info_dict()

    def health(self):
        ok = app.state.model is not None
        return {"status": "ok" if ok else "degraded", "model_loaded": ok, "uptime_s": round(time.time() - app.state.started, 1), "error": None if ok else app.state.error}


SVC = _Svc()
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
    v = int(max(os.path.getmtime(os.path.join(STATIC_DIR, n)) for n in ("index.html", "style.css", "lib.js", "app.js", "agent.js", "pdf.js")))
    for name in ("style.css", "lib.js", "app.js"):
        html = html.replace(f'"{name}"', f'"/static/{name}?v={v}"')
    return html


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def index():
    return HTMLResponse(_index_html())


@app.get("/api")
def api_root():
    return {"service": "burgers-fno-service", "docs": "/docs", "ui": "/", "mcp": "/mcp", "agent_manifest": "/agent.json",
            "endpoints": ["GET /health", "GET /model-info", "POST /predict", "POST /mcp", "GET /agent.json"]}


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
    return _model_info_dict()


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    _require_model()
    r = _compute(req.u0, req.include_solver)
    out = {
        "x": _round(r["x"]), "t": _round(r["t"]),
        "fno": {"u": _round(r["u"]), "seconds": r["fno_s"], "mass_drift": r["mass_fno"]},
        "ood": r["ood"], "self_check": r["self_check"], "model": r["model"], "equation": r["equation"],
        "verdict": mcp_server.verdict(r["ood"], r["self_check"], r["rel"]),
    }
    if r["s"] is not None:
        out["solver"] = {"u": _round(r["s"]), "seconds": r["solver_s"], "mass_drift": r["mass_solver"],
                         "scheme": "finite volume, MUSCL-minmod + Rusanov flux, SSP-RK3 (src/solver.py)",
                         "rel_l2_fno_vs_solver_whole": r["rel"], "rel_l2_fno_vs_solver_per_t": _round(r["per_t"])}
    return out


# ---------------------------------------------------------------- machine-friendly interfaces (MCP + manifest + run data)
def _origin_ok(request: Request):
    """MCP over HTTP must reject foreign browser origins (DNS-rebinding defence). Non-browser clients send no Origin header."""
    origin = request.headers.get("origin")
    if not origin:
        return True
    host = urlparse(origin).hostname
    return host in ("127.0.0.1", "localhost", "::1") or host == request.url.hostname


@app.post("/mcp", include_in_schema=False)
async def mcp_endpoint(request: Request):
    if not _origin_ok(request):
        return JSONResponse(status_code=403, content={"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": "Origin not allowed"}})
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
    if isinstance(payload, list):
        out = [r for r in [await run_in_threadpool(mcp_server.handle, m, SVC) for m in payload] if r is not None]
        return JSONResponse(out) if out else Response(status_code=202)
    resp = await run_in_threadpool(mcp_server.handle, payload, SVC)
    return Response(status_code=202) if resp is None else JSONResponse(resp)


@app.get("/mcp", include_in_schema=False)
@app.delete("/mcp", include_in_schema=False)
def mcp_no_stream():
    return Response(status_code=405, headers={"Allow": "POST"})       # no server-initiated stream and no sessions


@app.get("/agent.json")
def agent_manifest(request: Request):
    return mcp_server.manifest(str(request.base_url), SVC)


@app.get("/runs/{run_id}.csv", include_in_schema=False)
def run_csv(run_id: str):
    run = mcp_server.RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown run_id (runs are kept in memory; call the predict tool again)")
    return PlainTextResponse(mcp_server.run_csv(run), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="burgers_run_{run_id}.csv"'})
