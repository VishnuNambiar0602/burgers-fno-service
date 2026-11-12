"""A minimal Model Context Protocol (MCP) server over Streamable HTTP (JSON-RPC 2.0), plus the machine-readable manifest.

No SDK is used: the endpoint (POST /mcp, see app/main.py) answers initialize, ping, tools/list, tools/call, resources/list and
resources/read with plain JSON responses (no SSE stream, no sessions). Tools are read-only: they compute forecasts, they change nothing.
Everything an agent receives is also derivable from the REST API; MCP is just the standard doorway.
"""
import json
import time
import uuid
from collections import OrderedDict

import numpy as np

from . import limits, waves

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "burgers-fno-service", "title": "Burgers Wave Predictor", "version": "0.2.0"}
RUN_LIMIT = 20
DISAGREE = 0.06                       # same threshold as the web page's amber warning
RUNS = OrderedDict()                  # run_id -> stored arrays (in memory, newest last, at most RUN_LIMIT)
TIMES_DEFAULT = [0.0, 0.5, 1.0, 2.0]

INSTRUCTIONS = (
    "Predicts how a wave evolves under the 1D viscous Burgers equation u_t + (u^2/2)_x = (nu/pi) u_xx (nu = 0.01, periodic on [0,1), t in [0,2]) "
    "with a trained Fourier Neural Operator (the AI) and, optionally, a classic numerical solver (the reference). Typical use: call list_presets, "
    "then predict with a preset, sine components or your own 256 values, and READ verdict.level and verdict.best_answer before using any number. "
    "If verdict.best_answer is 'solver', use the solver's profiles (get_profile), not the AI's. Warnings are not guarantees: no warning does not prove "
    "the AI is accurate. Call explain('limits') once to see the known failure modes.")

TEXTS = {
    "equation": "u_t + d/dx(u^2/2) = (nu/pi) d2u/dx2 with nu = 0.01 on the periodic domain x in [0,1), t in [0,2]. u(x,t) is the wave height. The first term is how fast "
                "the height changes, the second is nonlinear transport (high parts move faster and steepen into shocks), the right side is viscosity that smooths sharp changes. "
                "The factor is nu/pi (not nu): that is the convention of the PDEBench data the AI learned from, checked against the data.",
    "limits": "Known failure modes of the AI (measured): (1) it was trained on one family of smooth sine-mixture waves at one viscosity and 256 points; unfamiliar waves "
              "can be badly wrong (hand-drawn-like curves: 75% of cases more than 6% off; random sine mixes: 52%); (2) errors are about 3x larger next to shocks; (3) it does not "
              "conserve total mass (drift ~100x the data's); (4) frame t=0 reproduces the input only to ~1.6%; (5) the input-range warning misses moderate problems such as a "
              "square wave; the self-check catches about half of the cases where the AI is >6% off (0.3% false alarms). The classic solver is the reference. Never rely on the AI "
              "alone when verdict.level is 'warn' or 'bad'.",
    "reading_results": "predict returns difference_percent (relative L2 gap between AI and solver over the whole run; typically 1-3% for familiar waves), a verdict, the "
                       "out-of-distribution result (ood: input outside the training range), the self_check (the AI's own forecast failing basic sanity checks), timings, mass drift, "
                       "small profiles at t = 0, 0.5, 1, 2 (32 points of x), and a run_id. Use get_profile for other times and the link links.csv for every number.",
    "verdicts": "verdict.level: 'good' = AI within 6% of the solver and no warning (still not a guarantee); 'warn' = no warning but AI and solver disagree by more than 6% "
                "(use the solver); 'bad' = out-of-distribution input or the AI failed its own sanity checks (use the solver); 'unchecked' = solver was not run and no warning fired. "
                "verdict.best_answer is 'ai' or 'solver'.",
}


def verdict(ood, sc, rel):
    """Same decision logic as the web page's coloured strip."""
    pct = None if rel is None else 100 * rel
    if ood["is_ood"]:
        return {"level": "bad", "best_answer": "solver", "text": "Unfamiliar starting wave: do not trust the AI forecast here. The classic solver is still reliable.",
                "reasons": ood["reasons"]}
    if sc and sc["flagged"]:
        return {"level": "bad", "best_answer": "solver", "text": "The AI's forecast fails its own sanity checks, so it is probably wrong here. Use the classic solver's answer.",
                "reasons": sc["reasons"]}
    if rel is None:
        return {"level": "unchecked", "best_answer": "ai", "text": "No warning was raised, but the solver was not run, so the AI forecast is unchecked.", "reasons": []}
    if rel > DISAGREE:
        return {"level": "warn", "best_answer": "solver", "text": f"The AI and the solver disagree by {pct:.1f}%. Treat the AI forecast with caution.", "reasons": []}
    return {"level": "good", "best_answer": "ai", "text": f"The AI matches the classic solver within {pct:.1f}%. No warning was raised (not a guarantee for every wave).", "reasons": []}


# ------------------------------------------------------------------ building inputs
def _num(v, name, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
        raise ValueError(f"{name} must be a finite number")
    if lo is not None and v < lo or hi is not None and v > hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return float(v)


def build_input(args):
    """Exactly one of: preset (+seed), components (+mean), u0 -> (u0 array, source, label)."""
    given = [k for k in ("preset", "components", "u0") if args.get(k) is not None]
    if len(given) != 1:
        raise ValueError("provide exactly one of 'preset', 'components' or 'u0'")
    if given[0] == "preset":
        key = args["preset"]
        if key not in waves.PRESET_INFO:
            raise ValueError(f"unknown preset '{key}'; choose one of {list(waves.PRESET_INFO)}")
        seed = int(_num(args.get("seed", 1), "seed", 1, 10 ** 9))
        return waves.preset(key, seed), "preset", f"{waves.PRESET_INFO[key]['label']}" + (f" (seed {seed})" if key in ("random", "noise") else "")
    if given[0] == "components":
        comps = args["components"]
        if not isinstance(comps, list) or not comps or len(comps) > 8:
            raise ValueError("'components' must be a list of 1 to 8 objects {amp, k, phase}")
        clean = []
        for i, c in enumerate(comps):
            if not isinstance(c, dict):
                raise ValueError(f"components[{i}] must be an object")
            k = _num(c.get("k"), f"components[{i}].k", 1, 60)
            if k != int(k):
                raise ValueError(f"components[{i}].k must be a whole number of waves")
            clean.append({"amp": _num(c.get("amp"), f"components[{i}].amp", -20, 20), "k": int(k), "phase": _num(c.get("phase", 0.0), f"components[{i}].phase", -50, 50)})
        mean = _num(args.get("mean", 0.0), "mean", -20, 20)
        u0 = waves.sine_mix(clean, mean)
        if not np.any(np.abs(u0) > 1e-12):
            raise ValueError("the wave is identically zero")
        return u0, "components", "Sine mix"
    u = args["u0"]
    if not isinstance(u, list) or len(u) not in (256, 512):
        raise ValueError("'u0' must be a list of exactly 256 (or 512) numbers")
    arr = np.array([_num(v, "u0[i]", -100, 100) for v in u], dtype=np.float64)
    return arr, "u0", "Supplied values"


def _r(a, d=5):
    return np.round(np.asarray(a, dtype=np.float64), d).tolist()


# ------------------------------------------------------------------ tools
def tool_list_presets(args, svc):
    out = []
    for key, info in waves.PRESET_INFO.items():
        m = svc.measured.get(key)
        out.append({"key": key, "label": info["label"], "description": info["description"], "stress_test": info["stress_test"],
                    "measured": None if not m else {"ai_vs_solver_relative_l2_median": m[0], "ai_vs_solver_relative_l2_max": m[1], "input_range_warning_rate": m[2]}})
    return {"presets": out, "note": "measured = our own tests of this preset (reports/preset_errors.json); random uses 40 draws. Seeds are reproducible here but differ from the web page's."}


def tool_make_wave(args, svc):
    u0, source, label = build_input(args)
    return {"label": label, "source": source, "points": len(u0), "max_abs": float(np.abs(u0).max()), "mean": float(u0.mean()), "u0": _r(u0, 6)}


def tool_predict(args, svc):
    u0, source, label = build_input(args)
    solver = args.get("include_solver", True)
    if not isinstance(solver, bool):
        raise ValueError("'include_solver' must be true or false")
    res = svc.compute(u0, solver)
    v = verdict(res["ood"], res["self_check"], res["rel"])
    run_id = uuid.uuid4().hex[:12]
    RUNS[run_id] = {"created": time.time(), "u0": u0, "x": res["x"], "t": res["t"], "fno": res["u"].astype(np.float32), "solver": None if res["s"] is None else res["s"].astype(np.float32)}
    while len(RUNS) > RUN_LIMIT:
        RUNS.popitem(last=False)
    prof = _profiles(RUNS[run_id], TIMES_DEFAULT, 8)
    return {"run_id": run_id, "input": {"source": source, "label": label, "points": len(u0), "max_abs": round(float(np.abs(u0).max()), 4), "mean": round(float(u0.mean()), 4)},
            "verdict": v, "difference_percent": None if res["rel"] is None else round(100 * res["rel"], 3),
            "ood": res["ood"], "self_check": res["self_check"],
            "timings": {"ai_ms": round(res["fno_s"] * 1000, 2), "solver_s": None if res["solver_s"] is None else round(res["solver_s"], 3)},
            "mass_drift": {"ai": res["mass_fno"], "solver": res["mass_solver"]}, "model": res["model"], "equation": res["equation"],
            "profiles": prof, "links": {"csv": f"/runs/{run_id}.csv"},
            "advice": "Use the solver's numbers when verdict.best_answer is 'solver'. Ask for other times with get_profile."}


def _profiles(run, times, stride):
    idx = [min(max(int(round(t / 0.01)), 0), 200) for t in times]
    xs = list(range(0, len(run["x"]), stride))
    out = {"t": [round(i * 0.01, 2) for i in idx], "x": _r(run["x"][xs], 5), "ai": [_r(run["fno"][i][xs]) for i in idx]}
    out["solver"] = None if run["solver"] is None else [_r(run["solver"][i][xs]) for i in idx]
    return out


def tool_get_profile(args, svc):
    rid = args.get("run_id")
    run = RUNS.get(rid) if isinstance(rid, str) else None
    if run is None:
        raise ValueError("unknown run_id (runs are kept in memory, the newest %d); call predict again" % RUN_LIMIT)
    times = args.get("times", TIMES_DEFAULT)
    if not isinstance(times, list) or not times or len(times) > 20:
        raise ValueError("'times' must be a list of 1 to 20 numbers in [0, 2]")
    times = [_num(t, "times[i]", 0, 2) for t in times]
    stride = int(_num(args.get("stride", 8), "stride", 1, 32))
    return {"run_id": rid, **_profiles(run, times, stride), "note": "values of u at the listed times; x is subsampled by 'stride'; nearest saved time step (0.01) is used"}


def tool_get_model_info(args, svc):
    return svc.model_info()


def tool_get_status(args, svc):
    return svc.health()


def tool_explain(args, svc):
    topic = args.get("topic")
    if topic not in TEXTS:
        raise ValueError(f"topic must be one of {list(TEXTS)}")
    return {"topic": topic, "text": TEXTS[topic]}


_WAVE_PROPS = {
    "preset": {"type": "string", "enum": list(waves.PRESET_INFO), "description": "A ready-made wave (see list_presets)."},
    "seed": {"type": "integer", "minimum": 1, "default": 1, "description": "For 'random' and 'noise' presets."},
    "components": {"type": "array", "minItems": 1, "maxItems": 8, "description": "Sine components: u = mean + sum amp*sin(2*pi*k*x + phase).",
                   "items": {"type": "object", "properties": {"amp": {"type": "number"}, "k": {"type": "integer", "minimum": 1, "maximum": 60}, "phase": {"type": "number", "default": 0}},
                             "required": ["amp", "k"], "additionalProperties": False}},
    "mean": {"type": "number", "default": 0, "description": "Added to the components."},
    "u0": {"type": "array", "minItems": 256, "maxItems": 512, "items": {"type": "number"},
           "description": "Your own wave: 256 values at x = (i + 0.5)/256 (512 also accepted), |u| <= 100."},
}


def _tool(name, title, description, props, fn, required=None):
    return {"name": name, "title": title, "description": description, "fn": fn,
            "inputSchema": {"type": "object", "properties": props, "additionalProperties": False, **({"required": required} if required else {})},
            "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}}


TOOLS = [
    _tool("list_presets", "List ready-made waves", "Lists the preset starting waves, including how well the AI did on each in our tests.", {}, tool_list_presets),
    _tool("make_wave", "Build a starting wave", "Builds the 256 values of a starting wave from a preset or from sine components, without running a forecast. Provide exactly one of preset, components or u0.",
          {k: _WAVE_PROPS[k] for k in ("preset", "seed", "components", "mean", "u0")}, tool_make_wave),
    _tool("predict", "Forecast a wave", "Runs the AI forecast (and, by default, the classic solver) for a starting wave and returns a verdict you must read before using any number. "
          "Provide exactly one of preset, components or u0. Takes about 0.5-1 s with the solver.",
          {**{k: _WAVE_PROPS[k] for k in ("preset", "seed", "components", "mean", "u0")}, "include_solver": {"type": "boolean", "default": True, "description": "Also run the reference solver (recommended)."}},
          tool_predict),
    _tool("get_profile", "Get wave profiles at chosen times", "Returns u(x) of the AI and the solver at the requested times for an earlier run_id.",
          {"run_id": {"type": "string"}, "times": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 2}, "minItems": 1, "maxItems": 20, "default": TIMES_DEFAULT},
           "stride": {"type": "integer", "minimum": 1, "maximum": 32, "default": 8}}, tool_get_profile, required=["run_id"]),
    _tool("get_model_info", "Model details", "Model card: parameters, measured errors, training data, known limitations, weights hash.", {}, tool_get_model_info),
    _tool("get_status", "Server status", "Health of the service and whether the model is loaded.", {}, tool_get_status),
    _tool("explain", "Explain a topic", "Plain-language text about the equation, the limits of the AI, how to read results, or what the verdict levels mean.",
          {"topic": {"type": "string", "enum": list(TEXTS)}}, tool_explain, required=["topic"]),
]
TOOL_BY_NAME = {t["name"]: t for t in TOOLS}


def public_tools():
    return [{k: v for k, v in t.items() if k != "fn"} for t in TOOLS]


# ------------------------------------------------------------------ resources
def resources(svc):
    return {
        "burgers://model-card": ("Model card", "application/json", lambda: json.dumps(svc.model_info(), ensure_ascii=False)),
        "burgers://limitations": ("Known limitations", "text/plain", lambda: TEXTS["limits"]),
        "burgers://guide": ("How to use this service", "text/plain", lambda: INSTRUCTIONS + "\n\n" + TEXTS["reading_results"] + "\n\n" + TEXTS["verdicts"]),
    }


# ------------------------------------------------------------------ JSON-RPC
def _err(mid, code, message, data=None):
    e = {"code": code, "message": message}
    if data is not None:
        e["data"] = data
    return {"jsonrpc": "2.0", "id": mid, "error": e}


def _ok(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _call_tool(params, svc):
    name = params.get("name")
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return None
    if not svc.ready():
        return {"content": [{"type": "text", "text": "The model is not loaded on the server."}], "isError": True}
    try:
        data = tool["fn"](params.get("arguments") or {}, svc)
    except ValueError as e:
        return {"content": [{"type": "text", "text": f"Invalid arguments: {e}"}], "isError": True}
    except limits.Busy:
        return {"content": [{"type": "text", "text": "The server is busy with other forecasts; try again in a few seconds."}], "isError": True}
    except Exception:                                    # never leak internals to a client
        return {"content": [{"type": "text", "text": "Internal error while running the tool."}], "isError": True}
    return {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False, separators=(",", ":"))}], "structuredContent": data, "isError": False}


def handle(msg, svc):
    """One JSON-RPC message in, one response dict out (None for notifications and for responses sent by the client)."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return _err(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid Request")
    if "method" not in msg:
        return None
    if "id" not in msg:
        return None                                      # notification, e.g. notifications/initialized
    mid, method, params = msg["id"], msg["method"], msg.get("params") or {}
    if not isinstance(params, dict):
        return _err(mid, -32602, "params must be an object")
    if method == "initialize":
        pv = params.get("protocolVersion")
        return _ok(mid, {"protocolVersion": pv if pv in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                         "capabilities": {"tools": {"listChanged": False}, "resources": {"subscribe": False, "listChanged": False}},
                         "serverInfo": SERVER_INFO, "instructions": INSTRUCTIONS})
    if method == "ping":
        return _ok(mid, {})
    if method == "tools/list":
        return _ok(mid, {"tools": public_tools()})
    if method == "tools/call":
        res = _call_tool(params, svc)
        return _err(mid, -32602, f"Unknown tool: {params.get('name')}") if res is None else _ok(mid, res)
    if method == "resources/list":
        return _ok(mid, {"resources": [{"uri": u, "name": n, "title": n, "mimeType": m} for u, (n, m, _) in resources(svc).items()]})
    if method == "resources/templates/list":
        return _ok(mid, {"resourceTemplates": []})
    if method == "resources/read":
        item = resources(svc).get(params.get("uri"))
        if item is None:
            return _err(mid, -32002, "Resource not found", {"uri": params.get("uri")})
        name, mime, fn = item
        return _ok(mid, {"contents": [{"uri": params["uri"], "mimeType": mime, "text": fn()}]})
    return _err(mid, -32601, f"Method not found: {method}")


def run_csv(run):
    """CSV of one stored run: t, x, u_fno, [u_solver, abs_error]."""
    t, x, f, s = run["t"], run["x"], run["fno"].astype(np.float64), run["solver"]
    head = "t,x,u_fno" + (",u_solver,abs_error" if s is not None else "")
    lines = [head]
    s = None if s is None else s.astype(np.float64)
    for k in range(len(t)):
        for i in range(len(x)):
            row = f"{t[k]:.2f},{x[i]:.6f},{f[k, i]:.6f}"
            if s is not None:
                row += f",{s[k, i]:.6f},{abs(f[k, i] - s[k, i]):.6f}"
            lines.append(row)
    return "\n".join(lines)


def manifest(base, svc, mcp_enabled=True):
    """Everything an agent needs to use this site, in one JSON document (served at /agent.json and shown by the page's Agent mode)."""
    return {
        "name": SERVER_INFO["title"], "version": SERVER_INFO["version"],
        "purpose": "Forecast how a wave evolves under 1D viscous Burgers (nu = 0.01) with a trained neural operator, next to a classic numerical solver, with trust warnings.",
        "instructions": INSTRUCTIONS,
        "interfaces": {
            "mcp": {"enabled": False, "note": "MCP is switched off on this server (set MCP_ENABLED=1 to enable it). The REST API below still works."} if not mcp_enabled else {"enabled": True, "url": base + "mcp", "transport": "streamable-http (JSON responses only, no sessions)", "protocolVersions": list(PROTOCOL_VERSIONS),
                    "client_config_example": {"mcpServers": {"burgers": {"url": base + "mcp"}}},
                    "tools": public_tools(), "resources": [{"uri": u, "name": n, "mimeType": m} for u, (n, m, _) in resources(svc).items()]},
            "rest": {"openapi": base + "openapi.json", "docs": base + "docs",
                     "endpoints": [{"method": "GET", "path": "/health"}, {"method": "GET", "path": "/model-info"},
                                   {"method": "POST", "path": "/predict", "body": {"u0": "256 (or 512) numbers", "include_solver": "boolean"}},
                                   {"method": "GET", "path": "/runs/{run_id}.csv", "note": "run_id comes from the MCP predict tool"},
                                   {"method": "GET", "path": "/agent.json"}]},
            "web_ui": {"url": base, "agent_mode": "Click the button with id 'agent-toggle' (or open " + base + "?mode=agent) for a machine-readable view of the page. Nothing machine-oriented is added to the page until then."},
        },
        "verdict_levels": {"good": "AI within 6% of the solver and no warning", "warn": "AI and solver disagree by more than 6%", "bad": "unfamiliar input or the AI failed its own sanity checks", "unchecked": "solver not run"},
        "rules_for_agents": ["Read verdict.level and verdict.best_answer before quoting any number.", "When best_answer is 'solver', use the solver's values.",
                             "Warnings are not guarantees; no warning does not prove accuracy.", "The AI is only validated for nu = 0.01, 256 points, t in [0,2], smooth sine-mixture waves."],
        "limits": TEXTS["limits"],
    }
