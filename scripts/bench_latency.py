"""Latency and memory benchmark. Parent mode runs one fresh child process per configuration so that
import time, load time and peak memory are real cold-process numbers.

  python scripts/bench_latency.py            -> runs all configs, writes reports/bench_latency.json
Child mode (internal): python scripts/bench_latency.py child <cpu|cuda> <threads> <fno_s1|unet_s1>
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def mem_mb():
    """(current, peak) working set of this process in MB, via the Windows psapi call."""
    k32, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wt.HANDLE
    ps.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    ps.GetProcessMemoryInfo.restype = wt.BOOL
    p = PMC(); p.cb = ctypes.sizeof(p)
    if not ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return p.WorkingSetSize / 1e6, p.PeakWorkingSetSize / 1e6


def child(dev, threads, name):
    t0 = time.perf_counter()
    import numpy as np
    import torch
    t_import = time.perf_counter() - t0
    sys.path.insert(0, ROOT)
    from src.models import FNO1d, UNet1d
    if threads > 0:
        torch.set_num_threads(threads)
    t0 = time.perf_counter()
    ck = torch.load(os.path.join(ROOT, "checkpoints", name, "best.pt"), map_location="cpu", weights_only=False)
    c = ck["args"]
    m = FNO1d(201, c["modes"], c["width"], c["layers"]) if name.startswith("fno") else UNet1d(201, c["base"])
    m.load_state_dict(ck["model"]); m.eval().to(dev)
    t_load = time.perf_counter() - t0
    ws_load, _ = mem_mb()
    val = np.load(os.path.join(ROOT, "data", "processed", "val.npy"), mmap_mode="r")[:64, 0, :].astype("float32").copy()
    out = {"device": dev, "threads": torch.get_num_threads(), "model": name, "import_torch_s": t_import,
           "load_model_s": t_load, "rss_after_load_MB": ws_load}
    with torch.inference_mode():
        for bs in (1, 8, 32):
            x = torch.from_numpy(val[:bs]).to(dev)
            for _ in range(10):
                m(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            ts = []
            for _ in range(200 if bs < 32 else 100):
                t1 = time.perf_counter()
                y = m(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t1) * 1000)
            ts.sort()
            out[f"batch{bs}_ms"] = {"p50": statistics.median(ts), "p95": ts[int(0.95 * len(ts)) - 1], "max": ts[-1]}
    ws, peak = mem_mb()
    out["rss_end_MB"], out["peak_rss_MB"] = ws, peak
    print(json.dumps(out))


def solver_bench():
    sys.path.insert(0, ROOT)
    import numpy as np
    from src.solver import solve
    val = np.load(os.path.join(ROOT, "data", "processed", "val.npy"), mmap_mode="r")[:8, 0, :].astype("float64").copy()
    t_out = np.arange(201) * 0.01
    ts = []
    for _ in range(3):
        t1 = time.perf_counter(); solve(val[:1], 0.01 / np.pi, t_out); ts.append(time.perf_counter() - t1)
    t1 = time.perf_counter(); solve(val, 0.01 / np.pi, t_out); tb = time.perf_counter() - t1
    return {"numpy_solver_256cells_1sample_s": min(ts), "numpy_solver_256cells_batch8_s": tb}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4])
        sys.exit()
    rows = []
    for dev, threads, name in [("cpu", 1, "fno_s1"), ("cpu", 2, "fno_s1"), ("cpu", 4, "fno_s1"), ("cpu", 0, "fno_s1"),
                               ("cpu", 2, "unet_s1"), ("cuda", 0, "fno_s1")]:
        t1 = time.perf_counter()
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "child", dev, str(threads), name],
                           capture_output=True, text=True)
        wall = time.perf_counter() - t1
        if r.returncode != 0:
            print("FAILED", dev, threads, name, r.stderr[-400:]); continue
        d = json.loads(r.stdout.strip().splitlines()[-1]); d["process_wall_s"] = wall; rows.append(d)
        print(f"{name} {dev} threads={d['threads']:2d} | import torch {d['import_torch_s']:.2f}s load {d['load_model_s']:.3f}s "
              f"| b1 p50 {d['batch1_ms']['p50']:.1f} p95 {d['batch1_ms']['p95']:.1f} ms | b8 p50 {d['batch8_ms']['p50']:.1f} ms "
              f"| b32 p50 {d['batch32_ms']['p50']:.1f} ms | RSS after load {d['rss_after_load_MB']:.0f} MB, peak {d['peak_rss_MB']:.0f} MB "
              f"| process wall {wall:.1f}s", flush=True)
    s = solver_bench()
    print("numpy solver:", s)
    json.dump({"runs": rows, "solver": s, "cpu": "i7-13650HX (20 threads); free HF CPU tier is ~2 vCPU, so expect slower there"},
              open(os.path.join(ROOT, "reports", "bench_latency.json"), "w"), indent=1)
