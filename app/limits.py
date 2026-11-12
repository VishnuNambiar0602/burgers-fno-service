"""Protection for the public service: a per-client rate limit and a server-wide cap on simultaneous forecasts. Standard library only.

Settings (environment variables, read on every request so tests and operators can change them without a restart):
  RATE_LIMIT_PER_MIN   requests per client per rolling 60 s on POST /predict and POST /mcp   (default 20; 0 switches the limit off)
  MAX_CONCURRENT       forecasts computed at the same time, server-wide                      (default 2; 0 = unlimited)
  TRUST_PROXY          1 = identify clients by the first X-Forwarded-For address (needed behind Render's proxy), 0 = socket address   (default 1)
This is NOT authentication. The counters live in this process's memory (they reset on restart and are not shared between instances), and an
X-Forwarded-For header can be faked by a client that reaches the server directly, so the concurrency cap is the backstop that cannot be bypassed.
"""
import math
import os
import threading
import time
from collections import deque

WINDOW_S = 60.0
_hits = {}
_lock = threading.Lock()
_active = 0
_active_lock = threading.Lock()


class Busy(RuntimeError):
    """Raised when too many forecasts are already running."""


def limit_per_min():
    return int(os.environ.get("RATE_LIMIT_PER_MIN", "20"))


def max_concurrent():
    return int(os.environ.get("MAX_CONCURRENT", "2"))


def client_key(request):
    if os.environ.get("TRUST_PROXY", "1") == "1":
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check(key, now=None):
    """Record one request for `key`. Returns (allowed, retry_after_seconds)."""
    n = limit_per_min()
    if n <= 0:
        return True, 0
    now = time.monotonic() if now is None else now
    with _lock:
        q = _hits.setdefault(key, deque())
        while q and now - q[0] >= WINDOW_S:
            q.popleft()
        if len(q) >= n:
            return False, max(1, math.ceil(WINDOW_S - (now - q[0])))
        q.append(now)
        if len(_hits) > 5000:                                     # keep memory bounded: drop clients with no recent hits
            for k in [k for k, v in _hits.items() if not v or now - v[-1] >= WINDOW_S]:
                _hits.pop(k, None)
    return True, 0


def reset():
    """Forget all counters (used by tests)."""
    global _active
    with _lock:
        _hits.clear()
    with _active_lock:
        _active = 0


class Slot:
    """Context manager holding one concurrency slot; raises Busy immediately (no waiting) when all slots are taken."""
    def __enter__(self):
        global _active
        cap = max_concurrent()
        with _active_lock:
            if cap > 0 and _active >= cap:
                raise Busy("too many forecasts are running")
            _active += 1
        return self

    def __exit__(self, *exc):
        global _active
        with _active_lock:
            _active -= 1
        return False
