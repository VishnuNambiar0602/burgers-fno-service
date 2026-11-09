"""Resumable, verified download of one PDEBench Burgers file from DaRUS.

Usage: python scripts/download_data.py            (run from the repo root)

Writes data/raw/<name>.part, resumes with an HTTP Range request if the server
allows it, logs MB/s every 10 s, and only renames to the final name once the
size AND md5 (taken from the DaRUS API) match. Nothing is run on the data.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error

DATAFILE_ID = 281363
NAME = "1D_Burgers_Sols_Nu0.01.hdf5"
EXPECTED_SIZE = 8_232_968_312  # measured from the DaRUS files API
API = "https://darus.uni-stuttgart.de/api"
OUT_DIR = os.path.join("data", "raw")
FINAL = os.path.join(OUT_DIR, NAME)
PART = FINAL + ".part"
MAX_ATTEMPTS = 30


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def expected_md5():
    url = f"{API}/files/{DATAFILE_ID}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "dl"}), timeout=60) as r:
        d = json.load(r)["data"]
    log(f"API says: {d['dataFile']['filename']} size={d['dataFile']['filesize']} md5={d['dataFile'].get('md5')}")
    if d["dataFile"]["filesize"] != EXPECTED_SIZE:
        raise SystemExit("API size differs from the size measured earlier; stopping.")
    return d["dataFile"].get("md5")


def fetch_once():
    have = os.path.getsize(PART) if os.path.exists(PART) else 0
    headers = {"User-Agent": "dl"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(f"{API}/access/datafile/{DATAFILE_ID}", headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        status = r.status
        log(f"HTTP {status}, Content-Length={r.headers.get('Content-Length')}, "
            f"Accept-Ranges={r.headers.get('Accept-Ranges')}, resuming_from={have}")
        if have and status != 206:
            log("Server ignored the Range request (200 instead of 206): restarting from byte 0.")
            have = 0
        mode = "ab" if have else "wb"
        got, t0, tl, gl = have, time.time(), time.time(), have
        with open(PART, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                now = time.time()
                if now - tl >= 10:
                    log(f"{got/1e9:.3f} GB / {EXPECTED_SIZE/1e9:.3f} GB "
                        f"({100*got/EXPECTED_SIZE:.1f}%) {((got-gl)/1e6)/(now-tl):.1f} MB/s "
                        f"(avg {((got-have)/1e6)/(now-t0):.1f} MB/s)")
                    tl, gl = now, got


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(FINAL):
        log(f"{FINAL} already exists ({os.path.getsize(FINAL)} bytes); delete it to re-download.")
        return
    md5 = expected_md5()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        size = os.path.getsize(PART) if os.path.exists(PART) else 0
        if size == EXPECTED_SIZE:
            break
        if size > EXPECTED_SIZE:
            raise SystemExit(f"{PART} is larger ({size}) than expected: corrupt. Not continuing.")
        log(f"attempt {attempt}: have {size} bytes")
        try:
            fetch_once()
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            log(f"download interrupted: {e!r}; retrying in {min(60, 5*attempt)} s")
            time.sleep(min(60, 5 * attempt))
    size = os.path.getsize(PART) if os.path.exists(PART) else 0
    if size != EXPECTED_SIZE:
        log(f"FAILED: have {size} bytes, expected {EXPECTED_SIZE}. Partial file left at {PART}.")
        sys.exit(1)
    log("size matches; computing md5 ...")
    got = md5_of(PART)
    if md5 and got != md5:
        log(f"FAILED: md5 {got} != expected {md5}. File NOT renamed; do not use it.")
        sys.exit(2)
    os.replace(PART, FINAL)
    log(f"OK: {FINAL} {size} bytes, md5 {got} (matches API: {bool(md5)})")


if __name__ == "__main__":
    main()
