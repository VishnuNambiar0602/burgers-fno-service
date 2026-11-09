#!/usr/bin/env bash
# Reproduce the local training environment (Windows Git Bash, run from repo root).
# --no-cache-dir keeps pip from writing a second copy of the wheels outside the project.
# The 2.6 GB torch wheel is fetched with a resumable curl first, because pip restarts
# from zero when a read times out (that is how the first attempt failed).
set -euo pipefail
PY=.venv/Scripts/python.exe
TORCH_URL="https://download-r2.pytorch.org/whl/cu126/torch-2.14.1%2Bcu126-cp313-cp313-win_amd64.whl"
WHEEL_DIR=data/wheels            # git-ignored
WHEEL="$WHEEL_DIR/torch-2.14.1+cu126-cp313-cp313-win_amd64.whl"
mkdir -p "$WHEEL_DIR"

LEN=$(curl -sIL "$TORCH_URL" | tr -d '\r' | grep -i '^content-length' | tail -1 | tr -dc 0-9)
echo "torch wheel expected size: $LEN bytes"
for i in $(seq 1 80); do
  SIZE=$( [ -f "$WHEEL" ] && stat -c %s "$WHEEL" || echo 0 )
  [ "$SIZE" = "$LEN" ] && break
  echo "torch download attempt $i: have $SIZE / $LEN"
  curl -L -C - --connect-timeout 30 --speed-time 60 --speed-limit 1000 -sS -o "$WHEEL" "$TORCH_URL" || true
done
[ "$(stat -c %s "$WHEEL")" = "$LEN" ] || { echo "ERROR: torch wheel incomplete"; exit 1; }
echo "torch wheel complete: $LEN bytes"

# install the local wheel; its dependencies come from the same CUDA index (mirrors of PyPI packages)
$PY -m pip install --no-cache-dir --default-timeout 120 --retries 10 "$WHEEL" --index-url https://download.pytorch.org/whl/cu126

# everything else from PyPI (versions are pinned afterwards in requirements.lock.txt)
$PY -m pip install --no-cache-dir --default-timeout 120 --retries 10 numpy h5py scipy matplotlib fastapi uvicorn pydantic pytest httpx

$PY -m pip freeze > requirements.lock.txt
echo "--- lockfile written:"; wc -l requirements.lock.txt
