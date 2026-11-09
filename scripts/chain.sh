#!/usr/bin/env bash
# Unattended chain (run from repo root). Logs every step to logs/chain.log.
#   1. wait for install -> GPU check -> commit skeleton + lockfile (no push)
#   2. wait for download -> verify size + md5 -> scripts/inspect_h5.py -> reports/
# Any failure writes FAILED to logs/chain.log and stops the chain. No retries here.
PY=.venv/Scripts/python.exe
RAW=data/raw/1D_Burgers_Sols_Nu0.01.hdf5
EXPECTED_SIZE=8232968312
EXPECTED_MD5=e6d9a4f62baf9a29121a816b919e2770
L=logs/chain.log
say() { echo "$(date '+%F %T') $*" | tee -a "$L"; }
fail() { say "FAILED: $*  -- chain stopped"; exit 1; }
mkdir -p reports

say "step 1: waiting for install"
until grep -q "lockfile written" logs/install.log 2>/dev/null || grep -qE "^ERROR" logs/install.log 2>/dev/null; do sleep 30; done
grep -q "lockfile written" logs/install.log || fail "install error, see logs/install.log"
say "install finished; lockfile has $(wc -l < requirements.lock.txt) lines"

say "step 1a: GPU check"
$PY -c "
import torch
print('torch', torch.__version__, '| built for CUDA', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA not available'
print('device:', torch.cuda.get_device_name(0), '| VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 2))
a = torch.randn(2048, 2048, device='cuda'); b = a @ a; torch.cuda.synchronize()
print('GPU matmul ok; result device:', b.device, '| sum:', float(b.sum()))
x = torch.randn(8, 64, 1024, device='cuda'); y = torch.fft.rfft(x, dim=-1)
print('rfft on GPU ok; out shape', tuple(y.shape), y.dtype)
" > logs/gpu_check.log 2>&1 || { cat logs/gpu_check.log >> "$L"; fail "GPU check, see logs/gpu_check.log"; }
cat logs/gpu_check.log | tee -a "$L"

say "step 1b: commit skeleton + lockfile (no push)"
git add -A
if git status --short | grep -E "\.venv|data/|\.hdf5|logs/" ; then fail "unexpected ignored-type file staged"; fi
git commit -q -m "Add repo skeleton, scripts and requirements.lock.txt

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" || fail "git commit"
say "committed: $(git log --oneline -1)"

say "step 2: waiting for download"
until [ -f "$RAW" ] || grep -qE "FAILED|Traceback|larger|differs" logs/download.log 2>/dev/null; do sleep 30; done
[ -f "$RAW" ] || fail "download did not finish, see logs/download.log"
SIZE=$(stat -c %s "$RAW")
[ "$SIZE" = "$EXPECTED_SIZE" ] || fail "size $SIZE != $EXPECTED_SIZE"
say "size OK: $SIZE bytes"
MD5=$(md5sum "$RAW" | cut -d' ' -f1)
[ "$MD5" = "$EXPECTED_MD5" ] || fail "md5 $MD5 != $EXPECTED_MD5"
say "md5 OK: $MD5"

say "step 2a: inspect_h5"
$PY scripts/inspect_h5.py "$RAW" > reports/inspect_report.txt 2>&1 || { tail -20 reports/inspect_report.txt >> "$L"; fail "inspect_h5.py, see reports/inspect_report.txt"; }
say "inspect OK: reports/inspect_report.txt, reports/samples.png, reports/inspect_summary.json"
say "CHAIN COMPLETE"
