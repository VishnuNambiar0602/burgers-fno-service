#!/usr/bin/env bash
# Smoke tests + timing runs only. These are NOT training runs and produce no usable model.
PY=.venv/Scripts/python.exe
set -u
$PY scripts/train.py --model fno  --name smoke_fno  --n-train 256 --epochs 15 --max-minutes 2.5 --label "SMOKE TEST (tiny subset, not a trained model)"
$PY scripts/train.py --model unet --name smoke_unet --n-train 256 --epochs 15 --max-minutes 2.5 --label "SMOKE TEST (tiny subset, not a trained model)"
echo "=== resume test: same config, 2 more epochs allowed"
$PY scripts/train.py --model fno  --name smoke_fno  --n-train 256 --epochs 17 --max-minutes 2.5 --label "SMOKE TEST resume"
echo "=== timing at full train size (4000 samples), 2 epochs each"
$PY scripts/train.py --model fno  --name timing_fno  --n-train 4000 --epochs 2 --label "TIMING RUN only"
$PY scripts/train.py --model unet --name timing_unet --n-train 4000 --epochs 2 --label "TIMING RUN only"
echo "=== smoke.sh done"
