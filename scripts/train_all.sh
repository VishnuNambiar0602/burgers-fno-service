#!/usr/bin/env bash
# Gate (b) approved by the user: FNO and U-Net, 3 seeds each, 300 epochs, 4000 train samples. Val only.
# Resumable: rerun this script and each run continues from checkpoints/<name>/last.pt (finished runs are skipped quickly).
PY=.venv/Scripts/python.exe
for model in fno unet; do
  for seed in 0 1 2; do
    $PY scripts/train.py --model $model --name ${model}_s${seed} --seed $seed --epochs 300 --n-train 4000 --batch 32 \
        --lr 1e-3 --label "MAIN RUN ${model} seed ${seed}" || { echo "$(date '+%F %T') FAILED ${model}_s${seed}, chain stopped" | tee -a logs/train_all.log; exit 1; }
    echo "$(date '+%F %T') done ${model}_s${seed}" >> logs/train_all.log
  done
done
echo "$(date '+%F %T') ALL RUNS DONE" >> logs/train_all.log
