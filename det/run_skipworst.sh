#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vit_sc
cd "$(dirname "$0")"
SD=results/sensitivity/schedules

CUDA_VISIBLE_DEVICES=0 python -u sc_eval.py --n-eval 10 --sc_ops_per_block_json $SD/skip_worst_20.json --out-tag skip_worst_20 > results/log_skip_worst_20.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u sc_eval.py --n-eval 10 --sc_ops_per_block_json $SD/skip_worst_30.json --out-tag skip_worst_30 > results/log_skip_worst_30.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u sc_eval.py --n-eval 10 --sc_ops_per_block_json $SD/skip_worst_40.json --out-tag skip_worst_40 > results/log_skip_worst_40.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u sc_eval.py --n-eval 10 --sc_ops_per_block_json $SD/skip_worst_50.json --out-tag skip_worst_50 > results/log_skip_worst_50.txt 2>&1 &

echo 'Launched 4 skip_worst evals'
wait
echo 'ALL DONE'
for k in 20 30 40 50; do
    echo "=== skip_worst_$k ==="
    tail -n 3 results/log_skip_worst_$k.txt
done
