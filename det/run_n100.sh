#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
cd /home/yjrcs/SC_V/vit_sc/det
SD=results/sensitivity/schedules

CUDA_VISIBLE_DEVICES=0 python -u fp_eval.py --n-eval 100 --out-tag fp100 > results/log_fp100.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u sc_eval.py --n-eval 100 --sc_ops 'qk,av,proj' --out-tag sc_qk_av_proj_100 > results/log_qk_av_proj_100.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u sc_eval.py --n-eval 100 --sc_ops_per_block_json $SD/skip_worst_10.json --out-tag skip_worst_10_100 > results/log_skip_worst_10_100.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u sc_eval.py --n-eval 100 --sc_ops_per_block_json $SD/skip_worst_20.json --out-tag skip_worst_20_100 > results/log_skip_worst_20_100.txt 2>&1 &

echo 'Launched 4 evals (n=100)'
wait
echo 'ALL DONE'
for tag in fp100 sc_qk_av_proj_100 skip_worst_10_100 skip_worst_20_100; do
    echo "=== $tag ==="
    tail -n 3 results/log_$tag.txt
done
