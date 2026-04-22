#!/bin/bash
# INT6: QK / QK+AV / QK+AV+Proj on 3 GPUs parallel
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=0 python -u sc_eval.py --n-eval 100 --sc_prec 6 \
    --sc_ops qk --out-tag sc_qk_int6 > results/log_qk_int6.txt 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python -u sc_eval.py --n-eval 100 --sc_prec 6 \
    --sc_ops qk,av --out-tag sc_qk_av_int6 > results/log_qk_av_int6.txt 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 python -u sc_eval.py --n-eval 100 --sc_prec 6 \
    --sc_ops qk,av,proj --out-tag sc_qk_av_proj_int6 > results/log_qk_av_proj_int6.txt 2>&1 &
P2=$!
echo "pids: QK=$P0  QK+AV=$P1  QK+AV+Proj=$P2"
wait $P0 $P1 $P2
echo "DONE"
for tag in qk_int6 qk_av_int6 qk_av_proj_int6; do
    echo "--- $tag ---"
    tail -n 5 results/log_$tag.txt
done
