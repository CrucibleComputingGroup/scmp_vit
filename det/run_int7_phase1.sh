#!/bin/bash
# INT7 Phase 1: QK / QK+AV+Proj / sensitivity in parallel on 3 GPUs
# GPU0: QK only, n_eval=100
# GPU1: QK+AV+Proj, n_eval=100
# GPU2: sensitivity sweep, all 40 blocks, n_eval=10
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
cd "$(dirname "$0")"

N_EVAL_FULL=100
N_EVAL_SENS=10
SC_PREC=7

mkdir -p results/sensitivity_int7

echo "========================================"
echo "INT7 Phase 1 (3 GPUs parallel)"
echo "  GPU0: QK only         (n=$N_EVAL_FULL)"
echo "  GPU1: QK+AV+Proj      (n=$N_EVAL_FULL)"
echo "  GPU2: sensitivity all (n=$N_EVAL_SENS)"
echo "========================================"

CUDA_VISIBLE_DEVICES=0 python -u sc_eval.py \
    --n-eval $N_EVAL_FULL --sc_prec $SC_PREC \
    --sc_ops qk --out-tag sc_qk_int7 \
    > results/log_qk_int7.txt 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -u sc_eval.py \
    --n-eval $N_EVAL_FULL --sc_prec $SC_PREC \
    --sc_ops qk,av,proj --out-tag sc_qk_av_proj_int7 \
    > results/log_qk_av_proj_int7.txt 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python -u experiments/sensitivity_sweep.py \
    --n_eval $N_EVAL_SENS --sc_prec $SC_PREC \
    --block_start 0 --block_end 40 \
    --out_json results/sensitivity_int7/sensitivity_all_ops.json --resume \
    > results/sensitivity_int7/log_shard_all.txt 2>&1 &
PID2=$!

echo "Launched pids: QK=$PID0  QK+AV+Proj=$PID1  sensitivity=$PID2"
wait $PID0 $PID1 $PID2
echo "All 3 Phase-1 jobs done."

echo "========================================"
echo "Phase 1 summary"
echo "========================================"
for tag in qk_int7 qk_av_proj_int7; do
    echo "--- $tag ---"
    tail -n 3 results/log_$tag.txt
done
echo "--- sensitivity ---"
tail -n 5 results/sensitivity_int7/log_shard_all.txt
echo "DONE_PHASE1"
