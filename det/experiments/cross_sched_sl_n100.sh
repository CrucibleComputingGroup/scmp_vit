#!/bin/bash
# Cross-validation of "schedule × stoc_len matching" hypothesis.
# 4 cells at n_eval=100, sz=1024, soft-NMS + beit interp:
#   A_baseline_int8     SL=256  sched=int8     (matches sc_sw30_int8_*    @ n=5000)
#   B_baseline_len192   SL=192  sched=len192   (matches sc_sw30_len192_*  @ n=5000)
#   C_cross_sched192_sl256  SL=256  sched=len192  (predicted > A)
#   D_cross_sched8_sl192    SL=192  sched=int8    (predicted < B)
set -euo pipefail
cd /home/yjrcs/SC_V/vit_sc/det
mkdir -p logs results

source /home/yjrcs/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2

N_EVAL=100
SIZE=1024
SCHED_INT8=sensitivity/skip/skip_worst_30_int8.json
SCHED_LEN192=sensitivity/skip/skip_worst_30_len192.json

run_one () {
    local TAG=$1 SC_PREC=$2 SCHED=$3
    shift 3
    local OUT_TAG="cross_${TAG}_sz${SIZE}_n${N_EVAL}"
    local LOG="logs/${OUT_TAG}.log"
    echo "==== $(date) ${OUT_TAG} ===="
    echo "  sc_prec=${SC_PREC}  sched=${SCHED}  extra=$*"
    python -u sc_eval.py \
        --n-eval "$N_EVAL" \
        --sc_prec "$SC_PREC" \
        --size "$SIZE" \
        --sc_ops_per_block_json "$SCHED" \
        --use_soft_nms \
        --interp_type beit \
        --out-tag "$OUT_TAG" \
        "$@" 2>&1 | tee "$LOG"
    echo "==== ${OUT_TAG} done ===="
}

run_one A_baseline_int8       8 "$SCHED_INT8"
run_one B_baseline_len192     8 "$SCHED_LEN192" --stoc_len 192
run_one C_sched192_sl256      8 "$SCHED_LEN192"
run_one D_sched8_sl192        8 "$SCHED_INT8"   --stoc_len 192

echo "All 4 cells done at $(date)"
