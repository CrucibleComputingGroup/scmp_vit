#!/bin/bash
# QwT cos sweep for the new "C" config: SL=256 (native sc_prec=8) + sched_len192.
# Raw baseline (cell C, n=100) bbox AP=69.978, the highest of the 4. Goal:
# does QwT comp lift it further, or is comp benefit smaller at lower noise?
set -euo pipefail
cd /home/yjrcs/SC_V/vit_sc/det
mkdir -p logs/cos_sweep_C results/cos_sweep_C

source /home/yjrcs/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2

N_EVAL=100
N_CALIB=16
SIZE=1024
SCHED=sensitivity/skip/skip_worst_30_len192.json

# Same cos density as the 192 sweep — refine later if peak is interior.
COS_LIST=(0.50 0.60 0.65 0.675 0.68 0.685 0.69 0.70 0.72)

run_cos () {
    local COS=$1
    local TAG="qwt_C_cos${COS//./}"
    local OUT_JSON="results/cos_sweep_C/${TAG}.json"
    local LOG="logs/cos_sweep_C/${TAG}.log"
    if [ -f "$OUT_JSON" ]; then
        echo "[$(date +%H:%M:%S)] skip ${TAG} (already exists)"
        return 0
    fi
    echo "==== $(date) ${TAG} cos=${COS} ===="
    python -u experiments/qwt_det_compensate.py \
        --sc_prec 8 \
        --sc_ops_per_block_json "$SCHED" \
        --size "$SIZE" \
        --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
        --calib_seed 1 --calib_seed_b 2 \
        --cos_threshold "$COS" --last_block_cos_threshold 0.8 \
        --lookahead_veto \
        --comp_mode sc --comp_sc_prec 8 --head_aligned --n_heads 16 --fwd_chunk 2 \
        --use_soft_nms --interp_type beit \
        --skip_baseline \
        --out_tag "$TAG" \
        --out_json "$OUT_JSON" \
        2>&1 | tee "$LOG"
    echo "==== ${TAG} done ===="
}

for c in "${COS_LIST[@]}"; do
    run_cos "$c"
done

echo "All cos_sweep_C cells done at $(date)"
