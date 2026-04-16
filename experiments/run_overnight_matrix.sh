#!/usr/bin/env bash
# Overnight experiment matrix for SC-comp accuracy recovery.
#
# Sequential because we have one 4080. Each run logs to logs/<name>.log
# and writes a JSON to results/overnight_<name>.json.
#
# After all runs finish, experiments/summarize_overnight.py aggregates.

set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vit_sc

mkdir -p logs results

# Common args
COMMON=(--n_calib 256 --n_eval 500 --batch_size 8 --workers 4 --skip_baseline
        --comp_mode sc)

run_one() {
    local name=$1; shift
    local logf=logs/${name}.log
    local outj=results/overnight_${name}.json
    if [ -f "$outj" ]; then
        echo "[skip] $outj exists" >&2
        return 0
    fi
    echo "[run] $name -> $outj" >&2
    python experiments/qwt_sc_overnight.py "$@" \
        --out_json "$outj" 2>&1 | tee "$logf"
    echo "[done] $name" >&2
}

# 2: skip_worst50 with K=6 + scales
run_one skip_worst50_K6_scales --sc_config skip_worst50 \
    "${COMMON[@]}" --n_variants 6 --w_scales 0.25,0.5,0.75,1.0

# 3: qk_only with head-aligned variants ONLY (head-aligned vs original SC)
run_one qk_only_head_aligned_only --sc_config qk_only \
    "${COMMON[@]}" --n_variants 1 --w_scales 0.5,0.75,1.0 \
    --head_aligned_only --n_heads 16

# 4: qk_only with head-aligned + regular SC (lets calibration pick best per block)
run_one qk_only_full_kitchen_sink --sc_config qk_only \
    "${COMMON[@]}" --n_variants 4 --w_scales 0.5,0.75,1.0 \
    --head_aligned --n_heads 16

# 5: qk_av with K=6 + scales (intermediate config)
run_one qk_av_K6_scales --sc_config qk_av \
    "${COMMON[@]}" --n_variants 6 --w_scales 0.5,0.75,1.0

# 6: full_attn with K=6 + scales (worst case)
run_one full_attn_K6_scales --sc_config full_attn \
    "${COMMON[@]}" --n_variants 6 --w_scales 0.5,0.75,1.0

# 7: W-scale only (no seed search) — isolates scale effect on qk_only
run_one qk_only_wscale_only --sc_config qk_only \
    "${COMMON[@]}" --n_variants 1 --w_scales 0.0,0.25,0.5,0.75,1.0

# 8: Pure-FP comp baseline for qk_only
run_one qk_only_fp_only --sc_config qk_only \
    "${COMMON[@]}" --comp_mode fp --w_scales 1.0

echo "[matrix] all runs done"
