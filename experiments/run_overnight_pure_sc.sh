#!/usr/bin/env bash
# Pure-SC overnight matrix: NO mixed FP/SC, NO FP fallback.
# ASIC-realistic: the hardware design picks ONE comp architecture at tape-out;
# per-block it only chooses among pre-selected SC variants (seed + scale + head-layout).
#
# Each sc_config gets: (a) regular SC (D=1024 matmul) seed-bank + scales, and
# (b) head-aligned SC (per-head D=64) seed-bank + scales. Calibration picks the
# best variant per block from the pool. No FP option.

set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vit_sc

mkdir -p logs results

COMMON=(--n_calib 256 --n_eval 500 --batch_size 8 --workers 4 --skip_baseline
        --comp_mode sc)

run_one() {
    local name=$1; shift
    local logf=logs/${name}.log
    local outj=results/pure_sc_${name}.json
    if [ -f "$outj" ]; then
        echo "[skip] $outj exists" >&2
        return 0
    fi
    echo "[run] $name -> $outj" >&2
    python experiments/qwt_sc_overnight.py "$@" \
        --out_json "$outj" 2>&1 | tee "$logf"
    echo "[done] $name" >&2
}

# Per sc_config: one combined run with both regular-SC and head-aligned variants,
# K=4 seed bank, 3 W-scales, no polarity flip → 4*3 + 3 = 15 variants (per-block
# choice is 4 bits in an ASIC LUT).

for CFG in qk_only qk_av full_attn skip_worst50; do
    run_one ${CFG}_combined_pureSC --sc_config $CFG \
        "${COMMON[@]}" \
        --n_variants 4 --w_scales 0.5,0.75,1.0 \
        --head_aligned --n_heads 16
done

# Head-aligned ONLY (no regular SC, no seed bank) — cleanest single-architecture
# story: all-SC comp, all-blocks use per-head D=64.
for CFG in qk_only skip_worst50; do
    run_one ${CFG}_hd_only --sc_config $CFG \
        "${COMMON[@]}" \
        --n_variants 1 --w_scales 1.0 \
        --head_aligned_only --n_heads 16
done

# Regular SC ONLY (no seed bank, no scale) — reproduces README's "naive SC comp"
# baseline under the new driver. Sanity check.
for CFG in qk_only skip_worst50; do
    run_one ${CFG}_sc_naive_repro --sc_config $CFG \
        "${COMMON[@]}" \
        --n_variants 1 --w_scales 1.0
done

echo "[pure-sc matrix] all runs done"
