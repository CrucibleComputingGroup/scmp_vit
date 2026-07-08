#!/usr/bin/env bash
# Validate τ=0.65 on the OTHER 3 collapsing configs: p7_mp, avg192_uniform, avg192_mp.
# Run sequentially on a single GPU (env CUDA_VISIBLE_DEVICES). Each cell ~50 min.
# Total wall: ~2.5 h.
#
# Caller is expected to have:
#   * conda env qwt_d2 active
#   * gcc/13.2.0 module loaded
#   * CUDA_VISIBLE_DEVICES set to a free GPU (e.g. 1, 2, or 3)
#
# Run via: bash experiments/_run_tau065_multiconfig.sh <gpu>
set -e
cd /home/allenjin/Projects/vit_sc/det
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate qwt_d2
module load gcc/13.2.0
GPU="${1:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p logs/gate_tuning results/gate_tuning

run_cell() {
  local desc="$1" out="$2" log="$3"; shift 3
  echo "=========================================="
  echo "[$(date +%H:%M:%S)] STARTING(GPU=$GPU): $desc"
  echo "  -> $out"
  echo "=========================================="
  python "$@" > "$log" 2>&1 || { echo "[FAIL] $desc — see $log"; tail -40 "$log"; exit 1; }
  echo "[$(date +%H:%M:%S)] DONE(GPU=$GPU): $desc"
}

# --- p7_mp at τ=0.65 ---
run_cell "τ=0.65 p7_mp" \
  "results/gate_tuning/tau_0.65_p7_mp.json" \
  "logs/gate_tuning/tau_0.65_p7_mp.log" \
  experiments/qwt_det_compensate.py \
    --sc_prec 7 \
    --sl_map_json results/smoke_sweep_2026-04-25/sl_maps/p7_mp_sl_map.json \
    --n_calib 16 --n_eval 10 \
    --calib_seed 1 --calib_seed_b 2 \
    --cos_threshold 0.65 --last_block_cos_threshold 0.9 \
    --lookahead_veto \
    --comp_mode sc --comp_sc_prec 8 --head_aligned --fwd_chunk 2 \
    --out_tag tau_0.65_p7_mp \
    --out_json results/gate_tuning/tau_0.65_p7_mp.json

# --- avg192_uniform at τ=0.65 ---
run_cell "τ=0.65 avg192_uniform" \
  "results/gate_tuning/tau_0.65_avg192_uniform.json" \
  "logs/gate_tuning/tau_0.65_avg192_uniform.log" \
  experiments/qwt_det_compensate.py \
    --sc_prec 8 --sc_ops_per_block_json sensitivity/skip/skip_worst_30_int7.json \
    --mp_levels 192 --mp_fractions 1.0 --mp_ops mlp_fc1,mlp_fc2,qkv_proj,out_proj \
    --qk_mp_levels 192 --qk_mp_fractions 1.0 \
    --av_mp_levels 192 --av_mp_fractions 1.0 \
    --n_calib 16 --n_eval 10 \
    --calib_seed 1 --calib_seed_b 2 \
    --cos_threshold 0.65 --last_block_cos_threshold 0.9 \
    --lookahead_veto \
    --comp_mode sc --comp_sc_prec 8 --head_aligned --fwd_chunk 2 \
    --out_tag tau_0.65_avg192_uniform \
    --out_json results/gate_tuning/tau_0.65_avg192_uniform.json

# --- avg192_mp at τ=0.65 ---
run_cell "τ=0.65 avg192_mp" \
  "results/gate_tuning/tau_0.65_avg192_mp.json" \
  "logs/gate_tuning/tau_0.65_avg192_mp.log" \
  experiments/qwt_det_compensate.py \
    --sc_prec 8 \
    --sl_map_json results/smoke_sweep_2026-04-25/sl_maps/avg192_mp_sl_map.json \
    --n_calib 16 --n_eval 10 \
    --calib_seed 1 --calib_seed_b 2 \
    --cos_threshold 0.65 --last_block_cos_threshold 0.9 \
    --lookahead_veto \
    --comp_mode sc --comp_sc_prec 8 --head_aligned --fwd_chunk 2 \
    --out_tag tau_0.65_avg192_mp \
    --out_json results/gate_tuning/tau_0.65_avg192_mp.json

echo "=========================================="
echo "[$(date +%H:%M:%S)] ALL τ=0.65 multiconfig DONE (GPU=$GPU)"
echo "=========================================="
