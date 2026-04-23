#!/usr/bin/env bash
# One-off: avg192 uniform (SL=192 everywhere) at K=30.
# SL=192 is valid via SC early termination; implemented through the MP path
# with a single level because --sc_prec is int-pow2-only.
# Runs qwtoff on GPU 0 and qwton on GPU 1 in parallel; leaves GPU 2 (busy
# with avg192_mp_qwton) and GPU 3 untouched.
set -euo pipefail

cd "$(dirname "$0")/.."
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vit_sc
export XFORMERS_DISABLED=1

RESULTS=results/sweep_int678_k30
LOGS=logs/sweep_int678_k30
SCHED=sensitivity/skip_worst30/skip_worst30_p8.json

MP_ARGS=(
  --mp_levels 192 --mp_fractions 1.0
  --mp_ops mlp_fc1,mlp_fc2,qkv_proj,out_proj
  --qk_mp_levels 192 --qk_mp_fractions 1.0
  --av_mp_levels 192 --av_mp_fractions 1.0
)

CUDA_VISIBLE_DEVICES=0 python eval.py --mode sc \
  --sc_ops_per_block_json "$SCHED" \
  --sc_prec 8 \
  --batch_size 16 --workers 8 \
  "${MP_ARGS[@]}" \
  --max_images 1000 \
  --out_json "$RESULTS/avg192_uniform_qwtoff.json" \
  > "$LOGS/avg192_uniform_qwtoff.log" 2>&1 &
PID_OFF=$!
echo "[launch gpu=0 pid=$PID_OFF] avg192_uniform_qwtoff"

CUDA_VISIBLE_DEVICES=1 python experiments/qwt_sc_overnight.py \
  --sc_ops_per_block_json "$SCHED" \
  --sc_prec 8 \
  --batch_size 16 --workers 8 \
  "${MP_ARGS[@]}" \
  --n_calib 1024 --n_eval 1000 \
  --comp_mode sc --comp_sc_prec 8 \
  --r2_threshold 0.5 --skip_baseline \
  --out_json "$RESULTS/avg192_uniform_qwton.json" \
  > "$LOGS/avg192_uniform_qwton.log" 2>&1 &
PID_ON=$!
echo "[launch gpu=1 pid=$PID_ON] avg192_uniform_qwton"

wait
echo "[done] both avg192_uniform runs complete"
