#!/bin/bash
# Wait until sweep's FullAttn_ipr config finishes (JSON exists), then run.
set -eu
cd "$(dirname "$0")/.."
OUT=results/qwt_av_entropy
mkdir -p "$OUT"

# Block until FullAttn_ipr.json appears AND no python holding GPU for 30s+
echo "[queue] waiting for main sweep to finish..."
until [ -f "$OUT/FullAttn_ipr.json" ]; do sleep 10; done
# Extra: ensure no lingering python processes holding GPU
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | awk '{print $1}')
  if [ "$used" -lt 2000 ]; then
    break
  fi
  sleep 10
done
echo "[queue] GPU free, starting"

echo '=== FullAttn+QKperhead  proj F4 + AV entropy + QK per-head F4 ==='
python cls/experiments/qwt_sc_compensation.py \
  --sc_config full_attn \
  --n_calib 128 --n_eval 100 --skip_baseline \
  --adaptive_mp 1 --adaptive_mp_levels 256,128,64,32 \
  --adaptive_mp_ops qkv_proj,out_proj,qk,av \
  --adaptive_mp_alpha 0.3 --adaptive_mp_beta 0.05 \
  --adaptive_mp_enable_pruning 0 \
  --adaptive_mp_fractions "0.05,0.35,0.40,0.20" \
  --av_metric_kind entropy \
  --vit_timestep 9 --vit_total_timesteps 10 \
  --out_json $OUT/FullAttn_entropy_qkph.json 2>&1 | tail -15
