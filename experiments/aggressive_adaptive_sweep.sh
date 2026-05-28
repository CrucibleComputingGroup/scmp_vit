#!/bin/bash
# Aggressive adaptive MP sweep on DINOv2 ViT-L/14.
# Runs both accuracy (cls/eval.py on 500 imgs) and compute-savings (measure_adaptive_savings.py on 32 imgs)
# for each config. Results collected into results/e2e/aggressive/*.json.
set -eu
cd "$(dirname "$0")/.."

N=500
OUT=results/e2e/aggressive
mkdir -p "$OUT"

# Each row: TAG LEVELS ALPHA BETA PRUNING T T_TOTAL
CONFIGS=(
  "L128_p1_a03_b05         256,128       0.3 0.05 0 9 10"
  "L64_p1_a03_b05          256,128,64    0.3 0.05 0 9 10"
  "L64prune_p1_a03_b05     256,128,64,0  0.3 0.05 1 9 10"
  "L64prune_p1_a05_b20     256,128,64,0  0.5 0.20 1 9 10"
  "L64big_p1_a03_b05       256,64,0      0.3 0.05 1 9 10"
  "L32big_p1_a03_b05       256,32,0      0.3 0.05 1 9 10"
  "L32big_p1_a05_b30       256,32,0      0.5 0.30 1 9 10"
)

for row in "${CONFIGS[@]}"; do
  read -r TAG LEVELS A B PRUNE T TT <<< "$row"
  echo "=== $TAG  levels=$LEVELS α=$A β=$B prune=$PRUNE t=$T/T=$TT ==="
  python cls/eval.py --mode sc \
    --sc_qk 1 --sc_qkv_proj 1 --sc_out_proj 1 \
    --adaptive_mp 1 --adaptive_mp_levels "$LEVELS" \
    --adaptive_mp_ops qkv_proj,out_proj \
    --adaptive_mp_alpha "$A" --adaptive_mp_beta "$B" \
    --adaptive_mp_enable_pruning "$PRUNE" \
    --vit_timestep "$T" --vit_total_timesteps "$TT" \
    --max_images "$N" --batch_size 32 --workers 4 \
    --out_json "$OUT/${TAG}.json" 2>&1 | \
    grep -E '\[result|adaptive_mp t' | tail -2

  python experiments/measure_adaptive_savings.py --n 32 --batch_size 16 \
    --levels "$LEVELS" --ops qkv_proj,out_proj \
    --alpha "$A" --beta "$B" \
    --timesteps "${T}/${TT}" \
    --out "$OUT/${TAG}.savings.json" 2>&1 | tail -4
  echo
done
