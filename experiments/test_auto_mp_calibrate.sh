#!/usr/bin/env bash
# Scale-up runner for auto-MP calibration on DINOv2 ViT-L/14.
#
# Reproduces the Pareto-best config from commit e16583b as the comparison
# baseline ('fractions=0.05,0.40,0.45,0.10'), then runs the same config
# with --auto_mp_calibrate to let oracle search populate free boundaries
# per (block, op). Expectation: auto >= fractions, since fractions is a
# strict subset of what free boundaries can express.
#
# Run from repo root. Assumes CUDA GPU and ImageNet parquet data at
# --data_root. Adjust n_calib / n_eval for faster smoke.
set -euo pipefail

cd "$(dirname "$0")/.."

SC_CONFIG=${SC_CONFIG:-full_attn}
LEVELS=${LEVELS:-256,128,64,32}
OPS=${OPS:-av}
N_CALIB=${N_CALIB:-128}
N_EVAL=${N_EVAL:-100}
OUT_DIR=${OUT_DIR:-results/auto_mp_pilot}
mkdir -p "$OUT_DIR"

echo "================ [auto_mp] zero-hyperparameter boundary search ================"
python cls/experiments/qwt_sc_compensation.py \
    --sc_config "$SC_CONFIG" --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
    --adaptive_mp 1 --adaptive_mp_levels "$LEVELS" --adaptive_mp_ops "$OPS" \
    --auto_mp_calibrate "$N_CALIB" \
    --auto_mp_n_candidates 8 \
    --auto_mp_outer_per_op 2 \
    --auto_mp_outer_block 1 \
    --comp_mode fp \
    --out_json "$OUT_DIR/auto_mp_${SC_CONFIG}.json" 2>&1 | tee "$OUT_DIR/auto_mp_${SC_CONFIG}.log"

echo
echo "================ [baseline] manual fractions (0.05/0.40/0.45/0.10) ================"
python cls/experiments/qwt_sc_compensation.py \
    --sc_config "$SC_CONFIG" --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
    --adaptive_mp 1 --adaptive_mp_levels "$LEVELS" --adaptive_mp_ops "$OPS" \
    --adaptive_mp_fractions "0.05,0.40,0.45,0.10" \
    --comp_mode fp \
    --out_json "$OUT_DIR/fractions_${SC_CONFIG}.json" 2>&1 | tee "$OUT_DIR/fractions_${SC_CONFIG}.log"

echo
echo "================ [compare] ================"
python - <<'PY'
import json, sys
from pathlib import Path
d = Path("results/auto_mp_pilot")
for tag in ["auto_mp", "fractions"]:
    j = json.loads((d / f"{tag}_full_attn.json").read_text())
    r = j["results"]
    print(f"  {tag:>10s}  fp={r['fp']['top1']:.4f}  "
          f"sc_raw={(r.get('sc_raw') or {}).get('top1', float('nan')):.4f}  "
          f"sc_comp={r['sc_comp']['top1']:.4f}")
PY
