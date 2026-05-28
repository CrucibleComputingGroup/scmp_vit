#!/bin/bash
# SC on ALL ops (QK + AV + proj + MLP) under QwT comp.
# Tests whether the linear block-level compensator can rescue SC-MLP
# (which without comp crashes top1 to 0).
set -eu
cd "$(dirname "$0")/.."
OUT=results/qwt_mlp_included
mkdir -p "$OUT"

CALIB=128
EVAL=100

run_uniform () {
  local TAG=$1 SL=$2 MLP_MODE=$3
  echo "=== $TAG  uniform sl=$SL  mlp_mode=$MLP_MODE  (all ops SC including MLP) ===" >&2
  python cls/experiments/qwt_sc_compensation.py \
    --sc_config full_everything \
    --sc_mlp_mode $MLP_MODE --sc_proj_mode bipolar \
    --n_calib $CALIB --n_eval $EVAL \
    --skip_baseline \
    --mp_levels $SL --mp_fractions 1.0 \
    --mp_ops qkv_proj,out_proj,mlp_fc1,mlp_fc2 \
    --out_json $OUT/${TAG}.json 2>&1 | grep -E "SUMMARY|comp top1|r2=|Error|Traceback" | tail -5
  echo >&2
}

run_hetero () {
  local TAG=$1 FRACS=$2 MLP_MODE=$3
  echo "=== $TAG  fractions=$FRACS  mlp_mode=$MLP_MODE ===" >&2
  python cls/experiments/qwt_sc_compensation.py \
    --sc_config full_everything \
    --sc_mlp_mode $MLP_MODE --sc_proj_mode bipolar \
    --n_calib $CALIB --n_eval $EVAL \
    --skip_baseline \
    --adaptive_mp 1 --adaptive_mp_levels 256,128,64,32 \
    --adaptive_mp_ops qkv_proj,out_proj,mlp_fc1,mlp_fc2 \
    --adaptive_mp_alpha 0.3 --adaptive_mp_beta 0.05 \
    --adaptive_mp_enable_pruning 0 \
    --adaptive_mp_fractions "$FRACS" \
    --vit_timestep 9 --vit_total_timesteps 10 \
    --out_json $OUT/${TAG}.json 2>&1 | grep -E "SUMMARY|comp top1|Error|Traceback" | tail -3
  echo >&2
}

# No-MP baselines (straight SC + comp)
run_uniform sl256_bi  256 bipolar
run_uniform sl256_uni 256 unipolar  # MLP often prefers unipolar for post-GELU
run_uniform sl128_bi  128 bipolar
run_uniform sl128_uni 128 unipolar

# Hetero F4 fractions
run_hetero F4_bi "0.05,0.40,0.45,0.10" bipolar
run_hetero F4_uni "0.05,0.40,0.45,0.10" unipolar
run_hetero F1_bi "0.05,0.35,0.40,0.20" bipolar

# ---------- Summary ----------
echo ""
echo "======================= SUMMARY ======================="
python - <<PY
import json
from pathlib import Path
out = Path("$OUT")

def avg_sl(fracs, levels=[256,128,64,32]):
    return sum(f*l for f, l in zip(fracs, levels))

records = [
    ("sl256_bi",  [1.0, 0, 0, 0], "homo",   "bipolar"),
    ("sl256_uni", [1.0, 0, 0, 0], "homo",   "unipolar"),
    ("sl128_bi",  [0, 1.0, 0, 0], "homo",   "bipolar"),
    ("sl128_uni", [0, 1.0, 0, 0], "homo",   "unipolar"),
    ("F4_bi",   [0.05, 0.40, 0.45, 0.10], "hetero", "bipolar"),
    ("F4_uni",  [0.05, 0.40, 0.45, 0.10], "hetero", "unipolar"),
    ("F1_bi",   [0.05, 0.35, 0.40, 0.20], "hetero", "bipolar"),
]
print(f"{'tag':<12s}  {'fractions':<25s}  {'avg_sl':>7s}  {'save%':>6s}  {'top1':>6s}  {'type':>6s}  {'mode':>8s}")
print("-" * 85)
for tag, fracs, ty, mode in records:
    f = out / f"{tag}.json"
    if not f.exists():
        print(f"{tag:<12s}  (missing)")
        continue
    d = json.load(open(f))
    t1 = d["results"]["sc_comp"]["top1"]
    a = avg_sl(fracs)
    save = 1 - a/256
    fstr = "/".join(f"{x:.2f}" for x in fracs)
    print(f"{tag:<12s}  {fstr:<25s}  {a:>7.1f}  {save*100:>5.1f}%  {t1:>6.3f}  {ty:>6s}  {mode:>8s}")
PY
echo "======================================================="
