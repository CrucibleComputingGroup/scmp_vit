#!/bin/bash
# Sweep target_fractions configs for adaptive MP + QwT comp on ViT-L/14.
# Compares several custom fraction distributions against uniform sl=128 / sl=64
# baselines. Prints a summary table at the end.
set -eu
cd "$(dirname "$0")/.."
OUT=results/qwt_custom_fractions
mkdir -p "$OUT"

CALIB=128
EVAL=100
LEV=256,128,64,32

# Each row: tag  f256  f128  f64  f32
CONFIGS=(
  "F1_ideal        0.05 0.35 0.40 0.20"
  "F2_more256      0.10 0.30 0.40 0.20"
  "F3_no256        0.00 0.40 0.40 0.20"
  "F4_less_lowest  0.05 0.40 0.45 0.10"
  "F5_more_lowest  0.05 0.30 0.35 0.30"
)

# Also reference runs (reuse results if present; compute otherwise)
run_ref_uniform () {
  local SL=$1
  local TAG=REF_sl${SL}
  if [[ ! -f "$OUT/${TAG}.json" ]]; then
    echo "=== $TAG  uniform sl=$SL ===" >&2
    python cls/experiments/qwt_sc_compensation.py \
      --sc_config full_attn --n_calib $CALIB --n_eval $EVAL \
      --skip_baseline \
      --mp_levels $SL --mp_fractions 1.0 --mp_ops qkv_proj,out_proj \
      --out_json $OUT/${TAG}.json 2>&1 | grep -E "SUMMARY|comp top1|Error|Traceback" | tail -3
  fi
}

run_custom () {
  local TAG=$1 F256=$2 F128=$3 F64=$4 F32=$5
  echo "=== $TAG  fractions=($F256,$F128,$F64,$F32) ===" >&2
  python cls/experiments/qwt_sc_compensation.py \
    --sc_config full_attn --n_calib $CALIB --n_eval $EVAL \
    --skip_baseline \
    --adaptive_mp 1 --adaptive_mp_levels $LEV \
    --adaptive_mp_ops qkv_proj,out_proj \
    --adaptive_mp_alpha 0.3 --adaptive_mp_beta 0.05 \
    --adaptive_mp_enable_pruning 0 \
    --adaptive_mp_fractions "$F256,$F128,$F64,$F32" \
    --vit_timestep 9 --vit_total_timesteps 10 \
    --out_json $OUT/${TAG}.json 2>&1 | grep -E "SUMMARY|comp top1" | tail -3
  echo >&2
}

# Run references (sl=256 / 128 / 64)
for SL in 256 128 64; do
  run_ref_uniform $SL
done

# Run custom fraction sweeps
for row in "${CONFIGS[@]}"; do
  read -r TAG F256 F128 F64 F32 <<< "$row"
  run_custom "$TAG" "$F256" "$F128" "$F64" "$F32"
done

# ---------- Summary ----------
echo ""
echo "===================== SUMMARY ====================="
python - <<PY
import json
from pathlib import Path
out = Path("$OUT")
rows = []

def avg_sl(fracs, levels=[256,128,64,32]):
    return sum(f*l for f, l in zip(fracs, levels))

# references
for sl in (256, 128, 64):
    f = out / f"REF_sl{sl}.json"
    if not f.exists(): continue
    d = json.load(open(f))
    t1 = d["results"]["sc_comp"]["top1"]
    save = 1 - sl/256
    rows.append(("REF_sl%d" % sl, [1.0 if i==[256,128,64,32].index(sl) else 0 for i in range(4)], sl, save, t1, "homo"))

configs = [
    ("F1_ideal",       [0.05, 0.35, 0.40, 0.20]),
    ("F2_more256",     [0.10, 0.30, 0.40, 0.20]),
    ("F3_no256",       [0.00, 0.40, 0.40, 0.20]),
    ("F4_less_lowest", [0.05, 0.40, 0.45, 0.10]),
    ("F5_more_lowest", [0.05, 0.30, 0.35, 0.30]),
]
for tag, fracs in configs:
    f = out / f"{tag}.json"
    if not f.exists(): continue
    d = json.load(open(f))
    t1 = d["results"]["sc_comp"]["top1"]
    a = avg_sl(fracs)
    save = 1 - a/256
    rows.append((tag, fracs, a, save, t1, "hetero"))

print(f"{'tag':<18s}  {'fractions (256/128/64/32)':<25s}  {'avg_sl':>7s}  {'save%':>6s}  {'top1':>6s}  {'type':>6s}")
print("-" * 80)
for tag, fracs, a, save, t1, ty in sorted(rows, key=lambda r: -r[2]):
    fstr = "/".join(f"{f:.2f}" for f in fracs)
    print(f"{tag:<18s}  {fstr:<25s}  {a:>7.1f}  {save*100:>5.1f}%  {t1:>6.3f}  {ty:>6s}")
PY
echo "==================================================="
