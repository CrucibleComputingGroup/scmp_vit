#!/bin/bash
# SC on full model EXCEPT fc2 in last 4 blocks (blocks 20-23 keep fc2 FP).
# Intuition: late-block fc2 has the worst outliers; freezing them lets the
# linear comp handle earlier blocks' fc2.
set -eu
cd "$(dirname "$0")/.."
OUT=results/qwt_fc2_skip4
mkdir -p "$OUT"

CALIB=128
EVAL=100
K=4

run_uniform () {
  local TAG=$1 SL=$2
  echo "=== $TAG  uniform sl=$SL  (fc2 skip last $K) ===" >&2
  python cls/experiments/qwt_sc_compensation.py \
    --sc_config full_everything \
    --fc2_skip_last_k $K \
    --sc_mlp_mode bipolar --sc_proj_mode bipolar \
    --n_calib $CALIB --n_eval $EVAL \
    --skip_baseline \
    --mp_levels $SL --mp_fractions 1.0 \
    --mp_ops qkv_proj,out_proj,mlp_fc1,mlp_fc2 \
    --out_json $OUT/${TAG}.json 2>&1 | grep -E "SUMMARY|comp top1|patched|Error|Traceback" | tail -4
  echo >&2
}

run_hetero () {
  local TAG=$1 FRACS=$2
  echo "=== $TAG  fractions=$FRACS  (fc2 skip last $K) ===" >&2
  python cls/experiments/qwt_sc_compensation.py \
    --sc_config full_everything \
    --fc2_skip_last_k $K \
    --sc_mlp_mode bipolar --sc_proj_mode bipolar \
    --n_calib $CALIB --n_eval $EVAL \
    --skip_baseline \
    --adaptive_mp 1 --adaptive_mp_levels 256,128,64,32 \
    --adaptive_mp_ops qkv_proj,out_proj,mlp_fc1,mlp_fc2 \
    --adaptive_mp_alpha 0.3 --adaptive_mp_beta 0.05 \
    --adaptive_mp_enable_pruning 0 \
    --adaptive_mp_fractions "$FRACS" \
    --vit_timestep 9 --vit_total_timesteps 10 \
    --out_json $OUT/${TAG}.json 2>&1 | grep -E "SUMMARY|comp top1|patched|Error|Traceback" | tail -3
  echo >&2
}

run_uniform sl256_bi 256
run_uniform sl128_bi 128
run_hetero F4_bi "0.05,0.40,0.45,0.10"
run_hetero F1_bi "0.05,0.35,0.40,0.20"

# ---------- Summary ----------
echo ""
echo "============== SUMMARY (fc2 skip last $K blocks) =============="
python - <<PY
import json
from pathlib import Path
out = Path("$OUT")
def avg_sl(fracs, levels=[256,128,64,32]):
    return sum(f*l for f, l in zip(fracs, levels))

records = [
    ("sl256_bi", [1.0, 0, 0, 0], "homo"),
    ("sl128_bi", [0, 1.0, 0, 0], "homo"),
    ("F4_bi", [0.05, 0.40, 0.45, 0.10], "hetero"),
    ("F1_bi", [0.05, 0.35, 0.40, 0.20], "hetero"),
]
print(f"{'tag':<12s}  {'fractions':<25s}  {'avg_sl':>7s}  {'save%':>6s}  {'top1':>6s}  {'type':>6s}")
print("-" * 75)
for tag, fracs, ty in records:
    f = out / f"{tag}.json"
    if not f.exists():
        print(f"{tag:<12s}  (missing)")
        continue
    d = json.load(open(f))
    t1 = d["results"]["sc_comp"]["top1"]
    a = avg_sl(fracs)
    save = 1 - a/256
    fstr = "/".join(f"{x:.2f}" for x in fracs)
    print(f"{tag:<12s}  {fstr:<25s}  {a:>7.1f}  {save*100:>5.1f}%  {t1:>6.3f}  {ty:>6s}")
PY
echo "================================================================"
