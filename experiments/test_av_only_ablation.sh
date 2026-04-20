#!/bin/bash
# Ablation: only AV goes SC; QK / proj / MLP all stay FP.
# Goal: quantify AV's isolated accuracy cost under different MP strategies.
set -eu
cd "$(dirname "$0")/.."
OUT=results/qwt_av_ablation
mkdir -p "$OUT"

CALIB=128
EVAL=100

run_uniform () {
  local TAG=$1 LEVELS=$2 FRACS=$3
  echo "=== $TAG  AV uniform via levels=$LEVELS fracs=$FRACS ===" >&2
  python cls/experiments/qwt_sc_compensation.py \
    --sc_config av_only \
    --n_calib $CALIB --n_eval $EVAL --skip_baseline \
    --adaptive_mp 1 --adaptive_mp_levels "$LEVELS" \
    --adaptive_mp_ops av \
    --adaptive_mp_alpha 0.3 --adaptive_mp_beta 0.05 \
    --adaptive_mp_enable_pruning 0 \
    --adaptive_mp_fractions "$FRACS" \
    --vit_timestep 9 --vit_total_timesteps 10 \
    --out_json $OUT/${TAG}.json 2>&1 | grep -E "comp top1|SUMMARY|patched" | tail -3
  echo >&2
}

run_adaptive () {
  local TAG=$1 FRACS=$2 REVERSE=$3
  echo "=== $TAG  AV adaptive $FRACS  reverse=$REVERSE ===" >&2
  python cls/experiments/qwt_sc_compensation.py \
    --sc_config av_only \
    --n_calib $CALIB --n_eval $EVAL --skip_baseline \
    --adaptive_mp 1 --adaptive_mp_levels 256,128,64,32 \
    --adaptive_mp_ops av \
    --adaptive_mp_alpha 0.3 --adaptive_mp_beta 0.05 \
    --adaptive_mp_enable_pruning 0 \
    --adaptive_mp_fractions "$FRACS" \
    --av_metric_reverse $REVERSE \
    --vit_timestep 9 --vit_total_timesteps 10 \
    --out_json $OUT/${TAG}.json 2>&1 | grep -E "comp top1|SUMMARY|patched|adaptive_mp|av_metric" | tail -4
  echo >&2
}

# Baselines via degenerate fractions
run_uniform U256 "256,128" "1.0,0.0"  # all rows at sl=256
run_uniform U128 "256,128" "0.0,1.0"  # all rows at sl=128
run_uniform U64  "128,64"  "0.0,1.0"  # all rows at sl=64

# Adaptive + direction sweeps
run_adaptive A_05_fwd   "0.05,0.35,0.40,0.20" 0
run_adaptive A_05_rev   "0.05,0.35,0.40,0.20" 1
run_adaptive A_00_fwd   "0.00,0.50,0.40,0.10" 0
run_adaptive A_00_rev   "0.00,0.50,0.40,0.10" 1

echo ""
echo "================= AV-ONLY ABLATION SUMMARY ================="
python - <<PY
import json
from pathlib import Path
out = Path("$OUT")
def avg_sl(fracs, levels=[256,128,64,32]):
    return sum(f*l for f, l in zip(fracs, levels))

rows = []
for tag, fracs, typ in [
    ("U256",      [1.0, 0, 0, 0],           "uniform"),
    ("U128",      [0, 1.0, 0, 0],           "uniform"),
    ("U64",       [0, 0, 1.0, 0],           "uniform"),
    ("A_05_fwd",  [0.05, 0.35, 0.40, 0.20], "adaptive fwd"),
    ("A_05_rev",  [0.05, 0.35, 0.40, 0.20], "adaptive rev"),
    ("A_00_fwd",  [0.00, 0.50, 0.40, 0.10], "adaptive fwd"),
    ("A_00_rev",  [0.00, 0.50, 0.40, 0.10], "adaptive rev"),
]:
    f = out / f"{tag}.json"
    if not f.exists(): continue
    d = json.load(open(f))
    fp = d["results"]["fp"]["top1"]
    sc = d["results"]["sc_comp"]["top1"]
    a = avg_sl(fracs)
    save = 1 - a / 256
    rows.append((tag, fracs, a, save, sc, fp, typ))
print(f"{'tag':<12s}  {'fractions':<25s}  {'avg_sl':>7s}  {'save%':>6s}  {'FP':>5s}  {'SC+comp':>8s}  {'Δ':>5s}  {'type':<14s}")
print("-" * 100)
for tag, fracs, a, save, sc, fp, typ in rows:
    fstr = "/".join(f"{x:.2f}" for x in fracs)
    d = sc - fp
    print(f"{tag:<12s}  {fstr:<25s}  {a:>7.1f}  {save*100:>5.1f}%  {fp:>5.3f}  "
          f"{sc:>8.3f}  {d:>+5.3f}  {typ:<14s}")
PY
echo "============================================================"
