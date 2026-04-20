#!/bin/bash
# Entropy vs -amax for AV MP classifier. Compare on AV-only and full_attn.
set -eu
cd "$(dirname "$0")/.."
OUT=results/qwt_av_entropy
mkdir -p "$OUT"

CALIB=128
EVAL=100
FRAC="0.05,0.35,0.40,0.20"

run () {
  local TAG=$1 PRESET=$2 METRIC=$3 REV=$4
  local EXTRA_OPS=""
  if [ "$PRESET" = "full_attn" ]; then
    EXTRA_OPS="--adaptive_mp_ops qkv_proj,out_proj,av"
  else
    EXTRA_OPS="--adaptive_mp_ops av"
  fi
  echo "=== $TAG  preset=$PRESET metric=$METRIC reverse=$REV ===" >&2
  python cls/experiments/qwt_sc_compensation.py \
    --sc_config $PRESET \
    --n_calib $CALIB --n_eval $EVAL --skip_baseline \
    --adaptive_mp 1 --adaptive_mp_levels 256,128,64,32 \
    $EXTRA_OPS \
    --adaptive_mp_alpha 0.3 --adaptive_mp_beta 0.05 \
    --adaptive_mp_enable_pruning 0 \
    --adaptive_mp_fractions "$FRAC" \
    --av_metric_kind $METRIC \
    --av_metric_reverse $REV \
    --vit_timestep 9 --vit_total_timesteps 10 \
    --out_json $OUT/${TAG}.json 2>&1 | grep -E "top1=|SUMMARY|Traceback|Error" | tail -3
  echo >&2
}

# AV-only comparisons
run AVonly_amax_rev   av_only amax 1
run AVonly_entropy    av_only entropy 0
run AVonly_ipr        av_only ipr 0

# Full-attn comparisons (more stringent, downstream SC amplifies differences)
run FullAttn_amax_rev full_attn amax 1
run FullAttn_entropy  full_attn entropy 0
run FullAttn_ipr      full_attn ipr 0

echo ""
echo "============ ENTROPY vs AMAX (AV metric) SUMMARY ============"
python - <<PY
import json
from pathlib import Path
out = Path("$OUT")
records = [
    ("AVonly_amax_rev",   "av_only",   "−amax"),
    ("AVonly_entropy",    "av_only",   "entropy"),
    ("AVonly_ipr",        "av_only",   "ipr"),
    ("FullAttn_amax_rev", "full_attn", "−amax"),
    ("FullAttn_entropy",  "full_attn", "entropy"),
    ("FullAttn_ipr",      "full_attn", "ipr"),
]
print(f"{'tag':<22s}  {'preset':<10s}  {'metric':<8s}  {'FP':>6s}  {'SC+comp':>8s}  {'Δ':>5s}")
print("-"*72)
for tag, preset, metric in records:
    f = out / f"{tag}.json"
    if not f.exists():
        print(f"{tag:<22s}  (missing)"); continue
    d = json.load(open(f))
    fp = d["results"]["fp"]["top1"]
    sc = d["results"]["sc_comp"]["top1"]
    print(f"{tag:<22s}  {preset:<10s}  {metric:<8s}  {fp:>6.3f}  {sc:>8.3f}  {sc-fp:>+5.3f}")
PY
echo "=============================================================="
