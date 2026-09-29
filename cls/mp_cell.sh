#!/usr/bin/env bash
# One MP cell end-to-end: calibrate -> deploy -> eval -> measure the REALIZED
# compute budget from the precision trace.
#
#   bash cls/mp_cell.sh <TAG> <BUDGET_HALVED> <LADDER> <METRIC> <ATTN_SL> [N_CALIB] [N_EVAL]
#
# BUDGET_HALVED is the target MAC-weighted average stream length in halved
# cycles; effective levels = 2x it. ATTN_SL pins qk/av (not MP-dispatchable,
# see calibrate_mp_thresholds.py's docstring) — they still consume budget, so
# the realized average reported at the end is what any comparison must use,
# NOT the nominal target.
#
# Env knobs (defaults reproduce every cell run before 2026-07-25):
#   OBJECTIVE={sigma|sigma2|delta_sigma2}   allocation objective
#   CLW={uniform|macs}                      cross-layer importance weight W_g
#   CALIB_SEED=<int>                        calibration subset shuffle seed
# CLW=macs is the fix for act_global pinning the cheap out_proj at the ladder
# ceiling while starving mlp_fc1 (MP_REPORT.md). Give a cell using a non-default
# pair its OWN tag — the calibration cache is keyed by tag alone.
set -uo pipefail

TAG=$1
T=$2
LADDER=$3
METRIC=${4:-amax}
ATTN=${5:-$T}
NCAL=${6:-256}
NEVAL=${7:-2048}
OBJECTIVE=${OBJECTIVE:-sigma}
CLW=${CLW:-uniform}

VIT_SC=/nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc
cd "$VIT_SC" || exit 1
export TORCH_HOME=$VIT_SC/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}
# Operators that get an MP table. Override with MPOPS=... to include qk,av —
# their calibration units match the runtime dispatch (qk per head, av per row
# pooled globally), and their MACs are already in the budget either way, so
# adding them is iso-compute by construction.
MPOPS=${MPOPS:-qkv_proj,out_proj,mlp_fc1,mlp_fc2}

CAL=$VIT_SC/results/mp_calib
OUT=$VIT_SC/results/mp_eval
mkdir -p "$CAL" "$OUT"

TOP=${LADDER%%,*}                       # ladder's top rung == budget reference
RATIO=$(python -c "print(f'{$T/$TOP:.6f}')")

echo "=== [$TAG] budget=$T halved (eff $((2*T))) ladder=$LADDER metric=$METRIC attn=$ATTN"
echo "=== ratio=$RATIO x $TOP | objective=$OBJECTIVE W_g=$CLW | n_calib=$NCAL n_eval=$NEVAL | $(hostname) $(date +%T)"

if [ ! -f "$CAL/${TAG}_wrapper.json" ]; then
  python -u cls/calibrate_mp_thresholds.py \
    --mp_levels "$LADDER" \
    --budget_ratio "$RATIO" --budget_ref_stoc_len "$TOP" \
    --sc_prec 8 --halve 1 --chunk_d 128 \
    --operators "$MPOPS" \
    --attn_stoc_len "$ATTN" \
    --metric-select "$METRIC" \
    --objective "$OBJECTIVE" --cross-layer-weight "$CLW" \
    --calib_seed "${CALIB_SEED:-1234}" \
    --n_calib_images "$NCAL" --batch_size 16 \
    --output_json "$CAL/$TAG.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|xFormers|warnings.warn" \
    | grep -E "^\[calib\]|rho=|metric=|Error|Traceback"
else
  echo "[cell] calibration cached"
fi
[ -f "$CAL/${TAG}_wrapper.json" ] || { echo "[cell] FAILED: no wrapper"; exit 1; }

SC_MP_TRACE=$OUT/${TAG}_trace.json python -u cls/eval.py \
  --mode sc --sc_prec 8 --halve 1 --chunk_d 128 \
  --sc_ops mlp_fc1,mlp_fc2,qk,av,proj \
  --stoc_len "$ATTN" \
  --adaptive_mp 1 --adaptive_mp_table "$CAL/${TAG}_wrapper.json" \
  --adaptive_mp_ops "$MPOPS" \
  --data_root "$D" --max_images "$NEVAL" --batch_size 16 --workers 4 --seed 0 \
  --out_json "$OUT/$TAG.json" 2>&1 \
  | grep -viE "futurewarning|pynvml|xFormers|warnings.warn" \
  | grep -E "^\[sc\]|^\[result\]|Error|Traceback"

python tests/summarize_trace.py "$OUT/${TAG}_trace.json" 2>/dev/null \
  | grep -E "realized|per-op|^    " | head -12
echo "=== [$TAG] done $(date +%T)"
echo
