#!/usr/bin/env bash
# Objective/W_g ablation at ONE budget, off ONE sigma draw.
#
#   bash cls/mp_variant_sweep.sh <TAG> <BUDGET_HALVED> <LADDER> [ATTN_SL] [N_CALIB] [N_EVAL]
#
# Calibrates once with --variants all (6 tables = 3 objectives x 2 W_g), then
# evaluates every table on the same image subset. Because all six share one
# sigma collection, the between-variant deltas carry NO calibration-sample
# noise — which matters, since that noise is ~0.33 pt on top-1 at n_calib=256
# and the effects here are ~1 pt (MP_REPORT.md).
#
# qk/av are not MP-dispatchable and run uniform at ATTN_SL; they still consume
# budget, so always read the trace-measured realized average, not the nominal
# target. Cells are resumable: an existing output json is skipped.
set -uo pipefail

TAG=$1
T=$2
LADDER=$3
ATTN=${4:-$T}
NCAL=${5:-256}
NEVAL=${6:-2048}
METRIC=${METRIC:-amax}
# Which (objective, W_g) pairs to export+evaluate. 'all' = the full 3x2 grid.
# Restrict it (e.g. 'sigma:uniform,sigma:macs') to replicate just a control and
# a winner across --calib_seed values without paying for the other four cells.
VARIANTS=${VARIANTS:-all}

VIT_SC=/nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc
cd "$VIT_SC" || exit 1
export TORCH_HOME=$VIT_SC/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}
MPOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2

CAL=$VIT_SC/results/mp_calib
OUT=$VIT_SC/results/mp_eval
mkdir -p "$CAL" "$OUT"

TOP=${LADDER%%,*}                       # ladder's top rung == budget reference
RATIO=$(python -c "print(f'{$T/$TOP:.6f}')")
QUIET='futurewarning|pynvml|xFormers|warnings.warn'

# Resolve VARIANTS into the table tags the calibrator will write. Must mirror
# _parse_variants() in calibrate_mp_thresholds.py: a bare objective means
# W_g=uniform, and 'all' is W_g-major over the three objectives.
if [ "$VARIANTS" = "all" ]; then
  TAGS="sigma_wuniform sigma2_wuniform delta_sigma2_wuniform \
        sigma_wmacs sigma2_wmacs delta_sigma2_wmacs"
else
  TAGS=""
  IFS=',' read -ra _items <<< "$VARIANTS"
  for it in "${_items[@]}"; do
    it="${it// /}"; [ -n "$it" ] || continue
    case "$it" in *:*) o="${it%%:*}"; w="${it##*:}";; *) o="$it"; w=uniform;; esac
    TAGS="$TAGS ${o}_w${w}"
  done
fi

echo "### [$TAG] budget=$T halved (eff $((2*T))) ladder=$LADDER attn=$ATTN metric=$METRIC"
echo "### ratio=$RATIO x $TOP | n_calib=$NCAL n_eval=$NEVAL calib_seed=${CALIB_SEED:-1234}"
echo "### variants:$TAGS | $(hostname) $(date)"

# ---- one sigma collection -> N tables -------------------------------------
NEED=0
for V in $TAGS; do
  [ -f "$CAL/${TAG}__${V}_wrapper.json" ] || NEED=1
done
if [ "$NEED" = 1 ]; then
  python -u cls/calibrate_mp_thresholds.py \
    --mp_levels "$LADDER" \
    --budget_ratio "$RATIO" --budget_ref_stoc_len "$TOP" \
    --sc_prec 8 --halve 1 --chunk_d 128 \
    --operators "$MPOPS" \
    --attn_stoc_len "$ATTN" \
    --metric-select "$METRIC" \
    --variants "$VARIANTS" \
    --calib_seed "${CALIB_SEED:-1234}" \
    --n_calib_images "$NCAL" --batch_size 16 \
    --output_json "$CAL/$TAG.json" 2>&1 \
    | grep -viE "$QUIET" \
    | grep -E "^\[calib\]|^variant|^-{10}|^(sigma|delta)|Error|Traceback"
else
  echo "[sweep] calibration cached"
fi

# ---- evaluate every variant ----------------------------------------------
for V in $TAGS; do
  W="$CAL/${TAG}__${V}_wrapper.json"
  [ -f "$W" ] || { echo "[sweep] MISSING $W — skipped"; continue; }
  NAME="${TAG}__${V}"
  if [ -f "$OUT/$NAME.json" ]; then echo "=== [$NAME] cached ==="; continue; fi
  echo "=== [$NAME] $(date +%T) ==="
  SC_MP_TRACE=$OUT/${NAME}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj \
    --stoc_len "$ATTN" \
    --adaptive_mp 1 --adaptive_mp_table "$W" \
    --adaptive_mp_ops "$MPOPS" \
    --data_root "$D" --max_images "$NEVAL" --batch_size 16 --workers 4 --seed 0 \
    --out_json "$OUT/$NAME.json" 2>&1 \
    | grep -viE "$QUIET" \
    | grep -E "^\[sc\]|^\[result\]|Error|Traceback"
done

echo "### [$TAG] done $(date)"
