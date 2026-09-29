#!/usr/bin/env bash
# One MP cell end-to-end with the 2026-07-29 knobs: calibrate -> deploy ->
# eval -> read the REALIZED compute budget back off the precision trace.
#
#   bash cls/mp_cell2.sh <TAG> <BUDGET_HALVED> <LADDER> [N_CALIB] [N_EVAL]
#
# BUDGET_HALVED is the target MAC-weighted average stream length in halved
# cycles; effective levels = 2x it. LADDER's top rung is the budget reference.
#
# Env knobs (empty/0 = off, which reproduces mp_cell.sh exactly):
#   OBJECTIVE={sigma|sigma2|delta_sigma2}   allocation objective
#   CLW={uniform|macs|measured|macs_measured}  cross-layer weight W_g
#   METRIC={amax|auto}                      dispatch-metric selection
#   GLK=<int>                               per-group ladder rungs (needs a
#                                           DENSE LADDER; 0 = one shared ladder)
#   ESC=<float>                             escape-gate k (unset = off)
#   ESC_SL=<int>                            escape stream length (0 = ceiling)
#   PCF=<float>                             protected input-channel fraction
#   PCM={act|act_weight|weight}             protected-channel salience metric
#   PC_SL=<int>                             protected stream length (0 = ceiling)
#   QKG={per_head|per_row}                  Q@K^T scope AND qk MP unit
#   MPOPS=<csv>                             ops that get an MP table
#   ATTN=<int>                              uniform stoc_len for non-MP attention
#   CALIB_SEED=<int>
#   EXCL=<int>   images removed from the CALIBRATION pool (default 2048).
#                MUST be >= the eval n this table will be deployed at, or the
#                eval set is inside the calibration pool and the thresholds saw
#                the images they are scored on. 2048 covers every n<=2048 run;
#                an n=4000 eval needs EXCL=4000.
#
# A cell using a non-default combination needs its OWN tag: the calibration
# cache is keyed by tag alone.
set -uo pipefail

TAG=$1
T=$2
LADDER=$3
NCAL=${4:-256}
NEVAL=${5:-2048}

OBJECTIVE=${OBJECTIVE:-sigma}
CLW=${CLW:-uniform}
METRIC=${METRIC:-amax}
GLK=${GLK:-0}
ESC=${ESC:-}
ESC_SL=${ESC_SL:-0}
PCF=${PCF:-0}
PCM=${PCM:-act}
PC_SL=${PC_SL:-0}
QKG=${QKG:-per_head}
# Argmin cost pricing. 'rep' (rep_g = true rows / stored rows) is only safe
# while every operator in the table sees the SAME rows per forward — true for
# a linear-only ViT table, FALSE once qk/av join, because attention has B*H*N
# rows against a linear's B*N. At H=16 that prices an attention row 16x higher
# than a linear one and the allocator starves attention to the ladder floor
# (measured: qk realized 16.6 cycles against a 32 budget, top-1 0.672 vs
# uniform 0.783). 'macs' is the KKT-consistent price: the subsample ratio
# cancels between benefit and cost.
PRICING=${PRICING:-rep}
ATTN=${ATTN:-$T}
MPOPS=${MPOPS:-qkv_proj,out_proj,mlp_fc1,mlp_fc2}

VIT_SC=/nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc
cd "$VIT_SC" || exit 1
export TORCH_HOME=$VIT_SC/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

CAL=$VIT_SC/results/mp_calib
OUT=$VIT_SC/results/mp_eval
mkdir -p "$CAL" "$OUT"

TOP=${LADDER%%,*}                       # ladder's top rung == budget reference
RATIO=$(python -c "print(f'{$T/$TOP:.6f}')")

EXTRA=()
[ "$GLK" != "0" ] && EXTRA+=(--group-ladder-k "$GLK")
[ -n "$ESC" ] && EXTRA+=(--escape-gate-k "$ESC" --escape-stoc-len "$ESC_SL")
if [ "$PCF" != "0" ]; then
  EXTRA+=(--protect-channel-frac "$PCF" --protect-channel-metric "$PCM"
          --protect-channel-stoc-len "$PC_SL")
fi

echo "=== [$TAG] budget=$T halved (eff $((2*T))) ladder=$LADDER"
echo "=== obj=$OBJECTIVE W_g=$CLW metric=$METRIC glk=$GLK esc=${ESC:-off} pcf=$PCF qk=$QKG pricing=$PRICING"
echo "=== mpops=$MPOPS attn=$ATTN n_calib=$NCAL n_eval=$NEVAL | $(hostname) $(date +%T)"

if [ ! -f "$CAL/${TAG}_wrapper.json" ]; then
  python -u cls/calibrate_mp_thresholds.py \
    --mp_levels "$LADDER" \
    --budget_ratio "$RATIO" --budget_ref_stoc_len "$TOP" \
    --sc_prec 8 --halve 1 --chunk_d 128 \
    --operators "$MPOPS" \
    --attn_stoc_len "$ATTN" \
    --metric-select "$METRIC" \
    --objective "$OBJECTIVE" --cross-layer-weight "$CLW" \
    --argmin-pricing "$PRICING" \
    --qk_granularity "$QKG" \
    --calib_seed "${CALIB_SEED:-1234}" \
    --eval_exclude_n "${EXCL:-2048}" \
    --n_calib_images "$NCAL" --batch_size 16 \
    "${EXTRA[@]}" \
    --output_json "$CAL/$TAG.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|xFormers|warnings.warn" \
    | grep -E "^\[calib\]|rho=|metric=|Error|Traceback|SystemExit"
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
  | grep -E "^\[sc\]|^\[result\]|Error|Traceback|SystemExit"

python tests/summarize_trace.py "$OUT/${TAG}_trace.json" 2>/dev/null \
  | grep -E "realized|per-op|^    " | head -14
echo "=== [$TAG] done $(date +%T)"
echo
