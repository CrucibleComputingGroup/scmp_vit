#!/usr/bin/env bash
# Why is the hybrid mask inert at eff-96? Four experiments, one per GPU.
#
#   MASK=<path> TAG=<tag> bash cls/diag_eff96.sh      # one eval lane
#   PROBE=1 bash cls/diag_eff96.sh                    # the int_swap probe lane
#
# THE ANOMALY. On the UNIFORM arm the 10% INT mask is worth +2.95 pt at eff-64
# and +0.88 at eff-128, but -0.02 at eff-96. Non-monotone, so it is not
# saturation -- eff-128 has LESS headroom and still gains.
#
# Four explanations were already ruled out with data on disk:
#   ranking noise   t48's top-14 sit 2.28 sd above its own tail, the BEST of the
#                   four budgets (t32 2.01, t64 1.90, t128 1.99)
#   composition     t48 and t64 masks are both mlp_fc2-dominated at 5.33% of MACs
#   execution       hyb_uni_48raw's trace is missing exactly the 14 masked groups
#   saturation      see above
#
# What is left is WHICH BLOCKS: t48 and t64 share ZERO of their 14 groups. These
# four cells separate the remaining hypotheses, and each is one variable off the
# failing cell (uniform SC @48 + mask10_t48 = 0.8273 vs uni_48 = 0.8275):
#
#   xfer   mask10_t64 at eff-96      Does a mask ranked at ANOTHER budget work
#                                    here? If yes, t48's selection is the fault
#                                    and the probe is budget-overfitting.
#   dose   mask20_t48 at eff-96      Is 10% simply too small a dose at this
#                                    budget? Nested with the 10% mask, so this
#                                    is a pure dose axis.
#   rand   maskrand10_t48            Per-op counts and MAC fraction (0.0975)
#                                    matched exactly; only WHICH blocks differ.
#                                    If rand == ranked == 0, selection is
#                                    irrelevant and the budget is mask-immune.
#   swap   int_swap ranking          Rank by measured L(g in INT) - L(all SC)
#                                    instead of by SC fragility. The fragility
#                                    proxy cannot express "INT would make this
#                                    group WORSE", which is the failure mode a
#                                    fragile-but-unhelpable group produces.
set -uo pipefail

_WANT_N=${N:-10000}
_WANT_OUT=${OUT:-results/final10k}
source cls/sc_lib.sh
export N=$_WANT_N
export OUT=$_WANT_OUT

if [ "${PROBE:-0}" = "1" ]; then
  echo "########## eff-96 diag: int_swap probe  $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(date +%T)"
  # Probe images come from OUTSIDE the reported 10000, unlike the original
  # fragility probes, which defaulted to --eval_exclude_n 2048 and therefore
  # fitted their masks on images inside the n=10000 report set.
  python -u cls/build_intswap_mask.py --stoc_len 48 --int_bits 7 --int_sym 0 \
    --tag t48 --n_images 128 --batch_size 16 --eval_exclude_n 10000 \
    --data_root "$DATA" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[swap\]|Error|Traceback"
  echo "########## probe COMPLETE $(date +%T)"
  exit 0
fi

MASK=${MASK:?set MASK=<mask path>}
TAG=${TAG:?set TAG=<cell tag>}
echo "########## eff-96 diag: $TAG  mask=$(basename "$MASK")  arm=${ARM:-uniform} gpu=$CUDA_VISIBLE_DEVICES $(date +%T)"

if [ "${ARM:-uniform}" = "mp" ]; then
  # Deploy mp_48's EXISTING table unchanged, at its own budget, plus the mask.
  # No recalibration: that keeps the MP allocation identical to the unmasked
  # reference, so the only variable between mp_48 and this cell is the mask.
  W=results/mp_calib/mp_48_wrapper.json
  [ -f "$W" ] || { echo "[diag] no $W"; exit 1; }
  if [ -f "$OUT/$TAG.json" ]; then echo "[diag] $TAG cached"; exit 0; fi
  SC_MP_TRACE=$OUT/${TAG}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len 48 \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table "$W" --adaptive_mp_ops "$ALLOPS" \
    --hybrid_config_json "$MASK" \
    --data_root "$DATA" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$TAG.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[hybrid\]|^\[result\]|^\[eval\] per-image|Error|Traceback"
else
  LADDER= GLK=0 PCF=0 sc_cell "$TAG" 48 --hybrid_config_json "$MASK"
fi
echo "########## $TAG COMPLETE $(date +%T)"
