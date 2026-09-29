#!/usr/bin/env bash
# The hybrid arms WITHOUT the 1/(1-f) budget raise, at n=10000.
#
#   HALF=a bash cls/run_unraised_10k.sh    # eff-64, eff-96
#   HALF=b bash cls/run_unraised_10k.sh    # eff-128
#
# ===================================================== WHY THIS EXISTS
# A masked group's SC cycles are not spent. vit_sc's hybrid cells hand them back
# to the surviving groups by raising the target to T/(1-f). Write the three
# quantities out and it is clear that "iso" is doing a lot of work there
# (M = total MACs, f = masked MAC fraction, T = budget):
#
#   unmasked reference        SC = M*T           INT = 0
#   hybrid, NOT raised        SC = (1-f)*M*T     INT = f*M
#   hybrid, raised to T/(1-f) SC = M*T           INT = f*M
#
# The raised cell matches the reference's SC cycles and then does f*M of INT ON
# TOP. So it is iso only if INT is free; in total work it is strictly the more
# expensive cell. The unraised cell is the conservative comparison: it spends
# FEWER SC cycles than the reference and pays f*M in INT for the difference.
#
# scmp_llm does not raise. Its CLAUDE.md prices the INT mask "on the energy
# axis, NOT iso-total-compute" and notes the unraised variants land "closer to
# the nominal budget if a strict iso-budget table is wanted". vit_sc diverged
# from that without recording a reason.
#
# WHAT THIS MAY COST US, stated up front: at n=1000 ordering (3) hybrid MP > MP
# was +2.10 raised and **-0.10 unraised**. If that reproduces at n=10000, (3)
# does not survive the conservative accounting and the report has to say so.
#
# NO RECALIBRATION. The unraised hybrid is the existing MP cell plus a mask at
# its own budget, so it deploys mp_<T>'s table unchanged. Six evals, no
# calibration -- which also means the MP allocation is identical to the unmasked
# reference's, making this a clean one-variable comparison.
set -uo pipefail

_WANT_N=${N:-10000}
_WANT_OUT=${OUT:-results/final10k}
source cls/sc_lib.sh
export N=$_WANT_N
export OUT=$_WANT_OUT
HALF=${HALF:-all}
echo "[raw] resolved: N=$N OUT=$OUT half=$HALF"

mask_for () {   # T -> level-matched mask, same as run_final_10k.sh
  case "$1" in
    32) echo results/hybrid/mask10_t32_int6.json ;;
    48) echo results/hybrid/mask10_t48.json ;;
    64) echo results/hybrid/mask10_t64.json ;;
    *)  echo results/hybrid/mask10_t$1.json ;;
  esac
}

# uniform + mask at the NOMINAL budget, and MP + mask at the NOMINAL budget
# using the unmasked cell's own table.
raw_pair () {   # T
  local t=$1 mask; mask=$(mask_for "$t")
  [ -f "$mask" ] || { echo "[raw] no $mask"; return; }
  echo "########## eff-$((t*2)) unraised, mask=$(basename "$mask")  $(date +%T)"
  LADDER= GLK=0 PCF=0 sc_cell "hyb_uni_${t}raw" "$t" --hybrid_config_json "$mask"

  local w=results/mp_calib/mp_${t}_wrapper.json
  [ -f "$w" ] || { echo "[raw] no table $w -- run run_final_10k.sh first"; return; }
  local tag="hyb_mp_${t}raw"
  if [ -f "$OUT/$tag.json" ]; then echo "[raw] $tag cached"; return; fi
  echo "=== [$tag] MP@$t + mask, table=mp_$t (unchanged) n=$N $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$t" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table "$w" --adaptive_mp_ops "$ALLOPS" \
    --hybrid_config_json "$mask" \
    --data_root "$DATA" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[hybrid\]|^\[result\]|^\[eval\] per-image|Error|Traceback"
}

[ "$HALF" = "a" ] || [ "$HALF" = "all" ] && { raw_pair 32; raw_pair 48; }
[ "$HALF" = "b" ] || [ "$HALF" = "all" ] && { raw_pair 64; }

echo "########## unraised vs raised, side by side $(date +%T)"
for t in 32 48 64; do
  [ -f "$OUT/hyb_mp_${t}raw.json" ] || continue
  h=hyb_mp_$t; [ -f "$OUT/hyb_mp_${t}2.json" ] && h=hyb_mp_${t}2
  echo ""
  echo "===== eff-$((t*2)) ====="
  echo "--- (3) hybrid MP > MP"
  echo "    raised:"
  python cls/paired_stats.py "$OUT" --base "mp_$t" --against "$h" 2>&1 | grep -E "^hyb"
  echo "    UNRAISED:"
  python cls/paired_stats.py "$OUT" --base "mp_$t" --against "hyb_mp_${t}raw" 2>&1 | grep -E "^hyb"
  echo "--- (2) hybrid MP > hybrid uniform, UNRAISED both arms"
  python cls/paired_stats.py "$OUT" --base "hyb_uni_${t}raw" --against "hyb_mp_${t}raw" 2>&1 | grep -E "^hyb"
done

echo ""
echo "########## true SC cycles -- the unraised arms should come in ~(1-f) low"
python cls/iso_check.py "$OUT" --mac-ref uni_32 --cells \
  "$(cd "$OUT" && ls *_trace.json 2>/dev/null | sed 's/_trace\.json//' | paste -sd, -)" 2>&1 | tail -24

echo "=== run_unraised_10k [$HALF] COMPLETE $(date +%T)"
