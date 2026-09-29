#!/usr/bin/env bash
# ONE-CLICK: the three orderings at eff-64 / eff-128 / eff-192.
#
#   bash cls/sweep_orderings.sh              # n=1000, reuses every cached cell
#   N=4000 bash cls/sweep_orderings.sh       # power run (needs EXCL=4000 tables)
#   BUDGETS="32 96" bash cls/sweep_orderings.sh
#   SEEDS="7 23 41" bash cls/sweep_orderings.sh   # replicate the hybrid-MP arm
#
# Produces, per budget, the four cells the orderings need:
#
#   (1) MP > uniform            MP cell        vs uniform cell
#   (2) hybrid MP > hybrid uni  hybrid MP cell vs hybrid uniform cell
#   (3) hybrid MP > MP          hybrid MP cell vs MP cell
#
# Everything is cache-keyed by tag, so a re-run costs nothing for cells that
# already exist and the script doubles as the provenance record.
#
# ============================================================ THE PARAMETERS
# Each of these was measured, not chosen. Changing one invalidates the cached
# tables, so give a modified cell its OWN tag.
#
# DATAPATH (identical in every cell, uniform and MP alike)
#   --sc_prec 8 --halve 1        halved cycle counts; eff levels = 2 x budget
#   --chunk_d 128                CORRECTNESS. chunk_d=0 on a D_in=4096 MLP gives
#                                top-1 0.0000. Every historical "SC on the MLP
#                                collapses" result was an unchunked run.
#   --sc_ops mlp_fc1,mlp_fc2,qk,av,proj    all five op types, all 24 blocks
#   --qk_granularity per_row     worth +0.59 pt to the UNIFORM arm on its own, so
#                                both arms must use it or MP books the scope gain
#                                a second time
#
# ALLOCATOR
#   OBJECTIVE=sigma              variance-proxy Lagrangian
#   PRICING=macs                 REQUIRED once qk/av are in the table. 'rep'
#                                prices an attention row 16x a linear row at
#                                H=16 and starves attention to the ladder floor:
#                                qk realized 16.6 vs a 32 budget, top-1 0.672.
#   CLW=macs                     cross-layer weight. W_g=uniform pins the cheap
#                                out_proj (8% of MACs) at the ladder ceiling and
#                                starves mlp_fc1 (32%) to 64.8 cycles at t96.
#   METRIC=amax                  --metric-select auto reorders rows WITHIN a
#                                group and cannot fix the cross-operator split;
#                                measured -1.20 pt.
#   GLK=0                        per-group ladders: -1.66 pt as run. See
#                                cls/lane_l_denseladder.sh -- that run restricted
#                                each group to 3 rungs of a dense 8 while the
#                                baseline had 5, so it measured "fewer rungs",
#                                and 94 of 96 groups lost the 64-cycle ceiling
#                                that <1% of MACs use and the heavy tail needs.
#   ESC unset                    escape gate: +1.08 while the row metric is
#                                broken, -2.00 once protection repairs it.
#   MPOPS=all six                qk and av included; they are only 4% of MACs but
#                                excluding them leaves them at uniform.
#   NCAL=128                     calibration-draw sd ~1 pt here vs 0.33 at 256,
#                                which is why SEEDS exists.
#   EXCL >= N                    calibration pool must exclude every eval image.
#
# PROTECTED CHANNELS -- the one parameter that is budget-dependent
#   Increment measured over 4 calibration seeds per cell:
#     eff-64  +4.52   eff-128  +0.15   eff-192  -0.35
#   and PC costs ~4x the calibration variance (sd 0.49/0.38 vs 0.13/0.10). So it
#   is ON at eff-64 and OFF above. CAVEAT, from ISO_REPORT.md: the switch point
#   was read off the same eval set the figure reports, and the rho diagnostic
#   cannot supply it (rho reads -0.09..-0.10 unprotected at EVERY budget, so it
#   would say "enable" everywhere). PCF_ALL=<f> forces one value at all budgets
#   if you want the single-recipe version instead.
#
# HYBRID MASK
#   10% of (op, block) groups -> INT7, ranked by marginal knock-down dLoss at
#   THAT budget (a mask measured at one stream length does not transfer: the t32
#   and t64 masks share only 2 of 14 groups). The ranking is worth +1.50 pt over
#   a per-op-matched random mask on the MP arm (p=0.063).
#   Re-budgeting is not optional: a masked group LEAVES the SC budget instead of
#   shrinking it, so an unadjusted hybrid cell is ~8-12% cheaper than its
#   reference and the comparison is not iso-compute. Target = T/(1-f) with f the
#   MAC-weighted fraction (NOT the group fraction -- attention groups are cheap),
#   then ITERATED against the trace, because T/(1-f) assumes the masked groups
#   sat at the average stream length and an MP allocation makes that false.
#
# HONESTY CAVEAT that must travel with every hybrid number: iso-SC-cycles counts
# the INT work as FREE. It is the optimistic bound. The repo has no energy model;
# the +2.10 at eff-64 is fully eaten only if one INT7 MAC costs 158 SC cycles.
set -uo pipefail
source cls/sc_lib.sh

BUDGETS=${BUDGETS:-"32 64 96"}
SEEDS=${SEEDS:-""}
MASK_IMGS=${MASK_IMGS:-128}
export MASK_IMGS

# Per-budget cell names and the one budget-dependent knob. Names are DATA, not
# derived, because these tags carry cached calibration tables from earlier runs
# and renaming them would silently recalibrate ~4 h of GPU work.
#   T    PCF   uniform  MP         hybrid-uniform   hybrid-MP
ROW_32="32   0.02  uni_32   r32_pc     hyb_uni_32iso    hyb_mppc_32iso"
ROW_64="64   0      uni_64   m64_base   hyb_uni_64iso    hyb_mp_64iso"
ROW_96="96   0      uni_96   m96_base   hyb_uni_96iso    hyb_mp_96iso"

report_lines=()

for T in $BUDGETS; do
  eval "row=\$ROW_$T"
  # shellcheck disable=SC2086
  set -- $row
  T=$1 PCF_B=$2 UNI=$3 MP=$4 HUNI=$5 HMP=$6
  [ -n "${PCF_ALL:-}" ] && PCF_B=$PCF_ALL
  EFF=$((T*2))

  echo ""
  echo "################################################################"
  echo "########## eff-$EFF  (halved T=$T)   PCF=$PCF_B   n=$N   $(date +%T)"
  echo "################################################################"

  # ---- (a) uniform SC: the ordering-(1) comparator, and the unpadded MAC
  #          reference every iso-compute number is computed against.
  LADDER= GLK=0 PCF=0 sc_cell "$UNI" "$T"

  # ---- (b) MP SC
  LADDER= GLK=0 PCF=$PCF_B mp_cell "$MP" "$T"

  # ---- (c) mask at THIS budget
  build_mask "$T"
  MASK=results/hybrid/mask10_t$T.json
  if [ ! -f "$MASK" ]; then
    echo "[sweep] eff-$EFF: no mask -- hybrid arms skipped"
    continue
  fi

  # ---- (d) iso target from the mask's TRUE MAC fraction
  F=$(mask_frac "$MASK" "$OUT/${UNI}_trace.json")
  TISO=$(python -c "print(round($T/(1-$F)))")
  echo "[sweep] eff-$EFF: mask f=$F -> iso target $T -> $TISO"

  # ---- (e) hybrid uniform. For a uniform allocation T/(1-f) is exact by
  #          construction, so no iteration.
  LADDER= GLK=0 PCF=0 sc_cell "$HUNI" "$TISO" --hybrid_config_json "$MASK"

  # ---- (f) hybrid MP, iso-fitted against the unmasked MP cell
  ISO_TAG=$HMP
  eval "$(LADDER= GLK=0 PCF=$PCF_B mp_hybrid_iso "$HMP" "$TISO" "$MASK" "$MP" 2 | tee /dev/stderr | grep '^ISO_TAG=')"
  echo "[sweep] eff-$EFF: hybrid-MP cell that landed iso = $ISO_TAG"

  # ---- (g) optional calibration-seed replication of the hybrid-MP arm, the
  #          cell whose effect is closest to the calibration noise floor.
  for s in $SEEDS; do
    LADDER= GLK=0 PCF=$PCF_B CALIB_SEED=$s \
      mp_cell "${ISO_TAG}_s$s" "$TISO" --hybrid_config_json "$MASK"
  done

  report_lines+=("$EFF|$UNI|$MP|$HUNI|$ISO_TAG")
done

# ============================================================== the report
echo ""
echo "################################################################"
echo "########## ISO AUDIT -- equal true SC cycles or the deltas mean nothing"
echo "################################################################"
for line in "${report_lines[@]}"; do
  IFS='|' read -r EFF UNI MP HUNI HMP <<< "$line"
  python cls/iso_check.py "$OUT" --mac-ref uni_32 --cells "$UNI,$MP,$HUNI,$HMP" 2>&1 | tail -8
done

echo ""
echo "################################################################"
echo "########## THE THREE ORDERINGS (paired: McNemar + bootstrap CI)"
echo "################################################################"
for line in "${report_lines[@]}"; do
  IFS='|' read -r EFF UNI MP HUNI HMP <<< "$line"
  echo ""
  echo "===== eff-$EFF ============================================="
  echo "--- (1) MP > uniform          [$MP vs $UNI]"
  python cls/paired_stats.py "$OUT" --base "$UNI"  --against "$MP"  2>&1 | grep -E "^\S+\s+0\.|top1"
  echo "--- (2) hybrid MP > hybrid uniform   [$HMP vs $HUNI]"
  python cls/paired_stats.py "$OUT" --base "$HUNI" --against "$HMP" 2>&1 | grep -E "^\S+\s+0\.|top1"
  echo "--- (3) hybrid MP > MP        [$HMP vs $MP]"
  python cls/paired_stats.py "$OUT" --base "$MP"   --against "$HMP" 2>&1 | grep -E "^\S+\s+0\.|top1"
done

# Headroom, so a null result is read as saturation rather than as failure. At
# eff-192 uniform SC is already within 0.4 pt of FP and resolving an effect that
# size needs ~25,000 images -- no allocator can show anything there at n=1000.
echo ""
echo "########## HEADROOM (FP - uniform SC): the most any allocator can win"
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
d = Path(sys.argv[1])
fp = json.loads((d / "fp.json").read_text())["top1"] if (d / "fp.json").exists() else None
for t, eff in (("uni_32", 64), ("uni_64", 128), ("uni_96", 192)):
    p = d / f"{t}.json"
    if not p.exists() or fp is None:
        continue
    u = json.loads(p.read_text())["top1"]
    print(f"  eff-{eff:<4d} uniform {u:.4f}  FP {fp:.4f}  headroom {100*(fp-u):+.2f} pt")
PY

echo ""
echo "=== sweep_orderings COMPLETE $(date +%T)"
