#!/usr/bin/env bash
# Prep for the n=10000 formal run. Everything here is a PREREQUISITE of
# run_final_10k.sh, not a throwaway: the masks and the calibration tables it
# builds are exactly the ones the big run consumes.
#
#   HALF=a bash cls/prep_final.sh     # GPU A: t48 mask, then the eff-96 recipe
#   HALF=b bash cls/prep_final.sh     # GPU B: t128 mask, then the eff-256 helper
#
# WHY A PREP PASS. eff-96 (halved T=48) has never been run. Two things about it
# are unknown and both change what the big script should launch:
#
#   1. Is `PCF` on or off at eff-96? Protection's increment flips sign with
#      budget -- +4.52 at eff-64, +0.15 at eff-128, -0.35 at eff-192 -- and
#      eff-96 sits between the first two. Guessing costs a 33-minute cell at
#      n=10000; measuring costs a 3-minute cell at n=1000.
#   2. Does the t48 ladder hold its budget? 96,72,48,36,24 is new. The
#      calibrator's `expected_flop_avg_stoc_len` has to land on 48.00 or the
#      allocator had nothing to trade (a target sitting on the ladder's top or
#      bottom rung makes the swap search exit at iteration 0 reporting
#      "converged" having moved nothing).
#
# EXCL=10000 THROUGHOUT. The calibration pool must exclude every image the table
# is ever scored on. These tables get deployed at n=10000, so they are
# calibrated with 10000 excluded now, and the n=1000 sanity evals here are a
# strict subset of that exclusion -- the same trick run_all_1k.sh used. Building
# them at EXCL=2048 would force a full recalibration for the big run.
source cls/sc_lib.sh

HALF=${HALF:-a}
export MASK_IMGS=${MASK_IMGS:-128}

if [ "$HALF" = "a" ]; then
  echo "########## PREP A: eff-96  $(hostname) $(date +%T)"

  # 1. mask at t48 -- SC fragility is budget-dependent, so eff-96 needs its own
  #    probe. The t32 and t64 masks share only 2 of their 14 groups.
  build_mask 48

  # 2. uniform reference at eff-96, and the protection decision.
  LADDER= GLK=0 PCF=0 sc_cell uni_48 48
  LADDER= GLK=0 PCF=0    EXCL=10000 mp_cell m48_base 48
  LADDER= GLK=0 PCF=0.02 EXCL=10000 mp_cell m48_pc   48

  echo "########## PREP A: is protection worth it at eff-96? $(date +%T)"
  python cls/paired_stats.py "$OUT" --base uni_48 --against m48_base,m48_pc 2>&1 | tail -6
  echo "--- PC increment (the number that decides PCF for eff-96) ---"
  python cls/paired_stats.py "$OUT" --base m48_base --against m48_pc 2>&1 | tail -4
  python cls/iso_check.py "$OUT" --mac-ref uni_32 --cells uni_48,m48_base,m48_pc 2>&1 | tail -6

  echo "########## PREP A: t48 mask fraction and iso target $(date +%T)"
  python cls/mask_mac_frac.py results/hybrid/mask10_t48.json \
    --ref "$OUT/uni_48_trace.json" --budget 48 | grep '^\[frac\]'

else
  echo "########## PREP B: eff-256  $(hostname) $(date +%T)"

  # 1. mask at t128.
  build_mask 128

  # 2. uniform at the grid ceiling.
  LADDER= GLK=0 PCF=0 sc_cell uni_128 128

  # 3. THE eff-256 ISO PROBLEM. 128/(1-f) is about 143, past the hard ceiling
  #    SC_MAX_STOC_LEN=128, and the kernel CLAMPS rather than errors -- so the
  #    obvious cell would silently run at 128 and spend ~(1-f) of uni_128's
  #    cycles while being plotted as its equal. Fix it from the other side: hold
  #    the hybrid at 128 and pull the REFERENCE down to 128*(1-f), so both arms
  #    spend the same cycles and both stay on the grid.
  F=$(mask_frac results/hybrid/mask10_t128.json "$OUT/uni_128_trace.json")
  TREF=$(python -c "print(round(128*(1-$F)))")
  echo "[prepB] mask10_t128: f=$F -> hybrid stays at 128, reference drops to $TREF"
  LADDER= GLK=0 PCF=0 sc_cell "uni_$TREF" "$TREF"
  LADDER= GLK=0 PCF=0 sc_cell hyb_uni_128 128 --hybrid_config_json results/hybrid/mask10_t128.json

  echo "########## PREP B iso audit -- hyb_uni_128 should match uni_$TREF, not uni_128"
  python cls/iso_check.py "$OUT" --mac-ref uni_32 --cells "uni_128,uni_$TREF,hyb_uni_128" 2>&1 | tail -6
fi

echo "########## PREP $HALF COMPLETE $(date +%T)"
