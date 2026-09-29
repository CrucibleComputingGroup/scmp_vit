#!/usr/bin/env bash
# Lane M (gl1805) -- eff-128 orderings (2) and (3) at n=4000, plus the one
# control LADDER_REPORT.md flags as missing.
#
# WHY. Ordering (1) at eff-128 was +1.30 (p=0.060) at n=1000 and **+0.50
# (p=0.198) at n=4000** on an independent draw with a fresh EXCL=4000 table. The
# effect SHRANK under more power. The 4-seed replication in ISO_REPORT.md
# (+1.12, sd 0.13, 4/4 positive) does not protect against this: all four seeds
# scored the same 1000 images, so it controls calibration-sample noise and not
# eval-subset bias. Going to n=4000 moved uniform +0.5 and MP -0.3, both toward
# the null.
#
# So the "only eff-64 survives" conclusion currently rests on a MEASURED null for
# (1) at eff-128 but only INFERRED nulls for (2) and (3) there (+0.80 p=0.26 and
# +0.70 p=0.17, both at n=1000, both inside the range that just proved
# untrustworthy). This lane measures them.
#
# Cells (references uni_64 and m64_base_x4k already exist in results/fig4k):
#   hyb_uni_64iso       uniform + mask10_t64 at the closed-form iso target
#   hyb_mp_64iso_x4k    MP + mask, iso-fitted against m64_base_x4k, EXCL=4000
#
# PHASE 2 closes the confound LADDER_REPORT.md section 4 records as open: the
# -1.66 pt per-group penalty became a tie (-0.70, p=0.48) but TWO things changed
# at once -- the endpoint rule AND protected channels. The trace argument says
# protection is the operative one (it supplies a 64-cycle path for the outlier
# channels regardless of the group's ladder ceiling), which predicts that with
# PCF=0 the per-group cell should LOSE again. r32_D_g3_nopc tests exactly that.
#
#   bash cls/lane_m_eff128power.sh
source cls/sc_lib.sh

echo "########## Lane M  $(hostname) $(date +%T)"
echo "########## Phase 1: eff-128 orderings (2) and (3) at n=4000"

M64=results/hybrid/mask10_t64.json
export N=4000 OUT=results/fig4k
mkdir -p "$OUT"

# f from the t64 mask, against the n=4000 uniform reference this lane pairs with.
F=$(mask_frac "$M64" "$OUT/uni_64_trace.json")
TISO=$(python -c "print(round(64/(1-$F)))")
echo "[M] mask10_t64: f=$F -> iso target 64 -> $TISO"

# Uniform arm: the closed form is exact for a uniform allocation.
LADDER= GLK=0 PCF=0 sc_cell hyb_uni_64iso "$TISO" --hybrid_config_json "$M64"

# MP arm: iso-fitted by measurement against the n=4000 unmasked MP cell. Note the
# n=1000 study needed 71 -> 73 for its PCF=0.02 arm; this is a PCF=0 arm against
# a different reference, so the correction is re-derived rather than reused.
LADDER= GLK=0 PCF=0 EXCL=4000 \
  mp_hybrid_iso hyb_mp_64iso_x4k "$TISO" "$M64" m64_base_x4k 2

echo "########## Phase 1 iso audit $(date +%T)"
python cls/iso_check.py "$OUT" --mac-ref uni_32 \
  --cells uni_64,m64_base_x4k,hyb_uni_64iso,hyb_mp_64iso_x4k,hyb_mp_64iso_x4k2 2>&1 | tail -10

echo "########## Phase 2: does the per-group tie survive PCF=0? $(date +%T)"
export N=1000 OUT=results/fig1k
DENSE=64,56,48,40,32,24,20,16
# Reference for this pair is r32_base-like: MP alone, CLW=macs, no protection, at
# the shared sparse ladder. m32_base does not exist yet, so build it too -- a
# per-group cell can only be differenced against a same-recipe shared-ladder one.
LADDER= GLK=0 PCF=0 mp_cell m32_base 32
LADDER=$DENSE GLK=3 PCF=0 mp_cell m32_D_g3_nopc 32

echo "########## Phase 2 iso audit $(date +%T)"
python cls/iso_check.py "$OUT" --mac-ref uni_32 \
  --cells uni_32,m32_base,m32_D_g3_nopc,r32_pc,r32_pcD_g3 2>&1 | tail -8

echo "########## Lane M COMPLETE $(date +%T)"
