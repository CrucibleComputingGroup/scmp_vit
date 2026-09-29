#!/usr/bin/env bash
# Lane K (gl1805) -- the eff-192 hybrid arm. THE hole in the 3x3 grid.
#
# ISO_REPORT.md section 5 reports orderings (2) hyb MP > hyb uniform and
# (3) hyb MP > MP as "no mask built" at eff-192, because run_all_1k.sh Phase 0
# only probed t32 and t64: the sensitivity probe is 145 forwards and measured
# 77 min per budget at 256 images, so t96 was cut on cost with the reasoning
# that "t96 is the mildest, where a mask has least to do". That left 2 of the 9
# ordering cells nonexistent rather than measured-and-null, which is not the
# same thing and cannot be plotted.
#
# EXPECTATION, stated before running so it cannot be fitted afterwards: eff-192
# has 0.4 pt of headroom against FP (uniform SC 0.8500 vs FP 0.8540), so
# resolving a real effect there needs ~25,000 images. At n=1000 these cells can
# only establish that the ordering is NOT INVERTED. A tie is the expected and
# honest outcome; the point is to replace a hole with a measurement.
#
# Two MP-hybrid arms, because protection's sign flips with budget (+4.52 at
# eff-64, +0.15 at eff-128, -0.35 at eff-192) and nothing says how it interacts
# with a mask that has already removed the fragile groups:
#   hyb_mp_96iso    MP alone + 10% INT7   <- the eff-192 recipe of ISO_REPORT
#   hyb_mppc_96iso  MP + 2% PC + 10% INT7 <- the recipe the other two budgets use
#
#   bash cls/lane_k_eff192hybrid.sh
source cls/sc_lib.sh

MASK_IMGS=${MASK_IMGS:-128}
M96=results/hybrid/mask10_t96.json
REF_TRACE=$OUT/uni_96_trace.json

echo "########## Lane K: eff-192 hybrid  $(hostname) $(date +%T)"

# ------------------------------------------------------------------ 1. mask
# A mask measured at one stream length does not transfer to another -- SC
# fragility is budget-dependent, and the t32 and t64 masks share only 2 of 14
# groups. So t96 needs its own probe.
build_mask 96
[ -f "$M96" ] || { echo "[K] FATAL: no mask10_t96"; exit 1; }

# ------------------------------------------------------------------ 2. target
# f is the MAC-weighted fraction, NOT the group fraction: attention groups are
# cheap, so 10% of groups was 7.9% of MACs at t32 and 9.8% at t64.
F=$(mask_frac "$M96" "$REF_TRACE")
T=$(python -c "print(round(96/(1-$F)))")
echo "[K] mask10_t96: f=$F  ->  1/(1-f) target = $T (from 96)"
python cls/mask_mac_frac.py "$M96" --ref "$REF_TRACE" --budget 96 | grep '^\[frac\]'

# ------------------------------------------------------------------ 3. cells
# Uniform arm first: for a uniform allocation the 1/(1-f) raise is EXACT (it
# assumes the masked groups sat at the average stream length, which is true by
# construction), so this arm needs no iteration.
sc_cell hyb_uni_96iso "$T" --hybrid_config_json "$M96"

# MP arms. Here 1/(1-f) is only an estimate: the rule assumes the masked groups
# were allocated at the average, and an MP allocation makes that false. At t64
# it undershot by 3.1% and had to be iterated 71 -> 73 (ISO_REPORT section 5).
# So: run at the closed-form target, measure true cycles off the trace against
# the unmasked reference, and re-run once at the corrected target if it is off
# by more than 1.5%.
fit_mp_hybrid () {   # tag_stem, unmasked_ref_tag, pcf
  local stem=$1 ref=$2 pcf=$3
  PCF=$pcf mp_cell "${stem}" "$T" --hybrid_config_json "$M96"
  [ -f "$OUT/${stem}.json" ] || { echo "[K] $stem failed"; return 1; }
  local pct; pct=$(iso_pct "$ref" "${stem}")
  echo "[K] $stem: true SC cycles = ${pct}% of $ref"
  local off; off=$(python -c "print(1 if abs($pct-100.0)>1.5 else 0)")
  if [ "$off" = "1" ]; then
    local T2; T2=$(python -c "print(round($T*100.0/$pct))")
    echo "[K] $stem is off budget -> iterating target $T -> $T2 as ${stem}2"
    PCF=$pcf mp_cell "${stem}2" "$T2" --hybrid_config_json "$M96"
    echo "[K] ${stem}2: true SC cycles = $(iso_pct "$ref" "${stem}2")% of $ref"
  fi
}

# Unmasked references at eff-192 already exist: m96_base (MP alone, CLW=macs)
# and m96_pc (MP + 2% PC, CLW=macs).
fit_mp_hybrid hyb_mp_96iso   m96_base 0
fit_mp_hybrid hyb_mppc_96iso m96_pc   0.02

# ------------------------------------------------------------------ 4. audit
echo "########## Lane K iso audit $(date +%T)"
python cls/iso_check.py "$OUT" --mac-ref uni_32 \
  --cells uni_96,m96_base,m96_pc,hyb_uni_96iso,hyb_mp_96iso,hyb_mp_96iso2,hyb_mppc_96iso,hyb_mppc_96iso2 \
  2>&1 | tail -20

echo "########## Lane K COMPLETE $(date +%T)"
