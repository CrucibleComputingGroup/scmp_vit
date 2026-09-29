#!/usr/bin/env bash
# Lane L (gl1807) -- two things the reports leave open.
#
# PART 1: is a per-group ladder >= a shared one? The monotonicity argument says
# yes: "every group picks the same rungs" is inside the per-group feasible set,
# so the per-group optimum cannot be worse. The measured -1.66 pt (`s32_glk`
# 0.7407 vs `s32_base` 0.7573) does NOT contradict that, because that pair was
# never a superset comparison:
#
#   s32_base  5 rungs shared,  span 64..16,  96/96 groups reach the 64 ceiling
#   s32_glk   3 rungs per group from a dense 8, mean span 40.7..24.8,
#             0/96 groups' spans cover base's, 94/96 capped below 64, 36 at <=32
#
# Read off the traces, `s32_base` puts 0.79% of its MACs at sl=64 and `s32_glk`
# 0.05%; the per-operator MAC-weighted averages are otherwise within 0.6 cycles
# of each other (fc1 23.81 vs 23.54, fc2 42.29 vs 42.90). So the entire -1.66
# came from removing a ceiling that <1% of MACs used -- the same heavy tail that
# makes 2% protected channels worth +4.5 pt. It measures "3 rungs < 5 rungs",
# not "per-group < shared".
#
# The clean test: sparse {64,48,32,24,16} is a STRICT SUBSET of dense
# {64,56,48,40,32,24,20,16}. Nothing in the entire study has ever run the dense
# ladder unrestricted -- all 111 calibrated cells are 5 rungs / GLK=0 except the
# two GLK=3 ones. Three cells settle it, all on the shipped recipe (CLW=macs,
# PCF=0.02, qk per_row, PRICING=macs) so they are paired against r32_pc = 0.8280:
#
#   r32_pcD     dense 8, GLK=0   TRUE SUPERSET of r32_pc. If this loses, the
#                                monotonicity argument fails on the objective,
#                                not on the configuration -- and the cause has
#                                to be that the sigma-surrogate is not top-1
#                                (rho(metric, benefit) is only +0.13..+0.40).
#   r32_pcD_g5  dense 8, GLK=5   ISO-RUNG-COUNT vs the 5-rung shared ladder.
#                                This is the comparison the >= argument actually
#                                predicts, and it has never been run.
#   r32_pcD_g3  dense 8, GLK=3   s32_glk's setting, but with the FIXED endpoint
#                                rule (_select_group_ladder now pins both ends of
#                                the occupied range) and the modern recipe. The
#                                fix was unit-checked on synthetic data and never
#                                evaluated on the model.
#
# PART 2: ordering (1) MP > uniform at eff-128 is listed as "one of four cells
# reaches p<0.05" -- it is +1.12 pt over 4 seeds (sd 0.13, 4/4 positive) against
# a paired CI half-width of 1.40 pt at n=1000. That is a POWER problem, not an
# effect problem: n=4000 takes the half-width to ~0.70 and settles it. Fresh
# tables at EXCL=4000, or the eval set sits inside the calibration pool.
#
#   bash cls/lane_l_denseladder.sh
source cls/sc_lib.sh

DENSE=64,56,48,40,32,24,20,16

echo "########## Lane L  $(hostname) $(date +%T)"
echo "########## Part 1: dense-ladder / per-group, n=$N"

# All three carry the shipped eff-64 recipe so the only difference from r32_pc
# is the ladder and its per-group restriction.
LADDER=$DENSE PCF=0.02 GLK=0 mp_cell r32_pcD    32
LADDER=$DENSE PCF=0.02 GLK=5 mp_cell r32_pcD_g5 32
LADDER=$DENSE PCF=0.02 GLK=3 mp_cell r32_pcD_g3 32

echo "########## Part 1 iso audit $(date +%T)"
python cls/iso_check.py "$OUT" --mac-ref uni_32 \
  --cells uni_32,r32_pc,r32_pcD,r32_pcD_g5,r32_pcD_g3 2>&1 | tail -12

# What each group actually kept, so a loss can be attributed to the ceiling
# rather than to "per-group" in the abstract.
for t in r32_pcD_g5 r32_pcD_g3; do
  [ -f "results/mp_calib/$t.json" ] || continue
  python - "$t" <<'PY'
import json, sys, collections
t = sys.argv[1]
j = json.load(open(f"results/mp_calib/{t}.json"))
gl = j.get("group_ladders") or {}
if not gl:
    print(f"[L] {t}: no group_ladders in table"); raise SystemExit
tops = collections.Counter()
by_op = collections.defaultdict(collections.Counter)
for k, v in gl.items():          # keys are "<op>:l<block>"
    lv = v if isinstance(v, list) else (v.get("levels") or v.get("stoc_len_levels"))
    tops[max(lv)] += 1
    by_op[k.split(":")[0]][tuple(lv)] += 1
print(f"[L] {t}: {len(gl)} groups, ceiling reached -> {dict(sorted(tops.items(), reverse=True))}")
for op in sorted(by_op):
    print(f"[L]   {op:10s} {by_op[op].most_common(3)}")
PY
done

echo "########## Part 2: eff-128 ordering (1) at n=4000 $(date +%T)"
# LADDER= clears the dense override: bash keeps a prefix assignment to a FUNCTION
# in effect after the call returns, so without this m64_base_x4k would silently
# calibrate on the t32 dense ladder.
mkdir -p results/fig4k
LADDER= N=4000 OUT=results/fig4k GLK=0 PCF=0 sc_cell uni_64 64
LADDER= N=4000 OUT=results/fig4k GLK=0 PCF=0 EXCL=4000 mp_cell m64_base_x4k 64

echo "########## Lane L COMPLETE $(date +%T)"
