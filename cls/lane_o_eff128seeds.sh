#!/usr/bin/env bash
# Lane O -- is ordering (3) at eff-128 real? It is the only borderline result
# left outside eff-64.
#
# At n=4000 with a fresh EXCL=4000 table, (3) hybrid MP > MP came out
# +0.62 [+0.00, +1.22] p=0.0567 -- the CI's lower bound is exactly zero. And it
# DISAGREES with the n=1000 measurement of the same recipe (-0.20, p=0.87), so
# the two draws straddle zero. One of them is a fluke and the interval cannot say
# which.
#
# The suspect is calibration draw, not eval noise: the n=1000 and n=4000 cells use
# DIFFERENT tables (EXCL=2048 vs EXCL=4000) as well as different eval sets, and
# ISO_REPORT measured the hybrid-MP arm's calibration sd at 0.36-0.59 pt -- the
# same size as the effect. So: hold the eval set at n=4000 and vary ONLY the
# calibration seed. Three more draws against the same references.
#
# Protocol note: only the hybrid arm's seed varies, against a fixed m64_base_x4k
# reference. That measures the hybrid cell's own calibration spread, which is what
# decides whether +0.62 is a lucky draw. It is the same protocol ISO_REPORT used
# for the eff-64 headline (4 seeds on the hybrid cell), so the numbers are
# comparable to that table.
#
#   srun --jobid=<J> --overlap bash cls/lane_o_eff128seeds.sh
source cls/sc_lib.sh

export N=4000 OUT=results/fig4k
M64=results/hybrid/mask10_t64.json
mkdir -p "$OUT"

echo "########## Lane O: eff-128 ordering (3), calibration-seed replication at n=4000"
echo "########## $(hostname) CUDA=$CUDA_VISIBLE_DEVICES $(date +%T)"

# T=73 is the iso-corrected target, re-derived by mp_hybrid_iso from the trace on
# this exact reference (T=71 came in 2.3% under). Reused directly here because the
# seeds share the recipe; each cell is iso-audited below regardless.
for s in 7 23 41; do
  LADDER= GLK=0 PCF=0 EXCL=4000 CALIB_SEED=$s \
    mp_cell "hyb_mp_64iso_x4k2_s$s" 73 --hybrid_config_json "$M64"
done

echo "########## Lane O iso audit $(date +%T)"
python cls/iso_check.py "$OUT" --mac-ref uni_32 --cells \
  m64_base_x4k,hyb_uni_64iso,hyb_mp_64iso_x4k2,hyb_mp_64iso_x4k2_s7,hyb_mp_64iso_x4k2_s23,hyb_mp_64iso_x4k2_s41 \
  2>&1 | tail -10

echo "########## Lane O: ordering (3) across seeds $(date +%T)"
python cls/paired_stats.py "$OUT" --base m64_base_x4k --against \
  hyb_mp_64iso_x4k2,hyb_mp_64iso_x4k2_s7,hyb_mp_64iso_x4k2_s23,hyb_mp_64iso_x4k2_s41 2>&1 | tail -8

echo "########## Lane O: ordering (2) across seeds $(date +%T)"
python cls/paired_stats.py "$OUT" --base hyb_uni_64iso --against \
  hyb_mp_64iso_x4k2,hyb_mp_64iso_x4k2_s7,hyb_mp_64iso_x4k2_s23,hyb_mp_64iso_x4k2_s41 2>&1 | tail -8

echo "########## Lane O COMPLETE $(date +%T)"
