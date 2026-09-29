#!/usr/bin/env bash
# Every table ISO_REPORT.md cites, regenerated from results/fig1k in one pass.
# CPU-only, so it runs on the login node while the GPUs keep working.
#
#   bash cls/finalize_iso.sh
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
OUT=results/fig1k

echo "################ 1. compute accounting (true vs padded MACs)"
python cls/iso_check.py $OUT --cells \
uni_32,r32_pc,hyb_uni_32iso,hyb_mppc_32iso,hyb_mppc_32,hyb_uni_32,hyb_mppc_rand_32iso,hyb_uni_rand_32iso

echo
echo "################ 2. iso verification for every claimed comparison"
python cls/iso_check.py $OUT --pairs \
uni_32:r32_pc,uni_32:hyb_uni_32iso,r32_pc:hyb_mppc_32iso,r32_pc:hyb_mppc_32,r32_pc:hyb_mppc_rand_32iso,uni_32:hyb_uni_rand_32iso

echo
echo "################ 3. THE THREE ORDERINGS at eff-64"
echo "--- (1) MP > uniform, and the full iso ladder"
python cls/paired_stats.py $OUT --base uni_32 \
  --against r32_base,hyb_uni_32iso,r32_pc,hyb_mppc_32iso
echo
echo "--- (3) hybrid MP > MP   [iso vs the old non-iso cell]"
python cls/paired_stats.py $OUT --base r32_pc \
  --against hyb_mppc_32iso,hyb_mppc_32
echo
echo "--- (2) hybrid MP > hybrid uniform"
python cls/paired_stats.py $OUT --base hyb_uni_32iso --against hyb_mppc_32iso
echo
echo "--- vs the lossless ceiling"
python cls/paired_stats.py $OUT --base fp --against hyb_mppc_32iso,int6_asymm,r32_pc,uni_32

echo
echo "################ 4. CONTROL: does the mask RANKING matter, at iso?"
echo "(ranked vs per-op-matched random, identical MAC fraction 0.0792, same target 35)"
python cls/paired_stats.py $OUT --base hyb_uni_rand_32iso --against hyb_uni_32iso
python cls/paired_stats.py $OUT --base hyb_mppc_rand_32iso --against hyb_mppc_32iso
echo "--- and does a RANDOM mask still beat no mask?"
python cls/paired_stats.py $OUT --base r32_pc --against hyb_mppc_rand_32iso

echo
echo "################ 5. CONTROL: level-matched INT6 (SC arm carries 70 levels)"
python cls/paired_stats.py $OUT --base r32_pc --against hyb_mppc_32iso_i6,hyb_mppc_32iso 2>/dev/null \
  || echo "(lane G not finished)"

echo
echo "################ 6. DOSE at iso: 5 / 10 / 20% of groups on INT7"
python cls/paired_stats.py $OUT --base uni_32 \
  --against hyb_uni_32iso_d05,hyb_uni_32iso,hyb_uni_32iso_d20
python cls/paired_stats.py $OUT --base r32_pc \
  --against hyb_mppc_32iso_d05,hyb_mppc_32iso,hyb_mppc_32iso_d20 2>/dev/null \
  || echo "(MP dose arm not finished)"

echo
echo "################ 7. calibration-seed replication of the headline"
python cls/paired_stats.py $OUT --base r32_pc \
  --against hyb_mppc_32iso,hyb_mppc_32iso_s7,hyb_mppc_32iso_s23,hyb_mppc_32iso_s41

echo
echo "################ 8. mild budgets: did CLW=macs change anything?"
python cls/paired_stats.py $OUT --base uni_64 --against r64_base,m64_base,r64_pc,m64_pc,hyb_uni_64iso,hyb_mppc_64iso
python cls/paired_stats.py $OUT --base uni_96 --against r96_base,m96_base,r96_pc,m96_pc

echo
echo "################ 9. figure CSV (true-MAC x axis, one recipe per budget)"
python cls/collect_figure.py $OUT --out results/figure/summary_fig.csv --only \
fp,int4_asymm,int6_asymm,int7_asymm,int8_asymm,uni_32,uni_64,uni_96,uni_128,r32_pc,m64_pc,m96_pc,hyb_uni_32iso,hyb_uni_64iso,hyb_mppc_32iso,hyb_mppc_64iso \
  | tail -25
python figs/plot_figure.py results/figure/summary_fig.csv
