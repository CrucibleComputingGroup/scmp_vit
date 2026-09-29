#!/usr/bin/env bash
# Turn results/fig1k into the figure + the paired-statistics tables.
#
#   bash cls/finalize.sh
#
# Everything here is CPU-only, so it can run on the login node while the GPU
# keeps working.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
OUT=results/fig1k

echo "########## 1. collect"
python cls/collect_figure.py "$OUT" --out results/figure/summary_1k.csv

echo
echo "########## 2. paired tests vs the matched uniform arm"
# Each MP/hybrid cell is differenced against the uniform SC cell at the SAME
# budget and the SAME qk scope. Marginal Wilson intervals at n=1000 are wider
# than these effects; the paired test is the one that answers the question.
for T in 32 64 96 128; do
  have=$(ls "$OUT"/*_hits.npy 2>/dev/null | grep -cE "/(r|hyb)[a-z_]*_?${T}(_|\.)" || true)
  [ "$have" = "0" ] && continue
  cells=$(ls "$OUT"/*_hits.npy 2>/dev/null | xargs -n1 basename 2>/dev/null \
          | sed 's/_hits.npy//' \
          | grep -E "^(r${T}_|hyb_[a-z]+_${T}$|r${T}$)" | paste -sd, -)
  [ -z "$cells" ] && continue
  echo
  echo "--- budget halved ${T} ---"
  python cls/paired_stats.py "$OUT" --base "uni_${T}" --against "$cells"
done

echo
echo "########## 3. mask-ranking control (ranked vs random 10%)"
for T in 32 64; do
  [ -f "$OUT/hyb_uni_rand_${T}_hits.npy" ] || continue
  python cls/paired_stats.py "$OUT" --base "hyb_uni_rand_${T}" \
    --against "hyb_uni_${T}"
done

echo
echo "########## 4. decomposition: what bought the accuracy"
# The chain uniform per_head -> per_row -> +protection -> +MP, at each budget
# where all four cells exist. This is the session's actual finding; the levels
# figure below is the family comparison.
for T in 32 64; do
  python figs/plot_decomposition.py "$OUT" $T
  for A in uni_head_$T uni_$T uniprot_$T; do
    B=$( [ "$A" = "uni_head_$T" ] && echo "uni_$T" \
         || { [ "$A" = "uni_$T" ] && echo "uniprot_$T" || echo "r${T}_pc"; } )
    [ -f "$OUT/${A}_hits.npy" ] && [ -f "$OUT/${B}_hits.npy" ] && \
      python cls/paired_stats.py "$OUT" --base "$A" --against "$B"
  done
done

echo
echo "########## 5. levels figure"
python figs/plot_figure.py results/figure/summary_1k.csv

echo
echo "=== finalize COMPLETE $(date +%T)"
