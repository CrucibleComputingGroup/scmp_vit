#!/usr/bin/env bash
# Control: are protected channels an MP feature, or a general SC fix?
#
# This study's other big win — qk per_row — turned out NOT to be about MP at
# all: it moved the uniform arm by +3.22 pt on its own. Protected channels
# attack the same failure (an outlier setting the shared per-row scale) on the
# linear side, and they were measured to REPAIR the dispatch metric
# (mlp_fc2 rho -0.105 -> +0.231), which is an MP-specific benefit. So the two
# readings are genuinely different and the experiment separates them:
#
#   uniform SC                      <- baseline (already in Phase A)
#   uniform SC + protected channels <- here; no per-row dispatch anywhere
#   MP + protected channels         <- Phase B (r32_pc)
#
# If uniform+protection captures most of MP+protection's gain, protection is a
# general SC fix and per-row MP is adding little on top. If it captures only
# part, the metric-repair effect is real and MP is doing work.
#
# ISO-COMPUTE. Protecting a fraction p at protect_sl while the rest run at
# base costs p*protect_sl + (1-p)*base. At p=0.0197 and protect_sl=64 a base of
# 32 realizes ~32.6 (2% over), so a base of 31 is also run: it realizes ~31.6,
# i.e. UNDER the uniform comparator, making any win conservative. Both are
# placed at their trace-measured average by the collector.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}
N=${N:-1000}
OUT=results/fig1k
TABLE=${TABLE:-results/mp_calib/s32_pc2.json}

echo "[prot] waiting for earlier queues $(date +%T)"
while pgrep -f "cls/run_all_1k.sh|cls/run_mask_control.sh" > /dev/null; do sleep 30; done
echo "[prot] GPU free $(date +%T)"

[ -f "$TABLE" ] || { echo "[prot] no table at $TABLE"; exit 1; }

for BASE in 32 31; do
  tag=uniprot_$BASE
  [ -f "$OUT/$tag.json" ] && continue
  echo "=== [$tag] base=$BASE protected@64 $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len $BASE \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --protect_from_table "$TABLE" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[sc\] protected|^\[result\]|Error|Traceback|SystemExit"
done
echo "=== run_protect_control COMPLETE $(date +%T)"
