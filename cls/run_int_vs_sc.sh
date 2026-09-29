#!/usr/bin/env bash
# INT fixed-point vs uniform SC at matched precision — scmp_cnn-style figure.
#
# Convention (scmp_llm / scmp_cnn "halved" convention):
#   halve ON, sc_prec=8 -> constant 128-level grid; the STREAM LENGTH is the
#   compute knob. A length-L halved bipolar stream carries 2L levels of
#   resolution (wu-hpca2022 sign-magnitude identity), so:
#
#     halved stream | effective levels | INT equivalent
#            32     |        64        | int6
#            64     |       128        | int7
#            96     |       192        | (none — SC-only point)
#           128     |       256        | int8   <- grid ceiling
#
#   INT also gets int4 (16 levels) to extend the fixed-point curve below SC.
#
# No hybrid / no skipping: every run is uniform over all 5 op types × 24 blocks.
#
#   bash cls/run_int_vs_sc.sh [N_IMAGES] [TAG]
set -uo pipefail

N=${1:-2048}
TAG=${2:-int_vs_sc}
VIT_SC=/nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc
OUT=$VIT_SC/results/$TAG
LOGS=$OUT/logs
mkdir -p "$LOGS"

export TORCH_HOME=$VIT_SC/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}
OPS=mlp_fc1,mlp_fc2,qk,av,proj

cd "$VIT_SC" || exit 1

run () {
  local name=$1; shift
  if [ -f "$OUT/$name.json" ]; then echo "=== [$name] cached ==="; return; fi
  echo "=== [$name] $(date +%T) ==="
  python -u cls/eval.py --data_root "$D" --max_images "$N" \
    --batch_size 16 --workers 4 --seed 0 \
    --out_json "$OUT/$name.json" "$@" 2>&1 \
    | grep -viE "futurewarning|import pynvml" \
    | tee "$LOGS/$name.log" | grep -E "^\[(int|sc|result)\]"
  echo
}

echo "### INT vs uniform SC | N=$N | $(hostname) | $(date) ###"
echo

# FP ceiling
run fp --mode fp

# INT fixed-point, symmetric (matches bipolar SC's sign-magnitude grid)
for B in 8 7 6 4; do
  run "int${B}_symm" --mode int --int_bits "$B" --int_sym 1 --sc_ops "$OPS"
done

# INT asymmetric (scmp_llm reports asymm wins below W6A6)
for B in 8 7 6 4; do
  run "int${B}_asymm" --mode int --int_bits "$B" --int_sym 0 --sc_ops "$OPS"
done

# Uniform SC, halved bipolar, sc_prec=8 (grid fixed at 128 levels)
for L in 128 96 64 32; do
  run "sc_halved${L}" --mode sc --sc_prec 8 --halve 1 --stoc_len "$L" \
      --chunk_d 128 --sc_ops "$OPS"
done

echo "### done $(date) ###"
OUT_DIR=$OUT python cls/collect_int_vs_sc.py
