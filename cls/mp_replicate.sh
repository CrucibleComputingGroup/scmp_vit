#!/usr/bin/env bash
# Replicate one MP config across calibration seeds, to separate a real effect
# from calibration-sample noise.
#
#   bash cls/mp_replicate.sh <BUDGET_HALVED> <LADDER> <METRIC> <SEED...>
#
# Why this exists: at n_calib=128 two independent calibration draws of the SAME
# config moved top-1 by ~1 pt (t32_amax 0.7319 vs t32v2_amax 0.7441). The
# MP-vs-uniform deltas being measured are 0.5-2 pt, i.e. the same size, so a
# single draw cannot decide the question. n_calib is raised to 256 here (the
# size scmp_cnn's notes call for) and each seed is a fresh draw.
set -uo pipefail

T=$1; LADDER=$2; M=$3; shift 3
NCAL=${NCAL:-256}
NEVAL=${NEVAL:-2048}
VIT_SC=/nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc

for S in "$@"; do
  TAG="r${T}_${M}_s${S}"
  echo "### replicate seed=$S -> $TAG"
  CALIB_SEED=$S bash "$VIT_SC/cls/mp_cell.sh" "$TAG" "$T" "$LADDER" "$M" "$T" "$NCAL" "$NEVAL"
done
