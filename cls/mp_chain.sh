#!/usr/bin/env bash
# Run the three MP budget points for one dispatch-metric setting, in sequence.
#
#   bash cls/mp_chain.sh <METRIC> [SUFFIX] [N_CALIB] [N_EVAL]
#
# Ladders straddle their target (a target on the ladder's top or bottom rung
# leaves the allocator no room to trade), and each ladder's TOP is the budget
# reference, so ratio = target / top.
set -uo pipefail

M=${1:-amax}
SFX=${2:-}
NCAL=${3:-128}
NEVAL=${4:-2048}
VIT_SC=/nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc

#      target  ladder                 (effective levels = 2x target)
run () { bash "$VIT_SC/cls/mp_cell.sh" "t$1${SFX}_$M" "$1" "$2" "$M" "$1" "$NCAL" "$NEVAL"; }

run 32 64,48,32,24,16
run 64 128,96,64,48,32
run 96 128,96,64,48
