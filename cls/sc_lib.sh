#!/usr/bin/env bash
# Shared cell primitives for the ordering study. Source, do not execute.
#
#   source cls/sc_lib.sh
#
# Everything here encodes a parameter that was MEASURED to matter. The comments
# say which, because most of these look like arbitrary defaults and are not:
#
#   --chunk_d 128     CORRECTNESS, not a cache knob. chunk_d=0 on a D_in=4096
#                     MLP gives top-1 0.0000 (CLAUDE.md); every historical
#                     "SC on MLP collapses" result was an unchunked run.
#   --qk_granularity  per_row moves the UNIFORM arm by +0.59 pt on its own, so a
#                     per_row MP cell differenced against a per_head uniform cell
#                     books the scope gain twice. Every cell here is per_row.
#   PRICING=macs      REQUIRED once qk/av are in the MP table. The default 'rep'
#                     pricing charges an attention row B*H*N vs a linear's B*N,
#                     i.e. 16x at H=16, and the allocator starves attention to
#                     the ladder floor: qk realized 16.6 against a 32 budget,
#                     top-1 0.672 vs uniform 0.783.
#   CLW=macs          Cross-layer weight. With W_g=uniform the cheap out_proj
#                     (8% of MACs) pins at the ladder ceiling while mlp_fc1
#                     (32%) starves. Fixing it is worth ~0 top-1 at the mild
#                     budgets but it makes ONE recipe correct at all three,
#                     which the figure needs to not be per-budget cherry-picking.
#   EXCL >= eval n    The calibration pool must exclude every image the table is
#                     scored on. Default 2048 covers n<=2048; an n=4000 eval
#                     needs EXCL=4000 or the thresholds saw the eval set.
#   GLK=0 / ESC unset Per-group ladders (-1.66) and the escape gate (sign
#                     reversal +1.08 -> -2.00 once the row metric is repaired)
#                     both lose. Deliberately off.
set -uo pipefail

VIT_SC=${VIT_SC:-/nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc}
cd "$VIT_SC" || exit 1
export TORCH_HOME=$VIT_SC/.torch_cache
DATA=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

N=${N:-1000}
NCAL=${NCAL:-128}
OUT=${OUT:-results/fig1k}
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av
mkdir -p "$OUT" results/logs results/hybrid results/mp_calib

# Ladders, by halved-cycle budget. The top rung is the budget reference, and the
# ladder MUST STRADDLE the target -- a target on the top or bottom rung leaves
# the allocator nothing to trade and the swap search exits at iteration 0
# reporting "converged" having moved nothing.
#
# SHAPE. t32 and t64 are both target x {2, 1.5, 1, 0.75, 0.5}; t48 follows them
# so eff-96 is not a differently-shaped point on the same curve. t96 cannot --
# 2x96=192 is past the hard grid ceiling 2**(sc_prec-1)=128 -- so it tops out at
# 128 and is the one budget whose ladder shape differs, by necessity.
LADDER_32=64,48,32,24,16
LADDER_48=96,72,48,36,24
LADDER_64=128,96,64,48,32
LADDER_96=128,112,96,80,64,48

ladder_for () {   # halved budget -> ladder that straddles it
  # Bands are set so each budget's ISO-HYBRID target lands in the same band as
  # its base target: 32->35, 48->~54, 64->71/73, 96->107.
  local t=$1
  if   [ "$t" -le 39  ]; then echo $LADDER_32
  elif [ "$t" -le 55  ]; then echo $LADDER_48
  elif [ "$t" -le 79  ]; then echo $LADDER_64
  else                        echo $LADDER_96
  fi
}

# Hard ceiling on any stream length: under --halve a bipolar magnitude spans
# sc_prec-1 bits, so the kernel silently CLAMPS stoc_len to 2**(sc_prec-1)=128
# (model/sc_vit.py::_resolve_stoc_len). A hybrid cell whose 1/(1-f) target
# exceeds this does not run at the requested length and is NOT iso with its
# reference -- it just spends ~(1-f) of the reference's cycles, silently. Above
# this ceiling, hold the hybrid at 128 and pull the REFERENCE down to
# 128*(1-f) instead.
SC_MAX_STOC_LEN=128

_filter () { grep -viE "futurewarning|pynvml|warnings.warn"; }

# ---------------------------------------------------------------- uniform SC
sc_cell () {   # tag, stoc_len, [extra eval args...]
  local tag=$1 sl=$2; shift 2
  [ -f "$OUT/$tag.json" ] && { echo "[lib] $tag cached"; return 0; }
  echo "=== [$tag] uniform sl=$sl n=$N $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$sl" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --data_root "$DATA" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" "$@" 2>&1 | _filter \
    | grep -E "^\[hybrid\]|^\[result\]|^\[eval\] per-image|Error|Traceback|SystemExit"
}

# ---------------------------------------------------------------- MP SC
# PCF / CALIB_SEED / EXCL / GLK are read from the environment so a caller can
# vary one knob per cell. A cell using a non-default combination needs its OWN
# tag: the calibration cache is keyed by tag alone.
mp_cell () {   # tag, budget_halved, [extra eval args...]
  local tag=$1 t=$2; shift 2
  # LADDER overrides the straddle-picked default -- used only by the dense-ladder
  # controls, which need a candidate set the shared ladder is a strict SUBSET of.
  local ladder; ladder=${LADDER:-$(ladder_for "$t")}
  if [ ! -f "results/mp_calib/${tag}_wrapper.json" ]; then
    echo "=== [$tag] calibrating T=$t ladder=$ladder pcf=${PCF:-0} glk=${GLK:-0} seed=${CALIB_SEED:-1234} excl=${EXCL:-2048} $(date +%T)"
    # Every knob passed EXPLICITLY: mp_cell2.sh is a separate process and reads
    # them from its environment, so relying on the caller having exported them
    # is how a cell silently gets the wrong recipe under a cached tag.
    CLW=macs PRICING=macs QKG=per_row MPOPS=$ALLOPS \
    PCF=${PCF:-0} GLK=${GLK:-0} CALIB_SEED=${CALIB_SEED:-1234} EXCL=${EXCL:-2048} \
      bash cls/mp_cell2.sh "$tag" "$t" "$ladder" "$NCAL" 16 2>&1 \
      | grep -E "^\[calib\] (expected|blocks|protected)|^\[cell\]|rho=|Error|Traceback|SystemExit|CUDA" || true
  fi
  if [ ! -f "results/mp_calib/${tag}_wrapper.json" ]; then
    echo "[lib] $tag: NO TABLE -- calibration failed, see above"; return 1
  fi
  [ -f "$OUT/$tag.json" ] && { echo "[lib] $tag eval cached"; return 0; }
  echo "=== [$tag] MP eval T=$t n=$N $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$t" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table "results/mp_calib/${tag}_wrapper.json" \
    --adaptive_mp_ops "$ALLOPS" \
    --data_root "$DATA" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" "$@" 2>&1 | _filter \
    | grep -E "^\[hybrid\]|^\[sc\] adaptive|^\[result\]|^\[eval\] per-image|Error|Traceback|SystemExit"
}

# ---------------------------------------------------------------- hybrid mask
build_mask () {   # budget_halved  -> results/hybrid/mask{05,10,20}_t<T>.json
  local t=$1
  [ -f "results/hybrid/mask10_t$t.json" ] && { echo "[lib] mask t$t cached"; return 0; }
  echo "=== [mask t$t] sensitivity probe, 145 forwards x $MASK_IMGS imgs $(date +%T)"
  python -u cls/build_hybrid_mask.py --stoc_len "$t" --tag "t$t" \
    --frac 0.05,0.10,0.20 --int_bits 7 --int_sym 0 --qk_granularity per_row \
    --n_images "${MASK_IMGS:-128}" --batch_size 16 --data_root "$DATA" 2>&1 | _filter \
    | grep -E "^\[sens\]|^\[mask\]|Error|Traceback"
}

# MAC-weighted fraction the mask removes, and the 1/(1-f) raised target.
mask_frac () {   # mask_path, uniform_ref_trace  -> prints f
  python cls/mask_mac_frac.py "$1" --ref "$2" | tail -1 | cut -d= -f2
}

# Total SC cycles a cell spent, on TRUE (unpadded) MACs, as a % of a reference
# cell's. 100.0 = iso-compute. See ISO_REPORT.md sections 1 and 3.
iso_pct () {   # ref_tag, cell_tag  -> prints percent
  python - "$OUT" "$1" "$2" <<'PY'
import sys
sys.path.insert(0, "cls")
from pathlib import Path
from iso_check import true_macs, load
d, ref_tag, cell_tag = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ref_macs = true_macs(d / "uni_32_trace.json")   # architecture-determined
r = load(d, ref_tag, ref_macs)
c = load(d, cell_tag, ref_macs)
print(f"{100.0 * c['true_cycles'] / r['true_cycles']:.2f}" if r and c else "nan")
PY
}

# An MP + mask cell placed at the stream length that makes it ISO-COMPUTE with
# its unmasked MP reference, by MEASUREMENT rather than by the closed form.
#
# `1/(1-f)` assumes the masked groups were allocated at the average stream
# length. That is exact for a uniform arm and FALSE for an MP one: at t64 the
# mask happened to remove groups whose MP allocation was above average, so
# dropping them pulled the mean down further than the raise pushed it up, and
# the cell came in 3.1% UNDER budget (71 had to become 73). At t32 the same rule
# landed at +0.5% by luck of which groups the mask took. So iterate against the
# trace: run, measure, correct, re-run. Two passes have always sufficed.
#
# Prints the tag of the cell that landed iso (or the last one tried).
mp_hybrid_iso () {   # tag_stem, budget_halved, mask, unmasked_ref_tag, max_iters
  local stem=$1 t=$2 mask=$3 ref=$4 iters=${5:-2}
  local tag=$stem target=$t i=1 pct off
  while : ; do
    mp_cell "$tag" "$target" --hybrid_config_json "$mask" || return 1
    pct=$(iso_pct "$ref" "$tag")
    echo "[lib] $tag @T=$target: true SC cycles = ${pct}% of $ref"
    off=$(python -c "print(1 if abs($pct-100.0)>${ISO_TOL:-1.5} else 0)" 2>/dev/null || echo 0)
    [ "$off" = "0" ] && break
    i=$((i+1)); [ "$i" -gt "$iters" ] && { echo "[lib] $stem: still off budget after $iters passes"; break; }
    target=$(python -c "print(round($target*100.0/$pct))")
    tag="${stem}$i"
    echo "[lib] iterating -> $tag @T=$target"
  done
  echo "ISO_TAG=$tag"
}
