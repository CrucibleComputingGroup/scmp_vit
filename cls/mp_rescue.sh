#!/usr/bin/env bash
# Search for an MP config that satisfies the orderings at one budget -- WITHOUT
# touching the reported eval set.
#
#   bash cls/mp_rescue.sh <T> [candidates...]
#   bash cls/mp_rescue.sh 48
#
# ===================================================== THE INTEGRITY PROBLEM
# "Try configs until the ordering holds" is tuning on the report set unless the
# search happens somewhere else. And n=1000 is NOT somewhere else: the eval
# subset is `shuffle(seed)[:N]`, so the first 1000 images are a strict SUBSET of
# the first 10000. Searching there and reporting at n=10000 leaks the search
# into the reported number -- the same defect ISO_REPORT.md already flags about
# the protection switch point ("the threshold was read off the same eval set the
# figure reports").
#
# So this script scores candidates on `--skip_images 10000 --max_images 2000`,
# i.e. `shuffle(0)[10000:12000]`, which is disjoint from the report set by
# construction (verified: 0 images in common). The winner is then deployed ONCE
# at n=10000 and that is the number that gets reported. The search numbers are
# diagnostics and must never appear in a results table.
#
# Residual contamination, stated rather than hidden: the calibration pool for
# EXCL=10000 tables is indices 10000..50000, so the ~128 calibration images can
# fall inside the search window -- about 6 of 2000 in expectation, 0.3%. That is
# immaterial for RANKING candidates and it never reaches the reported cell.
#
# ==================================================== WHEN NOT TO RESCUE
# A null is not a violation. At eff-192 uniform SC is within 0.4 pt of FP and
# ordering (2)'s comparator is AT FP -- no allocator can win there, and searching
# configs against a saturated budget is guaranteed to find noise that flatters
# whichever cell was drawn high. This script therefore REFUSES to search when the
# headroom (FP minus uniform SC) is below MIN_HEADROOM, and says so.
set -uo pipefail
source cls/sc_lib.sh

T=${1:?usage: mp_rescue.sh <T_halved> [candidate...]}
shift || true

SEARCH_SKIP=${SEARCH_SKIP:-10000}
SEARCH_N=${SEARCH_N:-2000}
MIN_HEADROOM=${MIN_HEADROOM:-1.5}
REPORT_OUT=${REPORT_OUT:-results/final10k}
export OUT=${SEARCH_OUT:-results/search}
export N=$SEARCH_N
export EXCL=10000
mkdir -p "$OUT"

# Candidates, cheapest and most-motivated first. Each is ONE knob off the shipped
# recipe, so a win is attributable:
#   pcf0/pcf2/pcf5   protection is the one budget-dependent parameter and its
#                    increment flips sign with budget (+4.52 / +0.15 / -0.35)
#   ncal256          calibration-draw sd is ~1 pt at NCAL=128 vs 0.33 at 256; a
#                    borderline ordering can be a draw rather than a config
#   clwuni           the other allocation lever (W_g)
#   sigma2           a different allocation objective
CANDS=${*:-"pcf2 pcf0 ncal256 pcf5 clwuni sigma2"}

cand_env () {   # name -> prints "PCF=.. NCAL=.. CLW=.. OBJECTIVE=.."
  case "$1" in
    pcf0)    echo "PCF=0     NCAL=128 OBJ=sigma  CLWX=macs" ;;
    pcf2)    echo "PCF=0.02  NCAL=128 OBJ=sigma  CLWX=macs" ;;
    pcf5)    echo "PCF=0.05  NCAL=128 OBJ=sigma  CLWX=macs" ;;
    ncal256) echo "PCF=0.02  NCAL=256 OBJ=sigma  CLWX=macs" ;;
    clwuni)  echo "PCF=0.02  NCAL=128 OBJ=sigma  CLWX=uniform" ;;
    sigma2)  echo "PCF=0.02  NCAL=128 OBJ=sigma2 CLWX=macs" ;;
    *)       echo "" ;;
  esac
}

echo "########## MP RESCUE at T=$T (eff-$((T*2)))  $(hostname) $(date +%T)"
echo "########## search window: shuffle(0)[$SEARCH_SKIP:$((SEARCH_SKIP+SEARCH_N))], disjoint from the report set"

# ---- the uniform comparator, on the SEARCH window
srch_uni () {
  local tag="s_uni_$T"
  [ -f "$OUT/$tag.json" ] && { echo "[resc] $tag cached"; return; }
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$T" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --data_root "$DATA" --max_images "$SEARCH_N" --skip_images "$SEARCH_SKIP" \
    --batch_size 16 --workers 8 --seed 0 --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" | grep -E "^\[result\]|Error|Traceback"
}
srch_uni

# ---- headroom guard
HR=$(python - "$REPORT_OUT" "$OUT" "$T" <<'PY'
import json, sys
from pathlib import Path
rep, srch, t = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
fp = rep / "fp.json"
u  = srch / f"s_uni_{t}.json"
if not fp.exists() or not u.exists():
    print("nan"); raise SystemExit
print(f"{100*(json.loads(fp.read_text())['top1'] - json.loads(u.read_text())['top1']):.2f}")
PY
)
echo "[resc] headroom at eff-$((T*2)) = $HR pt (FP minus uniform SC, on the search window)"
if [ "$HR" != "nan" ] && [ "$(python -c "print(1 if $HR < $MIN_HEADROOM else 0)")" = "1" ]; then
  echo "[resc] REFUSING to search: headroom $HR pt < $MIN_HEADROOM pt."
  echo "[resc] At this budget uniform SC is already at FP, so any config that"
  echo "[resc] 'fixes' the ordering is fitting noise. Report the null instead."
  exit 0
fi

# ---- candidates
for c in $CANDS; do
  spec=$(cand_env "$c")
  [ -z "$spec" ] && { echo "[resc] unknown candidate '$c'"; continue; }
  eval "$spec"
  tag="s_mp_${T}_$c"
  echo "=== [$tag] $spec  $(date +%T)"
  if [ ! -f "results/mp_calib/${tag}_wrapper.json" ]; then
    CLW=$CLWX PRICING=macs QKG=per_row MPOPS=$ALLOPS OBJECTIVE=$OBJ \
    PCF=$PCF GLK=0 EXCL=10000 \
      bash cls/mp_cell2.sh "$tag" "$T" "$(ladder_for "$T")" "$NCAL" 16 2>&1 \
      | grep -E "^\[calib\] expected|rho=|Error|Traceback|CUDA" || true
  fi
  [ -f "results/mp_calib/${tag}_wrapper.json" ] || { echo "[resc] $tag NO TABLE"; continue; }
  [ -f "$OUT/$tag.json" ] && { echo "[resc] $tag eval cached"; continue; }
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$T" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table "results/mp_calib/${tag}_wrapper.json" \
    --adaptive_mp_ops "$ALLOPS" \
    --data_root "$DATA" --max_images "$SEARCH_N" --skip_images "$SEARCH_SKIP" \
    --batch_size 16 --workers 8 --seed 0 --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" | grep -E "^\[result\]|Error|Traceback"
done

echo ""
echo "########## RESCUE RANKING at eff-$((T*2)) -- SEARCH WINDOW ONLY, NOT REPORTABLE"
python cls/paired_stats.py "$OUT" --base "s_uni_$T" --against \
  "$(cd "$OUT" && ls s_mp_${T}_*.json 2>/dev/null | grep -v _trace | sed 's/\.json//' | paste -sd, -)" 2>&1 | tail -12
echo ""
python cls/iso_check.py "$OUT" --mac-ref "s_uni_$T" --cells \
  "$(cd "$OUT" && ls s_*_trace.json 2>/dev/null | sed 's/_trace\.json//' | paste -sd, -)" 2>&1 | tail -12

echo ""
echo "[resc] Deploy the winner at n=10000 with its OWN tag, e.g.:"
echo "[resc]   PCF=<w> NCAL=<w> CLW=<w> OBJECTIVE=<w> EXCL=10000 N=10000 OUT=$REPORT_OUT \\"
echo "[resc]     mp_cell mp_$T $T"
echo "[resc] and record in the report that the config was SELECTED on a disjoint window."
echo "########## MP RESCUE COMPLETE $(date +%T)"
