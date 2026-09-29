#!/usr/bin/env bash
# cls smoke test on the scmp_kernels-backed SC path.
# Small ImageNet subset, DINOv2 ViT-L/14 + linear head. One GPU.
#
#   bash cls/run_smoke.sh [N_IMAGES] [TAG]
set -uo pipefail

N=${1:-512}
TAG=${2:-smoke}
VIT_SC=/nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc
OUT=$VIT_SC/results/$TAG
LOGS=$OUT/logs
mkdir -p "$LOGS"

export TORCH_HOME=$VIT_SC/.torch_cache
export IMAGENET_ROOT=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

cd "$VIT_SC"

run () {   # run <name> <extra args...>
  local name=$1; shift
  echo "=== [$name] $(date +%T) ==="
  python -u cls/eval.py \
    --data_root "$IMAGENET_ROOT" \
    --max_images "$N" --batch_size 16 --workers 4 --seed 0 \
    --out_json "$OUT/$name.json" "$@" 2>&1 \
    | grep -viE "futurewarning|import pynvml" \
    | tee "$LOGS/$name.log" | grep -E "^\[(env|data|sc|result|eval)\]|top1"
  echo
}

echo "### vit_sc cls smoke | N=$N | $(hostname) | $(date) ###"
echo

# 1. FP reference
run fp --mode fp

# 2. SC on Q@K^T only (cheapest SC surface)
run sc_qk_sl256 --mode sc --sc_prec 8 --stoc_len 256 --sc_ops qk

# 3. SC on attention (QK + AV)
run sc_attn_sl256 --mode sc --sc_prec 8 --stoc_len 256 --sc_ops qk,av

# 4. SC everywhere (attention + MLP + projections)
run sc_all_sl256 --mode sc --sc_prec 8 --stoc_len 256 \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj

# 5. Budget knob: same surface, shorter stream
run sc_all_sl64 --mode sc --sc_prec 8 --stoc_len 64 \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj

echo "### done $(date) — results in $OUT ###"
export OUT_DIR=$OUT
python - <<'PY'
import json, glob, os
out = os.environ.get("OUT_DIR") or ""
rows = []
for f in sorted(glob.glob(os.path.join(out, "*.json"))):
    d = json.load(open(f))
    rows.append((os.path.basename(f)[:-5], d.get("n"), d.get("top1"),
                 d.get("top5"), d.get("stoc_len"), d.get("img_per_s")))
if rows:
    print(f"{'config':<18}{'n':>6}{'top1':>9}{'top5':>9}{'stoc_len':>10}{'img/s':>9}")
    for r in rows:
        sl = r[4] if r[4] is not None else "-"
        print(f"{r[0]:<18}{r[1]:>6}{r[2]:>9.4f}{r[3]:>9.4f}{str(sl):>10}{r[5]:>9.1f}")
PY
