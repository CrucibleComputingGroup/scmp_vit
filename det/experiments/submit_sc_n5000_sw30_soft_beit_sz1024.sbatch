#!/bin/bash
#SBATCH --job-name=sc_n5k_sw30_sb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --output=logs/sbatch_%x_%j.out
#SBATCH --error=logs/sbatch_%x_%j.err
#
# Submit one config per call, five tags total:
#   TAG=int8         sbatch --job-name=sc_n5k_sw30_int8_sb         experiments/submit_sc_n5000_sw30_soft_beit_sz1024.sbatch
#   TAG=len192       sbatch --job-name=sc_n5k_sw30_len192_sb       experiments/submit_sc_n5000_sw30_soft_beit_sz1024.sbatch
#   TAG=int7         sbatch --job-name=sc_n5k_sw30_int7_sb         experiments/submit_sc_n5000_sw30_soft_beit_sz1024.sbatch
#   TAG=len96        sbatch --job-name=sc_n5k_sw30_len96_sb        experiments/submit_sc_n5000_sw30_soft_beit_sz1024.sbatch
#   TAG=int8_l192sch sbatch --job-name=sc_n5k_sw30_int8_l192sch_sb experiments/submit_sc_n5000_sw30_soft_beit_sz1024.sbatch
#
# Common: K=30 with TAG-matched schedule (skip_worst_30_<tag>.json),
# size=1024, n_eval=5000, --use_soft_nms, --interp_type beit.
#
# Resumable: sc_eval.py writes checkpoint.pt every 50 images. To resume
# a killed/timed-out run, resubmit with RESUME=1 in the env, e.g.
#   TAG=int8_l192sch sbatch --export=ALL,RESUME=1 ... <this script>
set -euo pipefail
cd /home/yjrcs/SC_V/vit_sc/det
mkdir -p logs results

: "${TAG:?must set TAG env var (int8 | len192 | int7 | len96 | int8_l192sch)}"
: "${N_EVAL:=5000}"
: "${SIZE:=1024}"

# Per-tag dispatch: precision + matched schedule + (optional) MP level args.
case "$TAG" in
  int8)
    SC_PREC=8
    SCHED=sensitivity/skip/skip_worst_30_int8.json
    EXTRA=()
    ;;
  len192)
    # Fast path: --stoc_len 192 directly, NO mp_cfg wrapping. The previous
    # version routed through --mp_levels 192 --mp_fractions 1.0 which adds
    # MP overhead (extra abs/amax + index_select per linear, plus a Python
    # BH loop in SCMatMul av_mp) and ends up SLOWER than int8 despite less
    # SC compute.
    SC_PREC=8
    SCHED=sensitivity/skip/skip_worst_30_len192.json
    EXTRA=(--stoc_len 192)
    ;;
  int7)
    SC_PREC=7
    SCHED=sensitivity/skip/skip_worst_30_int7.json
    EXTRA=()
    ;;
  len96)
    # Fast path: see len192 note above. --stoc_len 96 with sc_prec=7 (since
    # ceil(log2(96)) = 7).
    SC_PREC=7
    SCHED=sensitivity/skip/skip_worst_30_len96.json
    EXTRA=(--stoc_len 96)
    ;;
  int8_l192sch)
    # int8 precision (sc_prec=8, default stoc_len=256) but with the
    # len192-derived per-block sensitivity ranking. Tests whether the
    # skip mask tuned at stoc_len=192 transfers to stoc_len=256.
    SC_PREC=8
    SCHED=sensitivity/skip/skip_worst_30_len192.json
    EXTRA=()
    ;;
  *)
    echo "[sbatch] unknown TAG=$TAG" >&2; exit 2
    ;;
esac

OUT_TAG="sc_sw30_${TAG}_sz${SIZE}_n${N_EVAL}_soft_beit"

echo "[sbatch] job=$SLURM_JOB_ID node=$SLURMD_NODENAME TAG=$TAG SC_PREC=$SC_PREC SIZE=$SIZE N_EVAL=$N_EVAL  $(date)"
echo "[sbatch] OUT_TAG=$OUT_TAG"
echo "[sbatch] SCHED=$SCHED"
echo "[sbatch] EXTRA=${EXTRA[*]:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /home/yjrcs/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2

if [[ "${RESUME:-0}" == "1" ]]; then EXTRA+=(--resume); fi

python -u sc_eval.py \
    --n-eval "$N_EVAL" \
    --sc_prec "$SC_PREC" \
    --size "$SIZE" \
    --sc_ops_per_block_json "$SCHED" \
    --use_soft_nms \
    --interp_type beit \
    --save_every 50 \
    --out-tag "$OUT_TAG" \
    "${EXTRA[@]}"

echo "[sbatch] done $(date)"
