# vit_sc — Stochastic-Computing ViT evaluation

Evaluate a pretrained ViT (DINOv2 ViT-L/14 + linear classifier) on ImageNet-1k
with stochastic-computing (SC) matmuls swapped into the attention path, and
compare against the FP baseline.

No training. Only inference. The SC kernels (bipolar / XNOR matmul, Sobol RNG)
come from `sc/` and `sc_integration/`; `sc_attention_patch.py` monkey-patches
every DINOv2 attention block to route Q@K^T, Attn@V, qkv_proj, and out_proj
through the SC kernels at configurable precision.

## Model and dataset

- **Model:** `dinov2_vitl14_lc` loaded via `torch.hub` from
  `facebookresearch/dinov2` — DINOv2 ViT-L/14 backbone (24 blocks, d=1024,
  16 heads, head_dim=64) + the published ImageNet-1k linear head. Weights
  download automatically on first run (~1.15 GB).
- **Dataset:** ImageNet-1k **validation** split (50 000 images), stored as
  HuggingFace parquet shards (`validation-00000-of-00014.parquet` …
  `validation-00013-of-00014.parquet`, ~6.3 GB total).
  Preprocessing: resize 256 (bicubic) → center-crop 224 → ImageNet mean/std.

## Setup

```bash
# 1. Conda env
conda create -n vit_sc python=3.11 -y
conda activate vit_sc
pip install torch==2.10.0 torchvision triton==3.6.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install timm==1.0.26 pyarrow datasets huggingface_hub pillow numpy
```

Tested with torch 2.10.0+cu128, triton 3.6.0, timm 1.0.26, Python 3.11, CUDA
12.8. Requires a CUDA GPU (eval has no CPU path); tested on RTX 4080 (16 GB)
and RTX PRO 6000 Blackwell (97 GB).

```bash
# 2. ImageNet-1k val data (gated — needs an HF token)
export HF_TOKEN=hf_xxxxxxxx   # from https://huggingface.co/settings/tokens
# Accept terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k first.
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ILSVRC/imagenet-1k",
    repo_type="dataset",
    allow_patterns=["data/validation-*.parquet"],
    local_dir="./data/imagenet",
)
PY
```

This places the 14 val shards under `./data/imagenet/data/` — that directory
is what `--data_root` expects.

### Running on a SLURM cluster (recipe: UMich GreatLakes `gpu-rtx6000`)

Concrete recipe for the `nbleier_owned1` allocation on a 4× RTX PRO 6000
Blackwell (97 GB each) node. Generalises cleanly to any multi-GPU SLURM
partition — the only non-obvious tunable is `--mem`.

**Rules of thumb (from this repo's profiling):**

| Knob | Value | Why |
|---|---|---|
| `--mem` (total) | **≥ 32 GB × #GPUs** (so `128G` for 4 GPUs) | `--mem` is the cgroup total across the whole allocation, **not** per-GPU. A single ViT-L SC + QwT-comp job peaks at ~10–15 GB RAM (FP+SC model copies + calib activation buffer + DataLoader workers). Setting `--mem=32G` on a 4-GPU job is the #1 cause of silent SIGKILL the moment two jobs load models simultaneously. |
| `--cpus-per-gpu` | 16 | Enough for `--workers 2–4` per job; DataLoader is never the bottleneck (SC kernel is). |
| `--workers` (DataLoader) | `2` when 4-way parallel, `4` solo | Fewer fork-COW spikes against the cgroup cap. |
| `--batch_size` | `8` for SC eval, `32+` for FP | SC is launch-bound on the per-sample Sobol pool construction, not compute-bound — bigger batches barely help. |
| Data placement | `/scratch` | 6.3 GB of parquet shards — never put on `/home` (slow + quota). |
| `--out_json` paths | unique per run | Otherwise parallel jobs race on the same file. |

**One-shot allocation template** (runs until 1 AM tomorrow):

```bash
current_time=$(date +%s)
tomorrow_1am=$(date -d "tomorrow 01:00:00" +%s)
diff=$((tomorrow_1am - current_time))
time_string=$(printf "%d:%02d:00" $((diff/3600)) $(((diff%3600)/60)))

salloc --account=nbleier_owned1 --partition=gpu-rtx6000 \
       --gres=gpu:4 --time=$time_string \
       --cpus-per-gpu=16 --mem=128G
```

**Env + data setup (once per fresh env)** — run after `salloc` drops you on
the compute node:

```bash
# Conda env (see top-level Setup for full pip steps)
conda activate vit_sc
git submodule update --init --recursive   # pulls QwT-SC submodule

# Download 6.3 GB of ImageNet val parquet to /scratch, once.
DATA_ROOT=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet
export HF_TOKEN=hf_xxxxxxxx
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ILSVRC/imagenet-1k", repo_type="dataset",
                  allow_patterns=["data/validation-*.parquet"],
                  local_dir="$DATA_ROOT")
PY
```

**Four-way parallel sweep** — one config per GPU, `CUDA_VISIBLE_DEVICES`
pinning, backgrounded, joined with `wait`:

```bash
DATA=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data
mkdir -p logs results
for i in 50 40 30 20; do
    gpu=$(( (50 - i) / 10 ))    # 50→0, 40→1, 30→2, 20→3
    CUDA_VISIBLE_DEVICES=$gpu nohup python -u experiments/qwt_sc_overnight.py \
        --sc_config skip_worst$i --n_calib 256 --n_eval 500 \
        --batch_size 8 --workers 2 \
        --comp_mode sc --n_variants 4 --w_scales 0.5,0.75,1.0 \
        --head_aligned --n_heads 16 --skip_baseline \
        --data_root $DATA \
        --out_json results/skip_worst${i}_sccomp_headaligned.json \
        > logs/skip_worst${i}.log 2>&1 &
done
wait
echo "all done; summaries:"
for i in 50 40 30 20; do grep -A4 "=== SUMMARY ===" logs/skip_worst${i}.log; done
```

Each run is ~8 min (calib 5 min + eval 2 min + FP reference 3 s). With 4
GPUs in parallel the whole sweep finishes in ~10 min wall-clock (vs ~32 min
serial). Per-GPU throughput is identical to an RTX 4080 — SC is
launch-overhead bound, so Blackwell doesn't speed up the kernel, it just
lets you run four configs at once.

**Diagnosing silent kills.** If a run exits without writing its `--out_json`
and the log tail is stuck at `[data] building loaders`, it was SIGKILL'd by
the cgroup. Check with:

```bash
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,MaxRSS,ReqMem
```

`MaxRSS` close to `ReqMem` = cgroup OOM — bump `--mem` and re-run. (Note:
pipelines like `python ... 2>&1 | tee log` mask the kill as "exit 0" because
`tee` itself exits cleanly; prefer `> log 2>&1` for unambiguous exit codes.)

**Serial fallback** if `--mem` can't be raised: drop the `&` and `wait`,
keep every run on GPU 0. Memory footprint per job is fine under 32 GB
alone; the problem is only parallelism.

## Running eval

`XFORMERS_DISABLED=1` is required so DINOv2 falls back to the SDPA forward path
that `sc_attention_patch.py` can patch; `eval.py` sets it by default.

### FP baseline

```bash
python eval.py --mode fp \
    --data_root ./data/imagenet/data \
    --batch_size 32 --workers 4 \
    --max_images 1000 --seed 0 \
    --out_json results/fp_n1k.json
```

### SC runs

`--sc_prec K` sets the SC precision (stochastic bitstream length = `2**K`; the
report uses `K=8` → length 256). Per-op flags turn SC on/off for each
attention submodule:

| Flag | Op affected |
|---|---|
| `--sc_qk 1` | Q @ K^T |
| `--sc_av 1` | Attn @ V (per-row bipolar grouping — needed for softmax'd attn) |
| `--sc_qkv_proj 1` | input qkv linear |
| `--sc_out_proj 1` | output projection linear |
| `--sc_mlp 1` | MLP fc1 + fc2 (**collapses to ~0% accuracy** at uniform prec=8; not recommended) |
| `--mlp_skip_first_k K` / `--mlp_skip_last_k K` | keep first/last K blocks' MLP in FP when `--sc_mlp 1` |

Example — full attention in SC, MLP and head stay FP:

```bash
python eval.py --mode sc --sc_prec 8 \
    --sc_qk 1 --sc_av 1 --sc_qkv_proj 1 --sc_out_proj 1 \
    --data_root ./data/imagenet/data \
    --batch_size 8 --workers 4 \
    --max_images 1000 --seed 0 \
    --out_json results/sc_full_attn_n1k.json
```

`--max_images 0` evaluates the full 50 000-image val set. Use `--batch_size 8`
for SC (the per-sample RNG-pool construction is the bottleneck); FP runs happily
at `--batch_size 32` or higher.

## Replication (N=1000, seed=0, RTX 4080)

Matches `TASK_REPORT.md` to within ±1 pt; FP is bit-identical. SC variance is
from the Sobol RNG and is deterministic for a given seed/stoc_len.

| Config | This repo (replica) |           | `TASK_REPORT.md` (original) |           |
|---|---|---|---|---|
|    | **top-1** | top-5 | top-1 | top-5 |
| FP baseline                       | **0.854** | 0.983 | 0.854 | 0.983 |
| SC QK                             | **0.740** | 0.934 | 0.746 | 0.923 |
| SC QK + AV                        | **0.719** | 0.908 | 0.723 | 0.915 |
| SC QK + AV + qkv_proj + out_proj  | **0.698** | 0.899 | 0.705 | 0.886 |

Raw JSON results are in `results/replica_*.json`.

## Fine-grained per-block per-op SC scheduling

Beyond the uniform-across-all-blocks flags above, `patch_model` accepts an
explicit schedule saying *which block's which operator* goes SC. The five
operators are:

| op name | what it is | FP fallback path |
|---|---|---|
| `mlp_fc1` | MLP's first linear (D=1024 → 4096) in each block | `nn.Linear` |
| `mlp_fc2` | MLP's second linear (D=4096 → 1024) | `nn.Linear` |
| `qk`      | Q @ K^T attention-score matmul | scaled dot product |
| `av`      | Attn @ V | FP matmul |
| `proj`    | qkv input linear + out_proj output linear (toggled together) | `nn.Linear` |

A schedule is a `dict[op_name, list[0/1]]` with each list of length 24 (one
entry per block, `1` = SC, `0` = FP). Missing ops default to all-FP.

### Python API

```python
from sc_attention_patch import patch_model

# SC everything except late-block fc2 (the most sensitive linears):
schedule = {
    "mlp_fc1": [1] * 24,
    "mlp_fc2": [1] * 16 + [0] * 8,   # block 16..23 fc2 stays FP
    "qk":      [1] * 24,
    "av":      [1] * 24,
    "proj":    [1] * 24,
}
stats = patch_model(model, sc_prec=8, sc_ops_per_block=schedule)
```

### CLI

Write the same dict as JSON, then:

```bash
python eval.py --mode sc --sc_prec 8 \
    --sc_ops_per_block_json my_schedule.json \
    --data_root ./data/imagenet/data \
    --batch_size 8 --max_images 500 --seed 0 \
    --out_json results/my_run.json
```

The flag takes priority over `--sc_mlp / --sc_qk / ...`. A coarser
uniform-across-blocks shortcut is also available:

```bash
# 5-vector in op order [mlp_fc1, mlp_fc2, qk, av, proj]
python eval.py --mode sc --sc_ops 1,1,0,0,1 ...
# or comma-separated names
python eval.py --mode sc --sc_ops mlp_fc1,mlp_fc2,proj ...
```

## End-to-end results with fine-grained scheduling (N=500)

`experiments/sensitivity_all_ops.py` builds a 24×5 per-(op, block) sensitivity
matrix using a calibrated Sobol-SC noise surrogate (closed-form Gaussian
approximation, ~100× faster than real SC — see `sc_integration/noise_matmul.py`).
`experiments/e2e_fine_grained.py` takes that matrix, ranks all 120 (op, block)
pairs by logit-L2 drift, and runs real SC schedules that skip the top-K worst.

Real SC, bipolar, `sc_prec=8`, DINOv2 ViT-L/14 + linear head, 500 val images:

| config | #SC ops | top-1 | top-5 | notes |
|---|---:|---:|---:|---|
| `fp`                 |   0 | **0.846** | 0.986 | baseline (no SC) |
| `attn_full`          |  72 | 0.696 | 0.902 | uniform qk+av+proj across all 24 blocks |
| `sc_skip_worst50`    |  70 | **0.748** | 0.944 | **+5.2 pt over `attn_full` at ~same SC coverage** |
| `sc_skip_worst40`    |  80 | 0.742 | 0.944 | |
| `sc_skip_worst30`    |  90 | 0.688 | 0.896 | knee of the curve |
| `sc_skip_worst20`    | 100 | 0.438 | 0.670 | |
| `sc_skip_worst10`    | 110 | 0.322 | 0.512 | |
| `everything`         | 120 | 0.000 | 0.006 | all ops, all blocks |

Headline: **per-(op, block) sensitivity scheduling beats block-level
attention-only SC by ~5 pt top-1 at matched SC coverage**, with no retraining,
no quant-aware calibration, no mixed precision. The flat knee lies around
70–90 SC ops out of the 120 possible; beyond that the long tail of
mildly-sensitive ops compounds non-linearly.

### Why it works

The 24×5 sensitivity matrix surfaces three universal patterns (see
`results/sensitivity_all_ops.json`):

- **Late-block fc2 is the dominant damage path.** Block 22's fc2 alone has
  surrogate logit-L2 = 35.3 — larger than the next 4 worst (op, block) pairs
  combined. Blocks 20–23's fc2 are ranks 1–4.
- **All five operators are depth-monotonic in late layers.** Every op's
  sensitivity curve is flat for blocks 0–15, then rises sharply toward the
  classifier. The final LN + linear head has no downstream averaging
  operator, so late-layer noise lands directly in the logits.
- **Only one early-layer anomaly**: `mlp_fc1` block 0 (L2 = 10.08), because
  it takes raw patch-embedding activations before LayerNorm settles.

Skipping the top-50 worst (op, block) pairs therefore mostly drops *late*
ops across all five kinds, which is why it outperforms a naive "SC all
attention" rule that leaves all fc1/fc2 FP but SCs *every* block's qk+av+proj
— including the final ones that hurt most.

## QwT-style residual compensation (no retraining)

Port of [QwT's](https://arxiv.org/abs/2411.13918) closed-form
`CompensationBlock` to the SC setting. Per block, fit a small residual
`(Y_fp − Y_sc) ≈ X W + b` on 256 calibration images via ridge LS, then wrap
each block as `out = block_sc(x) + x·W + b`. No backprop, no fine-tuning,
~5 min on an RTX 4080. The library lives in the QwT-SC fork, vendored here
as a git submodule at `third_party/QwT-SC/QwT-vit-sc/` — see that
sub-project's [README](third_party/QwT-SC/QwT-vit-sc/README.md) for the full
formulation, overhead analysis, and API.

### Setup

After cloning this repo:

```bash
git submodule update --init --recursive
```

### Reproducing the sweep

```bash
# Worst-hit baseline: full-attention SC.
python experiments/qwt_sc_compensation.py \
    --sc_config full_attn --n_calib 256 --n_eval 500 \
    --batch_size 8 --out_json results/qwt_sc_full_attn_n500.json

# Best combo: fine-grained sensitivity scheduling + QwT comp.
python experiments/qwt_sc_compensation.py \
    --sc_config skip_worst50 --n_calib 256 --n_eval 500 \
    --batch_size 8 --out_json results/qwt_sc_skip_worst50_n500.json
```

Supported `--sc_config` presets: `full_attn`, `qk_only`, `qk_av`, and the
fine-grained `skip_worst{50,40,30,20}` / `all_ops` configs.

### Results (N=500 val images, seed=0, `sc_prec=8`, RTX 4080)

| Config | SC ops | Raw SC top-1 | + QwT top-1 | Δ | Gap-to-FP closed |
|---|---:|---:|---:|---:|---:|
| FP baseline                           |   0/120 | **0.846** | —       | —        | —    |
| `full_attn` (qk+av+qkv_proj+out_proj) |  72/120 |     0.696 | **0.786** | +9.0 pt  | 60% |
| `qk_av`                               |  48/120 |     0.718 | **0.822** | +10.4 pt | 81% |
| `qk_only`                             |  24/120 |     0.736 | **0.834** | +9.8 pt  | 89% |
| `skip_worst50` (fine-grained)         |  70/120 |     0.748 | **0.836** | +8.8 pt  | 90% |

Headline: uniformly +8.8 to +10.4 pt top-1 across every SC config tested;
compensation **stacks** with sensitivity-aware scheduling, landing
`skip_worst50 + QwT` at **83.6%** — within **1.0 pt of FP** at 70/120 SC
ops. `qk_only + QwT` reaches **83.4%** at only 24 SC ops, strictly
dominating the previous state-of-the-art (`skip_worst50` at 74.8% without
compensation). Raw JSON in `results/qwt_sc_*_n500.json`.

**Why it works.** The linear fit captures the mean-shift component of the SC
residual (per-dim bias, linear-in-x drift, cross-dim correlated error). It
can't chase the per-sample Sobol-realization variance, but downstream
LayerNorm + 257-token aggregation averages that remainder, so the logit
error is dominated by the correctable mean-shift. See the sub-project
README for per-block r² diagnostics and overhead analysis (+25 M params,
+21% FLOPs, no inference slowdown — lost in the SC bottleneck).

### SC-kernel compensator (`--comp_mode sc`)

The driver also supports running the residual `x @ W + b` itself through SC
kernels — the same XNOR / Sobol kernels that the block uses — instead of
keeping it in FP. Goal: a *fully SC* inference path so an SC accelerator
doesn't need an FP unit just for the comp. The W, b solve stays closed-form
ridge; only the inference-time matmul flips to SC via `SCLinear`. The comp's
`(D=1024, sc_prec=8)` SNG is taken from the same `_CFG_CACHE` entry as any
block ops at the same dim, so the comp's Sobol pool is deterministic across
calibration ↔ inference and overlaps with block ops at matching dims.

```bash
# SC-kernel comp — drop-in replacement of FP comp.
python experiments/qwt_sc_compensation.py \
    --sc_config skip_worst50 --n_calib 256 --n_eval 500 \
    --batch_size 8 --comp_mode sc --comp_sc_prec 8 \
    --out_json results/qwt_sc_skip_worst50_n500_sccomp.json
```

**Driver flags (SC comp):**

| Flag | Default | What it does |
|---|---|---|
| `--comp_mode {fp,sc}` | `fp` | Comp matmul kernel. `sc` uses `SCLinear`. |
| `--comp_sc_prec INT` | `8` | SC bit-precision of the comp matmul. |
| `--comp_sc_mode {bipolar,unipolar}` | `bipolar` | Per-row SC quant mode of the comp matmul. |
| `--comp_refit_iters INT` | `0` | If `>0` and `comp_mode=sc`: K rounds of *noise-aware refit* — measure the comp's own SC noise δ on the calibration set, refit `W` against `R − δ`. Linearization can overshoot on early/small-residual blocks; tune cautiously. |
| `--comp_variant_select` | off | Per-block, try `{default, antithetic-K}` Sobol configs for the comp and pick the lower-residual one (1 config bit per block in HW). Empirically only ~−0.15 noise correlation, so gain is small and not strictly Pareto. |

### Results — SC comp vs FP comp (N=500, sc_prec=8, RTX 4080)

Apples-to-apples (`avg_sc_draws=1`, `ridge=1e-2`, `start_block=0`):

| sc_config | Raw SC | + FP-comp (baseline) | + SC-comp (naive) | Δ vs FP-comp | Note |
|---|---:|---:|---:|---:|---|
| `qk_only`     | 0.736 | 0.834 | 0.810 | **−2.4 pt** | block uses D=64 per-head SNG → no pool overlap with comp's D=1024 SNG → SC noise of comp adds independently. |
| `qk_av`       | 0.718 | 0.822 | _TBD_ | — | (untested at present) |
| `full_attn`   | 0.696 | 0.786 | _TBD_ | — | (untested at present) |
| `skip_worst50`| 0.748 | 0.836 | **0.840** | **+0.4 pt** | block has SC ops at D=1024 (`qkv_proj`, `out_proj`, `mlp_fc1`) → pool literally collides with the comp via `_CFG_CACHE` → free control-variate cancellation. |

**Take-away.** SC-comp matches or beats FP-comp **wherever the comp's Sobol
pool overlaps a block SC op at the same dim**. Where it doesn't (per-head
QK at D=64 vs comp at D=1024), naive SC-comp loses to FP-comp by ~2 pt
because its SC noise is informationally independent of the block residual.
Per-block scrambling tweaks (`--comp_variant_select`) and noise-aware refit
(`--comp_refit_iters`) were both attempted as fixes; neither reliably
recovers the gap on `qk_only`-style configs. **The head-aligned SC
compensator below is the structural fix.**

### Head-aligned SC compensator (recommended default)

The naive SC compensator uses a monolithic `SCLinear` at D=1024 — its Sobol
pool is `_CFG_CACHE[(1024, 8)]`, which has zero informational overlap with a
block's per-head QK operating at D=64. `HeadAlignedSCLinear`
(`sc_integration/head_aligned_comp.py`) fixes this by restructuring the
comp into **16 per-head (D=64) SC matmuls** whose Sobol pool is literally
the same `_CFG_CACHE[(64, 8)]` the block QK uses.

```
Naive SC comp:     x (N,1024) ─[SCLinear D=1024]─> Δ(N,1024)       pool: _CFG_CACHE[(1024,8)]
Head-aligned comp: x (N,1024)─reshape (N,16,64)
                   ├─head 0: (N,64) ─[SCLinear D=64 ] ─> Δ0
                   ├─head 1: (N,64) ─[SCLinear D=64 ] ─> Δ1        each pool: _CFG_CACHE[(64,8)]
                   ├─ …                                             = same as block QK
                   └─head15: (N,64) ─[SCLinear D=64 ] ─> Δ15
                   sum + bias ─────────────────────> Δ(N,1024)
```

**Identical FLOPs:** 16 × 64 × 1024 = 1024 × 1024 inner reductions. Same
bit-level work, same weights (just sliced differently). **Different noise
structure:** comp's SC noise now shares the block's per-head Sobol
sequences → anti-correlation reappears via shared-pool control-variate.

GPU speed: the 16-launch overhead is mitigated by CUDA-stream dispatch
(bit-exact same output; 22 % faster than the sequential version; actually
slightly *faster* than monolithic D=1024 SCLinear on ViT-L token counts
because smaller kernels overlap better).

#### Results — all-SC comp, head-aligned variant (N=500, sc_prec=8, RTX 4080)

With per-block variant selection from a small menu of
**{head-aligned, naive D=1024} × {W-scales 0.5, 0.75, 1.0}** (4 bits of LUT
per block → 96 bits for ViT-L), calibration-picked:

| sc_config | Raw SC | + FP-comp (ref) | **+ all-SC comp (head-aligned menu)** | Δ vs FP-comp |
|---|---:|---:|---:|---:|
| `qk_only`     | 0.736 | 0.834 | **0.828** | −0.6 pt |
| `qk_av`       | 0.718 | 0.822 | **0.824** | **+0.2 pt** |
| `full_attn`   | 0.696 | 0.786 | **0.790** | **+0.4 pt** |
| `skip_worst50`| 0.748 | 0.836 | **0.842** | **+0.6 pt** |

**3 of 4 sc_configs now beat FP-comp at zero FP-MAC cost.** Calibration
picks `head_aligned/s1.00` in 89 / 96 blocks across all four configs; the
remaining 7 picks are minor scale/architecture tweaks on `skip_worst50`
only. Antithetic Sobol and random alternative seeds were never selected —
the head-alignment architecture does essentially all the work.

#### Reproduce

```bash
# Pure all-SC comp with head-aligned menu (recommended default):
python experiments/qwt_sc_overnight.py \
    --sc_config qk_only --n_calib 256 --n_eval 500 --batch_size 8 \
    --comp_mode sc --n_variants 4 --w_scales 0.5,0.75,1.0 \
    --head_aligned --n_heads 16 --skip_baseline \
    --out_json results/pure_sc_qk_only_combined_pureSC.json

# Batch all four sc_configs + ablations (≈2 hours on a 4080):
bash experiments/run_overnight_pure_sc.sh
```

**Driver flags (head-aligned):**

| Flag | Default | What it does |
|---|---|---|
| `--head_aligned` | off | Add head-aligned variants to the per-block menu. |
| `--head_aligned_only` | off | Drop regular SC variants entirely; menu is head-aligned only. |
| `--n_heads INT` | `16` | Number of head splits (must divide D=1024). |
| `--n_variants INT` | `1` | Sobol seed-bank size (default + antithetic + random alt K-side seeds). Empirically K=1 is usually enough when head-aligned is in the menu. |
| `--w_scales LIST` | `"1.0"` | Comma-separated W-premultipliers. `"0.5,0.75,1.0"` is the recommended menu. |

See `OVERNIGHT_REPORT.md` for the full ablation and variant-usage
statistics.

### Hardware framing

For an SC accelerator, the relevant cost is **SC ops removed from the FP
path**, not GPU wall-clock:

- FP comp: 24 × 1024² = **25.2 M FP-MACs per image** (21 % of total FP).
- All-SC (head-aligned) comp: 24 × (16 × 64 × 1024) = **25.2 M SC-MACs**
  (identical inner reductions; one SC-MAC at sc_prec=8 = 256 bit-ANDs +
  popcount, no FP multiplier needed).
- Per-block scheduler LUT: ≤ 4 bits × 24 blocks = 96 bits.
- Net silicon delta: **one fewer FP multiplier**. No other mandatory new
  hardware (the SC matmul unit is already required for the attention path).

On GPU the Triton SC kernel has launch overhead (see above — mitigated by
CUDA streams but still present); the numbers above are **accuracy at
theoretical FLOPs/energy parity**, not stopwatch numbers.

## File layout

```
eval.py                              FP / SC evaluation driver
imagenet_parquet.py                  ImageFolder-style dataset over HF parquet shards
sc_attention_patch.py                DINOv2 patcher: per-block per-op SC schedule support
sc/                                  SC kernels (Triton XNOR matmul, Sobol RNG, configs)
sc_integration/                      SC matmul wrappers + calibrated noise surrogate
    noise_matmul.py                  Closed-form Gaussian surrogate for fast sweeps
experiments/                         Sensitivity sweeps + end-to-end configs
    sensitivity_all_ops.py           24 x 5 per-(op, block) sensitivity matrix
    e2e_fine_grained.py              Real-SC eval of sc_skip_worstK schedules
    mlp_onebyone.py                  MLP first-N / last-N block sensitivity sweep
    calibrate_noise_model.py         Calibrate surrogate correction factor against real SC
    qwt_sc_compensation.py           QwT-style residual compensation driver (uses QwT-SC submodule)
third_party/QwT-SC/                  git submodule -> Allenjin123/QwT.git (QwT fork with QwT-vit-sc)
results/                             Per-run JSON (top-1, top-5, sensitivity data, qwt_sc_*)
TASK_REPORT.md                       Original run log on gl1804 (2×RTX PRO 6000)
```
