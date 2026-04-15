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
results/                             Per-run JSON (top-1, top-5, sensitivity data)
TASK_REPORT.md                       Original run log on gl1804 (2×RTX PRO 6000)
```
