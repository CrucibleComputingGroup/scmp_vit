# vit_sc/det — SC on EVA-01 ViTDet COCO detection

Ports the classification SC pipeline in `vit_sc/` to **object detection +
instance segmentation on COCO** using an **EVA-01 ViT-L/14 ViTDet** backbone.
Reuses:

- `vit_sc/sc/` (Triton XNOR matmul kernels, Sobol RNG, configs)
- `vit_sc/sc_integration/` (`SCLinear`, `sc_matmul_*`)
- Detectron2 + the EVA-ViT backbone under
  `QwT/QwT-det-RepQ-ViT/eva_det/`

## Files

```
det/
├── README.md
├── sc_patch/
│   ├── __init__.py
│   ├── sc_matmul.py      # SCMatMul module (swaps EVA's matmul1 / matmul2)
│   └── sc_model_eva.py   # module-swap driver, per-(op, block) schedule
├── fp_eval.py            # FP baseline COCO mAP
├── sc_eval.py            # SC eval with skip_worst-style schedule
└── results/              # metrics.json / coco_instances_results.json
```

## Why module-swap (and not vit_sc's monkey-patch)

`vit_sc/sc_attention_patch.py` rewrites the attention `forward` for DINOv2's
3-D `(B, N, C)` signature. EVA-ViT in ViTDet uses a 4-D `(B, H, W, C)`
signature with window partition — the rewritten forward would not slot in.
Instead the EVA `Attention` module already exposes `matmul1` / `matmul2` as
named child modules, so we simply swap them (like the QwT-det reference).

## Op taxonomy

Mirrors `vit_sc`'s `SC_OP_NAMES`:

| Name      | What's replaced                            |
|-----------|--------------------------------------------|
| `mlp_fc1` | `block.mlp.fc1` (nn.Linear → SCLinear)     |
| `mlp_fc2` | `block.mlp.fc2` (nn.Linear → SCLinear)     |
| `qkv_proj`| `block.attn.qkv` (nn.Linear → SCLinear)    |
| `out_proj`| `block.attn.proj` (nn.Linear → SCLinear)   |
| `qk`      | `block.attn.matmul1` (MatMul → SCMatMul)   |
| `av`      | `block.attn.matmul2` (MatMul → SCMatMul)   |

`proj` is kept as two independent names (`qkv_proj`, `out_proj`) to match
`vit_sc`; a convenience `"proj"` alias expands to both.

EVA ViTDet has **40 blocks** (not 24 — DINOv2-L). Schedules are lists of
length 40.

## Skip-worst schedule

Provide one of (flags on `sc_eval.py`):

- `--sc_config all_on` — every op SC in every block
- `--sc_config skip_last_fc2_K --skip_last_k 4` — keep the last K blocks' fc2
  in FP (transfer of the `vit_sc` finding that late fc2 is most sensitive)
- `--sc_ops_per_block_json path.json` — load an arbitrary schedule
  `{"mlp_fc1":[0/1]*40, "mlp_fc2":..., "qk":..., "av":..., "qkv_proj":...,
  "out_proj":...}`

A true sensitivity-matrix-driven `skip_worst_K` for EVA-on-COCO requires
running the per-(op, block) single-swap sweep (≈240 evals × N_eval images).
That is a **follow-up**, not in this scaffold.

## Usage

```bash
# FP baseline on first 100 COCO val images
cd <repo_root>/det
python fp_eval.py --n-eval 100

# SC eval, all ops on, sc_prec=8
python sc_eval.py --n-eval 100 --sc_config all_on --sc-prec 8

# SC eval, skip last-4-fc2 (approximate skip_worst analog)
python sc_eval.py --n-eval 100 --sc_config skip_last_fc2_K --skip_last_k 4

# SC eval with custom per-block schedule
python sc_eval.py --n-eval 100 --sc_ops_per_block_json schedule.json
```

## Known limitations

- **No QwT compensation yet.** The scaffold runs bare SC. QwT-style LS
  compensation from `QwT/QwT-det-RepQ-ViT/eva_eval/quant/compensation.py`
  is drop-in compatible but not wired up here (follow-up).
- **No multi-GPU shard.** `fp_eval.py` / `sc_eval.py` are single-process. A
  sharded wrapper mirroring `QwT-det/eva_eval/quant_eval_shard.py` can be
  bolted on later.
- **Sensitivity matrix for EVA/COCO not measured.** `skip_last_fc2_K` is a
  heuristic transferred from DINOv2/ImageNet.

## Checkpoint

- EVA ViTDet cascade_mask_rcnn weights:
  `/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva_coco_det.pth`
- D2 datasets root:
  `DETECTRON2_DATASETS=/scratch/nbleier_owned_root/nbleier_owned1/shared_data`
