# vit_sc/det — 实验结果

本文件汇总 det 侧的 QwT-SC 补偿实验结果。当前的生产配方 = **cross-seed
余弦门控 + head-aligned SC 补偿器**，与 cls 侧 ([`docs/SC_COMP_ALGORITHM.md`](../docs/SC_COMP_ALGORITHM.md))
使用同一套门控算法和同一系列补偿核。

项目规划见 [`README.md`](README.md)。

## FP baseline (COCO val, n_eval=100)

| metric | value |
|---|---:|
| bbox AP | 68.73 |
| segm AP | 61.61 |

源自 `results/fp100/metrics.json`。生产配置（n_eval=5000）下 FP 数字应
在同一量级，因为 EVA-ViTDet 在 N=5000 vs N=100 之间几乎无 noise（metric
是检测 AP，不是 top-1 比例）。

## Production sweep (n_eval=5000, 2026-04-25 之后)

> **生产驱动器**：[`experiments/qwt_det_compensate.py`](experiments/qwt_det_compensate.py)
> 用 cross-seed 余弦门控 + head-aligned 补偿核（默认配置：
> `cos_threshold=0.5, last_block_cos_threshold=0.8, lookahead_veto,
> n_heads=16, comp_sc_prec=8, comp_mode=sc`）。
>
> **Sweep 矩阵**（与 cls
> [`sweep_int678_skip30.sh`](../cls/experiments/sweep_int678_skip30.sh) 同形）:
> `prec_tag ∈ {p7, avg192, p8} × mode ∈ {uniform, mp} × qwt ∈ {off, on}`
> = 10 个 cell（p8_mp 省略，与 cls 一致）。
>
> - `p7` = `--sc_prec 7`（base stoc_len=128）。
> - `avg192` = `--sc_prec 8 --mp_levels 192 --mp_fractions 1.0` 应用于所有
>   SC ops（uniform fractional MP，效力 stoc_len=192 ≈ 7.58 bits）。
> - `p8` = `--sc_prec 8`（base stoc_len=256）。
> - `uniform` = 同一 stoc_len 应用于所有 (op, block) 对。
> - `mp` = 由
>   [`experiments/mp_budget_swap_search.py`](experiments/mp_budget_swap_search.py)
>   产生的 per-(op, block) sl_map（搜索目标 main_sl 等于该 prec_tag 的均值）。
>
> 运行脚本：
> [`experiments/sweep_2026-04-25.sh`](experiments/sweep_2026-04-25.sh)。

详见 `results/sweep_2026-04-25/` 下的每个 cell。**TBD：** 等 sweep 跑完
后填入下表。

| prec_tag | mode | qwt | bbox AP | segm AP | admitted | comments |
|---|---|---:|---:|---:|:---:|---|
| p7 | uniform | off | TBD | TBD | n/a | raw SC, sc_prec=7 |
| p7 | uniform | on  | TBD | TBD | TBD/40 | + cross-seed + head-aligned comp |
| p7 | mp      | off | TBD | TBD | n/a | per-(op,block) sl_map |
| p7 | mp      | on  | TBD | TBD | TBD/40 | + comp |
| avg192 | uniform | off | TBD | TBD | n/a | uniform stoc_len=192 |
| avg192 | uniform | on  | TBD | TBD | TBD/40 | + comp |
| avg192 | mp      | off | TBD | TBD | n/a | search-derived |
| avg192 | mp      | on  | TBD | TBD | TBD/40 | + comp |
| p8 | uniform | off | TBD | TBD | n/a | raw SC, sc_prec=8 |
| p8 | uniform | on  | TBD | TBD | TBD/40 | + comp |

下面的旧 sc_prec×K 矩阵保留作为记录（替换前的初版）。

| sc_prec | K | mode | raw SC bbox | comp bbox | Δ bbox | admitted blocks |
|---:|---:|---|---:|---:|---:|:---:|
| 7 | 20 | uniform | TBD | TBD | TBD | TBD/40 |
| 7 | 20 | mp192 | TBD | TBD | TBD | TBD/40 |
| 7 | 30 | uniform | TBD | TBD | TBD | TBD/40 |
| 7 | 30 | mp192 | TBD | TBD | TBD | TBD/40 |
| 8 | 20 | uniform | TBD | TBD | TBD | TBD/40 |
| 8 | 20 | mp192 | TBD | TBD | TBD | TBD/40 |
| 8 | 30 | uniform | TBD | TBD | TBD | TBD/40 |
| 8 | 30 | mp192 | TBD | TBD | TBD | TBD/40 |

> **预期定性结论**（来自 cls N=50k）：
> - largest Δ at lowest sc_prec：cls p7 +3.93 vs p8 +0.13；det 应同向。
> - admitted blocks ≈ (N-2)/N：EVA-40 应 ≈38/40；blocks 0 / 39 拒掉。
> - mp192 vs uniform 在同一 target SL 下基本持平（结构性，详见
>   [`docs/SC_COMP_ALGORITHM.md`](../docs/SC_COMP_ALGORITHM.md) §"MP-vs-uniform parity"）。

## Sensitivity matrices 来源

[`sensitivity/skip/skip_worst_{10,20,30}_int{6,7}.json`](sensitivity/skip/) —
来自 [`sensitivity/sensitivity_int{6,7}_n100.json`](sensitivity/) 的 6×40
COCO-AP-drop 矩阵，按 bbox_ap_drop 排序后取最差 K 个 (op, block) 留 FP。

> **`int8` schedule 缺失**：当前没跑过 sc_prec=8 的 sensitivity sweep。
> sc_prec=8 的 sweep 暂用 `skip_worst_K_int7.json`（保守替代：int7 最敏
> 感的 cell 在 int8 下也是最敏感的 ordering，但绝对数值更小）。如果
> int8 实测下出现 admitted < (N-2)/N 的反常，再做一次 `int8_n100`
> sensitivity sweep。

## 配方变化（vs 旧版 RESULTS）

旧版本（≤ 2026-04-23）使用 r²-门 + `--min_r2`、`--avg_sc_draws` 这两
个被弃用的 flag。**当前驱动器已重写**为 cross-seed 余弦门，详见
[`experiments/qwt_det_compensate.py`](experiments/qwt_det_compensate.py)
模块文档。Cross-seed 门相对 r² 门的物理推导和 N=50k 证据见
[`third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py`](../third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py)。

## 复现命令

```bash
# Smoke (10 min)
python experiments/qwt_det_compensate.py \
    --sc_prec 8 \
    --sc_ops_per_block_json sensitivity/skip/skip_worst_30_int7.json \
    --n_calib 64 --n_eval 100 \
    --calib_seed 1 --calib_seed_b 2 \
    --cos_threshold 0.5 --last_block_cos_threshold 0.8 --lookahead_veto \
    --comp_mode sc --comp_sc_prec 8 \
    --head_aligned --n_heads 16 \
    --out_json results/qwt_det/smoke.json

# Production sweep (4-GPU pool, ~10-12 h wall)
nohup env \
  RES_DIR=results/sweep_2026-04-25 \
  LOG_DIR=logs/sweep_2026-04-25 \
  HEAD_ALIGNED=1 HEAD_ALIGNED_HEADS=16 \
  bash experiments/sweep_2026-04-25.sh \
  > logs/sweep_2026-04-25/sched.log 2>&1 &
```

## 旧版结果（已弃用，仅供参考）

旧 `min_r2` sweep 的 n_eval=10 表参见 git 历史
（`8742921 det: RESULTS.md with min_r2 sweep + CLI overrides for datasets/ckpt`）。
那一轮是 r²-门下的早期探索，已被 cross-seed 门取代，绝对数字不再具备
生产参考价值（n_eval=10 噪声 ±2 AP）。
