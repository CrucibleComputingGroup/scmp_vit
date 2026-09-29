# Per-row mixed precision on ViT — what actually buys accuracy

DINOv2 ViT-L/14 + linear head, ImageNet-1k val, **1000-image seeded subset**
(`--seed 0`), run 2026-07-29 on gl1807. Calibration draws 128 images disjoint
from the eval set. Every SC cell runs all five op types across all 24 blocks,
`sc_prec=8`, `--halve 1`, `chunk_d=128`.

> **This supersedes the 2026-07-25 edition of this file.** That study ran at
> n=2048 with `qk` quantized `per_head` and without protected channels; both
> turned out to matter more than anything it measured. Its numbers are not
> comparable to these and should not be mixed with them.

## Units

`t32` / `r32_*` are **halved** cycle counts (`--halve 1`). The figure's x-axis
is representable levels = 2 × halved, the only axis on which SC and INT are
comparable.

| tag | halved cycles | nominal | levels | INT equivalent |
|---|---:|---:|---:|---|
| t32 | 32 | 64 | 64 | int6 |
| t64 | 64 | 128 | 128 | int7 |
| t96 | 96 | 192 | 192 | **none** |
| t128 | 128 | 256 | 256 | int8 |

## Statistics

At n=1000 the marginal Wilson half-width is ±2.3–2.7 pt, **wider than every
effect here**. All cells score the same images in the same order and dump
per-image top-1 hits, so differences are tested PAIRED (McNemar exact +
bootstrap CI) by `cls/paired_stats.py`. Sanity check: int8 vs FP differ on only
4 of 1000 images, giving a paired interval of ±0.4 pt — about 6× tighter than
the marginal one.

**Do not quote a marginal error bar against a paired result in this report.**
The figure still draws marginal bars because that is what a single point
honestly carries on its own; the claims come from the paired tables.

---

## 1. Headline

At **eff 64** — the only budget with real headroom (uniform SC 0.783 vs FP
0.854):

| cell | MP row axis | MP channel axis | realized | top-1 | paired Δ vs uniform | p |
|---|:-:|:-:|---:|---:|---:|---:|
| `uni_32` uniform SC | — | — | 32.00 | 0.7830 | — | — |
| `r32_base` | ✓ | — | 32.25 | 0.7810 | −0.20 | tie |
| `uniprot_31` | — | ✓ | 31.62 | 0.8020 | +1.90 | 0.13 |
| **`r32_pc`** | ✓ | ✓ | 32.09 | **0.8280** | **+4.50** [+2.2,+6.8] | **1e-4** |

```
row axis alone      −0.2
channel axis alone  +1.9
both                +4.5      >>  −0.2 + 1.9 = 1.7
```

**Super-additive; the interaction is ≈ +2.6.** Measured directly, MP's row-axis
increment on top of channel protection is **+2.60** (CI [+0.90, +4.30],
p=0.0038).

Replicated over 4 calibration seeds: 0.8280 / 0.8280 / 0.8270 / 0.8220 →
mean **0.8263, sd 0.28 pt, 4/4 significant**. The effect is 16× the
calibration-draw spread.

## 2. Why — outliers break the metric the allocator ranks on

The dispatch metric is `‖row‖_inf`. On `mlp_fc2` the post-GELU outlier channel
is present in nearly every row, so the metric measured *how big that row's
outlier is* rather than *how hard that row is*. Spearman ρ(metric, σ-benefit),
same ops / budget / seed, with and without protection:

| op | no protection | 2% protected | Δ |
|---|---:|---:|---:|
| `mlp_fc2` | **−0.105** | **+0.231** | +0.336 |
| `mlp_fc1` | +0.035 | +0.212 | +0.177 |
| `out_proj` | +0.310 | +0.436 | +0.126 |
| `qkv_proj` | +0.166 | +0.143 | −0.023 |

`mlp_fc2` goes from ranking rows **backwards** to ranking them usefully.
`qkv_proj`, which has no outlier problem, is unchanged — the control that makes
this causal rather than an artefact of narrowing the row.

**So protection buys twice**: the outlier channels get precision directly, AND
per-row dispatch becomes possible. An allocator cannot fix a scale an outlier
already ruined.

## 3. Everything that works *around* a broken ranking fails once it is fixed

| knob | metric broken | metric repaired |
|---|---:|---:|
| escape gate k=2 | **+1.08** | **−2.00** (p=0.017) |
| per-group ladders | −1.66 | — |
| `--metric-select auto` | — | −1.20 |

The escape gate is a **sign reversal**, not a magnitude change: it is an
absolute threshold that bypasses the band ranking, worth +1.08 when the ranking
is broken and −2.00 once it is not. Group ladders lose for the mirror reason —
subdividing a broken ranking buys nothing. A knob whose benefit is proportional
to how damaged the thing it bypasses is, is a workaround, not a feature.

**The simplest configuration wins.** 2% protection beats 5% (0.828 vs 0.820);
adding the gate, metric-select, or group ladders all cost accuracy.

## 4. Budget dependence

| budget | uniform | `r{t}_pc` | paired Δ | p | headroom vs FP |
|---|---:|---:|---:|---:|---:|
| eff 64 | 0.7830 | **0.8280** | **+4.50** | **1e-4** | 7.1 pt |
| eff 128 | 0.8360 | **0.8520** | **+1.60** | **0.040** | 1.8 pt |
| eff 192 | 0.8500 | 0.8460 | −0.40 | 0.58 | 0.4 pt |

Significant at the two aggressive budgets. At eff 192 nothing is resolvable —
uniform SC is already within 0.4 pt of FP, so no method can show anything and
none is claimed.

## 5. Hybrid SC/INT mask — dominated

10% of (op, block) groups routed to INT7, ranked by marginal knock-down ΔLoss:

| comparison | Δ | p |
|---|---:|---:|
| uniform+mask vs uniform @64 | +2.30 | — |
| MP+mask vs uniform+mask @64 | +2.10 | 0.038 |
| MP+mask vs uniform+mask @128 | +0.20 | 0.89 |
| **MP+mask vs MP alone @64** | **−0.10** | — |
| **ranked mask vs RANDOM mask @64** | **−0.30** | **0.82** |
| ranked mask vs RANDOM mask @128 | +1.10 | 0.12 |

Two negative results:

* **The mask adds nothing to a correctly configured MP cell** (0.827 vs 0.828).
  It helps uniform SC because uniform SC has no other outlier handling.
* **The sensitivity RANKING buys nothing.** A random 10% mask matched per-op
  performs the same. Predicted by two symptoms: the t32 and t64 masks shared
  only 2 of 14 groups, and the probe's mean ΔL was negative for every operator
  — the ranking is mostly sampling noise. The probe costs 77 min/budget and is
  not worth running.

⚠ Masked cells move work OFF the SC budget into INT, so their SC average
understates total compute. Masked cells are only comparable to other masked
cells; against an unmasked cell it is an energy-axis claim.

## 6. Two caveats that do not go away

**INT-asymmetric still wins at the aggressive end.** INT6 (64 levels) = 0.8540,
equal to FP; SC after all of the above is 0.8280 — a **2.6 pt** gap at the same
level count. On this network fixed-point is simply better at low precision. MP's
honest claim is *relative to uniform SC*, not against INT.

**The SC-specific claim survives, and is the strongest thing here.** At eff
**192** SC reaches 0.8500–0.8540 against INT8's 0.8520 at 256 levels — matched
accuracy at 25% fewer levels, on a budget fixed-point has no format for. This
does not depend on any MP machinery.

## 7. Reproduce

```bash
bash cls/run_all_1k.sh        # masks, references, MP ladder, hybrid, replication
bash cls/run_controls.sh      # uniprot / random-mask / per_head controls
bash cls/finalize.sh          # collect + paired tables + both figures
```

Figures: `figs/figure_{light,dark}.png` (levels axis),
`figs/decomposition_t32_{light,dark}.png` (what bought what).
Full session log incl. three fixed bugs: `results/NOTES_20260729.md`.
