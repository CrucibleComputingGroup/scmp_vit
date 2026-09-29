# vit_sc — repo guide

Stochastic-computing (SC) evaluation of Vision Transformers. Sibling of
`scmp_llm` (LLM) and `scmp_cnn` (MobileNetV3), sharing the same SC kernel
package.

## Layout (restructured 2026-07-25)

```
vit_sc/
├── kernels/          # OUR OWN scmp_kernels checkout — branch HPCA_VIT
├── model/
│   ├── _bootstrap.py # forces kernels/ onto sys.path ahead of any global install
│   └── sc_vit.py     # SCLinear + SC attention + patch_model  ← the integration
├── cls/              # ImageNet-1k classification (DINOv2 ViT-L/14 + linear head)
│   ├── eval.py       # main entry point
│   ├── run_smoke.sh  # small-subset FP-vs-SC smoke
│   └── sc_attention_patch.py   # compat shim -> model.sc_vit
├── det/              # COCO detection (EVA) — NOT yet migrated, see below
├── deprecated/       # retired in-tree SC stack (sc/ + sc_integration/)
├── tests/
│   └── test_sc_vit_smoke.py    # no-network end-to-end check of the SC path
└── third_party/QwT-SC
```

## The kernel package — three copies on this machine, pick deliberately

| repo | checkout | branch |
|---|---|---|
| `vit_sc/kernels` | ours | `HPCA_VIT` |
| `scmp_llm/kernels` | submodule | `sc/precision-trace` (parent commit) |
| `scmp_cnn/kernels` | standalone | `main` + uncommitted ρ-metric edits |

**`pip show scmp_kernels` reports an editable install pointing at
`scmp_cnn/kernels`.** So a bare `import scmp_kernels` from a neutral cwd gets
the *CNN* copy, which is on `main` and has neither `trace.py` nor the
precision-trace MP features. `model/_bootstrap.py` prepends `vit_sc/kernels` to
`sys.path` to override this; it runs on `import model.sc_vit`. Every run prints
which copy won:

```
[env] scmp_kernels vit_sc-local: /nfs/.../vit_sc/kernels/scmp_kernels/__init__.py
```

If that ever says `EXTERNAL:`, the numbers are from the wrong kernels.

**Never rename `kernels/` to `scmp_kernels/`.** A directory of that name on
`sys.path[0]` shadows the real package as an empty PEP-420 namespace package
(`__file__ = None`, no `sc_matmul`) and SC silently degrades to FP — the
symptom is SC being as fast as FP *and* giving identical predictions.

## SC surface

One unified entry point, `scmp_kernels.sc_matmul`:

| op | call |
|---|---|
| Q@K^T | `granularity="per_head"` — 3D, per-head (max,min) computed in-kernel |
| attn@V | `granularity="per_row"` — 3D batched, **one** call for all (B,H) |
| `nn.Linear` | `granularity="per_row"` + `chunk_d=128` |

### `chunk_d` is a CORRECTNESS requirement, not an optimization

ViT-L/14 `fc2` has `D_in=4096`. Measured 2026-07-25 (SC on `mlp_fc1`+`mlp_fc2`,
`sc_prec=8`, `stoc_len=256`, 256 images, everything else FP):

| `chunk_d` | top-1 |
|---|---|
| 128 (default) | **0.8438** — same as FP |
| 0 | **0.0000** |

**This resolves the old "SC on MLP collapses to ~0%" result.** The README's
`--sc_mlp 1` warning and the `everything` = 0.000 row in the fine-grained
scheduling table were both produced on the unchunked path: the legacy
`SCLinear` defaulted to `chunk_d=0` and the legacy `patch_model` never passed
it. The collapse is a property of unchunked SC on wide layers, **not** of SC on
MLP. With `chunk_d=128` no skipping is needed at `stoc_len=256`.

Mechanism (consistent with the old README's own diagnosis): post-GELU
activations put ~2/3 of their mass at `|x_int| <= 2` under per-row bipolar
quant, so accumulating all 4096 dims against one `cum_indicator` table lets SC
noise on the near-zero bulk dominate; the block output norm inflates (~2.5× by
block 22) and compounds through the residual stream.

`SCLinear` emits a `RuntimeWarning` if built with `chunk_d=0` on a layer of
width ≥ 1024, because the failure is otherwise silent.

`chunk_d` is the *only* cause: an `SC_OWEN_MODE` A/B (`bitrev` / `off` /
`random`) ruled the Owen scramble out as a second collapse channel.

### Owen scramble only matters at short `stoc_len`

Kernel-level check (`tests/check_owen_knob.sh`, 64×512 @ 128×512, sc_prec=8):

| mode | rel_err @ sl=256 | rel_err @ sl=16 |
|---|---|---|
| `bitrev` (default) | 0.027020 | 0.3195 |
| `off` | 0.027020 | 0.4841 |
| `random` | 0.027020 | 0.3113 |

At `stoc_len=256` all three are **bit-identical** — the Sobol sequence is fully
enumerated, so scrambling is a no-op. It only bites at short streams, where
`off` is ~51 % worse than `bitrev`. So: ignore `SC_OWEN_MODE` for uniform
`sl=256` work, but keep it at `bitrev` for MP ladders with low rungs, and make
sure calibration and evaluation use the same mode (thresholds do not transfer
across modes).

`sc_prec` sets the quantization grid; **`stoc_len` is the compute budget** (the
cycle count) and any integer works via early termination of one `sc_prec=8`
Sobol stream. `--halve 1` turns on the uSystolic/HUB bipolar cycle halving
(stream runs at `2**(sc_prec-1)`); it is OFF by default here, unlike scmp_llm
where the canonical operating point has it on — so **budgets in this repo are
nominal cycles** unless you pass `--halve 1`.

Two `--halve` footguns are now closed (2026-07-25):

* `eval.py --halve` **unset** inherits the setting from `--adaptive_mp_table`
  when one is given (a table's ladder rungs are cycle counts in *halved* space,
  so it is only valid at the `halve` it was calibrated at). No table ⇒ still
  OFF, so every non-MP invocation keeps its old meaning. Passing `--halve`
  explicitly against a table that disagrees now **raises** — as does a
  `sc_prec` or `chunk_d` mismatch (`load_adaptive_mp`).
* `_resolve_stoc_len` clamps the uniform stream to the grid ceiling, which under
  `--halve` is `2**(sc_prec-1)`. Previously `--halve 1` with no `--stoc_len` ran
  a **256-cycle stream on a 128-level grid** — 2× the cycles, no extra
  resolution, and the precision trace recorded the inflated 256. scmp_llm had
  this min-clamp; the port did not. Every committed script passes `--stoc_len`
  explicitly, so no published number was affected.

## Mixed precision

MP comes from `scmp_kernels.mp`, not a local fork, so calibration tables
transfer between vit_sc / scmp_llm / scmp_cnn.

* `MPConfig` — fixed quantile split over `stoc_len_levels`.
* `AdaptiveMPConfig` — **data-driven only**: a calibrated threshold table
  (`--adaptive_mp_table <wrapper.json>`) or quantile `target_fractions`
  (`--adaptive_mp_fractions`). The old vit_sc `threshold = α·progress + β`
  closed-form mode **was removed upstream** — `--adaptive_mp_alpha/beta` no
  longer exist.
* `RangeMPConfig` — static per-weight-group levels, resolved once at patch time.

### The four kernel MP features vit_sc did not use (wired 2026-07-29)

`scmp_kernels.mp` carried these all along — the classes were importable and
`scmp_llm` used them — but nothing in vit_sc produced or consumed them, so
every vit_sc table shipped without them. Session log: `results/NOTES_20260729.md`.

| knob | flag | what it buys |
|---|---|---|
| escape gate | `--escape-gate-k K` (+ `--escape-stoc-len`) | rows whose NORMALIZED metric exceeds `mu_b + K*sigma_b` jump to a high rung regardless of the band thresholds. The constants are the per-bucket `metric_mean`/`metric_std` the calibrator ALREADY wrote, so this is a wrapper-only setting — existing tables gain it without recalibration. |
| per-group ladders | `--group-ladder-k K` with a DENSE `--mp_levels` | one shared ladder wastes ~half its rungs: `out_proj` pins near the ceiling while `mlp_fc1` lives at the floor, so each sees only its half. Calibrate sigma on all rungs, deploy the K each (op × layer-bucket) puts mass on, then re-solve ONE shared λ against those ladders (`_global_lambda_multi`) so the budget still holds — restricting the argmin moves realized cost, so the second pass is not optional. |
| protected channels | `--protect-channel-frac F` | pin the top-F input channels of each linear at a high rung for every row, split off along the contraction dim. Exact: a matmul is additive over that dim, so `y = x[:,P]@W[:,P]ᵀ + x[:,R]@W[:,R]ᵀ`. Aimed at `mlp_fc2`, whose post-GELU rows carry sparse outliers over a near-zero bulk — the outlier sets the shared per-row scale, so splitting it off shrinks the step for every other channel. The residual budget is compensated so the cell stays iso-compute. |
| attention MP | `--qk_granularity per_row`, `qk,av` in `--operators` | `qk` used to dispatch per HEAD (16 units/block — a very thin population to fit thresholds on) and neither `qk` nor `av` was in `DEFAULT_OPERATORS`, so neither ever got a table. per_row gives each QUERY ROW its own range, which is what makes a row-subset re-quantization exact. |

**`--qk_granularity` also moves the UNIFORM arm** — measured +0.59 pt at eff 128
on 2048 images with no MP involved (0.8330 per_head → 0.8389 per_row). A
per_row MP cell must therefore be differenced against a per_row uniform cell;
against the published per_head row it books the scope gain twice. `eval.py`
inherits the setting from the table and RAISES on an explicit disagreement, the
same contract `--halve` has.

### Two bugs this surfaced (fixed — don't reintroduce)

1. **`_apply_escape_gate` rebuilt `level_row_indices` against the GLOBAL
   ladder** while `_classify_rows_by_thresholds` had keyed it by the BUCKET's.
   With per-group ladders on, every rung is re-keyed and rows run at stream
   lengths from a ladder they were never fitted on — silently, since equal-length
   ladders keep the indices in range. Fixed in `kernels/scmp_kernels/mp/config.py`
   to resolve the same bucket `get_levels` does. **This is a shared submodule:
   scmp_llm has the same bug at commit `01f8357`.**
2. **`av`'s dispatch metric could not survive a table round-trip.** The
   calibrator recorded the entropy peakiness score as the primary, but
   `select_dispatch_metrics` labels the primary `"amax"` unconditionally — so a
   ρ sign-flip would export `{"metric": "amax", "sign": -1}` and the runtime
   would deploy the row amax, a different ranking than the thresholds were cut
   on. Entropy is also absent from `compute_row_metric`, so no table can name
   it. `av`'s primary is now the row amax with l2/crest as candidates;
   `_av_row_score` survives only for the fixed-`MPConfig` path.

### What the 2026-07-29 run concluded (supersedes the numbers below)

Full write-up: `MP_REPORT.md`. Session log incl. three fixed bugs:
`results/NOTES_20260729.md`. At n=1000, eff 64, paired on identical images:

| configuration | top-1 | Δ vs uniform SC |
|---|---:|---:|
| uniform SC (qk per_row) | 0.7830 | — |
| MP, row axis only | 0.7810 | −0.20 (tie) |
| MP, channel axis only (2% protected) | 0.8020 | +1.90 |
| **MP, both axes** | **0.8280** | **+4.50** (p=1e-4) |

**Per-row MP alone buys nothing on this model; protected channels turn it into
+4.50, and the two are SUPER-ADDITIVE.** Protection works because it removes
the outliers AND repairs the row metric the allocator ranks on (`mlp_fc2`
ρ: −0.105 → +0.231). Corollary: every knob that instead works *around* a bad
ranking fails once the ranking is fixed — the escape gate REVERSES sign
(+1.08 → −2.00), group ladders lose (−1.66), `--metric-select auto` costs
−1.20. **Ship the simplest thing: `PCF=0.02`, nothing else.**

Also settled: the 10% hybrid INT mask adds nothing to a correctly configured MP
cell (0.827 vs 0.828), and its sensitivity RANKING is worth nothing — a random
per-op-matched mask ties it (−0.30, p=0.82). The 77-min/budget knock-down probe
is not worth running.

Unchanged caveat: INT-asymmetric still beats SC by 2.6 pt at eff 64 (INT6 =
0.8540 = FP). The SC-specific win is eff **192**, where SC matches INT8 at 25%
fewer levels on a budget fixed-point cannot express.

### Where the headroom actually is

The calibrator prints Spearman ρ(metric, σ-benefit) per operator — how well the
dispatch metric ranks rows by their true need for cycles. Measured with
attention in the table: `av` **−0.175**, `mlp_fc2` +0.130, `qkv_proj` +0.147,
`mlp_fc1` +0.243, `qk` +0.331, `out_proj` +0.399. At ρ=+0.13 on `mlp_fc2` —
a third of all MACs — the per-row split is close to arbitrary *within* the op,
so most of MP's win is coming from the CROSS-group allocation. **The lever is a
better metric, not more rungs**: `--metric-select auto` and protected channels
attack this; group ladders do not.

### Hybrid SC/INT mask

`--hybrid_config_json` routes chosen `(op, block)` groups to INT RTN while the
rest stay on SC, and composes with `--adaptive_mp`. Build one with
`cls/build_hybrid_mask.py`, which ranks groups by MARGINAL knock-down ΔLoss
(a small step below the deployment budget, not a knock to the floor — past the
collapse cliff ΔL saturates and ranks "who collapses" instead of "who is
fragile here"). Masks at different `--frac` off ONE sensitivity run are
strictly NESTED, so the fraction is a genuine dose axis.

⚠ **A masked group leaves the SC budget rather than shrinking it** — its cycles
are simply not spent, and not redistributed to the surviving SC groups. So a
hybrid cell at nominal budget T is NOT iso-compute with an unmasked cell at T;
it is cheaper. Measured 2026-07-29 off the traces at n=1000:

| pair | true SC MACs | true SC cycles | top-1 |
|---|---:|---:|---:|
| `r32_pc` → `hyb_mppc_32` | −7.9% | **92.1%** | −0.10 |
| `r64_pc` → `hyb_mppc_64` | −10.0% | **87.9%** | −0.70 |

Every pre-2026-07-29 "hybrid vs unmasked" comparison therefore charged the
hybrid arm for a budget it never spent, and `hyb_mppc_32`'s −0.10 pt was
actually a ~+0.6 pt WIN on the energy axis.

**Fix: raise the surviving groups' budget by `1/(1-f)`**, where `f` is the mask's
TRUE MAC-weighted fraction (compute it against a uniform cell's per-(op, block)
MAC counts, not the group count — the 14 groups of `mask10_t32` are 7.9% of MACs,
not 10%). At t32 that is 32 → 35, at t64 64 → 71. Check the ladder still
straddles the raised target. With that correction the hybrid arm goes from a tie
to **+2.10 pt over unmasked MP at matched true cycles** (`hyb_mppc_32iso` 0.8490
vs `r32_pc` 0.8280, p=0.017). `cls/lane_a_isohybrid.sh` is the driver;
`cls/iso_check.py --pairs` verifies the match.

Holding SC cycles constant hands the hybrid arm its INT7 work for free, so this
is the OPTIMISTIC bound. `cls/iso_check.py --int-cycle-cost c` prices an INT MAC
at `c` SC-cycle equivalents; at c=7 the t32 target would be 34 rather than 35,
i.e. about one cycle (~0.26 pt) of the gain.

### The precision trace OVER-COUNTS attention MACs — weight by true MACs

`model/sc_vit.py::_padded_level_gather` pads every (b, h) slice out to the
LARGEST per-slice row count at each rung (`M = counts.max()`) so ONE batched
`sc_matmul` serves all BH slices instead of a BH-long Python loop. Pad slots
repeat an already-selected row (a zero row would have a degenerate per_row
range) and their outputs are discarded by `valid` — correct, but **the kernel
still ran on them, and the trace records the padded tensor shape.**

Measured on `r32_pc` (2026-07-29): `qk` MACs come out **3.42×** and `av`
**3.76×** the uniform cell's, while every linear matches exactly. Raw total SC
cycles are inflated **+10.7%**, and `qk`+`av` appear as ~13% of MACs instead of
their true 4%. This is also why MP cells run at ~5.4 img/s against uniform's 8.2.

Padding is a GPU-batching artifact, not an architectural cost — a per-row SC
dispatcher spends exactly `Σ_rows sl(row)`. **So for any energy or iso-compute
claim, weight each (op, block) by its TRUE MAC count** (from any uniform cell's
trace; MAC counts are architecture-determined and identical across budgets) and
take only the stream length from the cell's own trace. It matters:

| accounting | `r32_pc` realized | vs `uni_32` 32.00 |
|---|---:|---|
| raw trace MACs | 32.09 | +0.3%, and total cycles +10.7% |
| **true MACs** | **31.81** | **−0.6% — iso, slightly conservative** |

So the headline +4.50 survives, but only the corrected accounting shows it is
on budget. `cls/iso_check.py` and `cls/collect_figure.py --mac-ref` both do this;
`collect_figure.py` additionally divides by the FULL model MAC count so a
masked cell lands at its real cost and equal x means iso-compute.

Tables carry per-op ρ-selected dispatch metrics (`amax` / `l2` / `crest` plus a
sign); `SCLinear` reads them via `get_dispatch_metric`. On scmp_cnn this metric
selection was worth +4.3 pt top-1 at an iso-compute int7 budget, so it is worth
calibrating rather than leaving at the `amax` default.

### The allocation objective — `--objective` / `--cross-layer-weight`

`act_global` counts benefit **per row** (`W_g = 1`) while pricing cost **per
MAC**. Those are different currencies, and the mismatch is what
`MP_REPORT.md` measured: `out_proj` (1.05 M MACs/row, 8 % of SC MACs) offers the
most σ-reduction per MAC, so the Lagrangian pins it at the ladder ceiling —
60 of a possible 64 cycles — while `mlp_fc1` (4.19 M MACs/row, 32 % of MACs) is
starved to 22, below its own quality cliff, and MP goes **negative** vs uniform
SC at that budget.

> ⚠ **`--argmin-pricing` is NOT a no-op once `qk`/`av` are in the MP table.**
> This file previously claimed it was, reasoning that on a ViT every operator
> sees the same rows/forward (257·B) so `rep_g` and `mac_per_row` differ by a
> global constant λ absorbs. That holds for a LINEAR-ONLY table. Attention has
> `B*H*N` rows against a linear's `B*N`; at H=16 the default `rep` pricing
> charges an attention row **16× a linear row**, and the allocator starves
> attention to the ladder floor — measured 2026-07-29: `qk` realized 16.6
> cycles against a 32 budget, top-1 **0.672 vs uniform's 0.783**. Use
> `--argmin-pricing macs` (the KKT-consistent price, where the subsample ratio
> cancels) for any table containing attention. Evidence:
> `results/diag/reppricing_32*`.

Two knobs ported from scmp_llm (2026-07-25) address the objective itself:

| flag | default | effect |
|---|---|---|
| `--cross-layer-weight {uniform,macs}` | `uniform` | `macs` sets `W_g = MACs/row`, so benefit and cost share one currency. The per-row argmin collapses to `σ + λ′·cycles` for **every** operator: one global cycles-per-σ exchange rate, no cost-side favouritism toward narrow layers. |
| `--objective {sigma,sigma2,delta_sigma2}` | `sigma` | `sigma2` penalizes the low-precision cliff superlinearly; `delta_sigma2` prices only the *increase* over each row's error at the top rung. Shared verbatim with scmp_llm via `cls/mp_objectives.py`. |

Defaults are byte-identical to the pre-2026-07-25 calibrator (`objective_errors`
is a passthrough at `sigma`, and `W_g` is exactly `1.0`). Non-default settings
tag the table's `method` (`..._wmacs`, `..._sigma2`) and export a
`group_weights` map, so a bundle always says which objective produced it.

**Measured 2026-07-25**, 3 objectives × 2 `W_g` at two budgets, 2048 images,
every cell trace-verified iso-compute (`MP_REPORT.md` §Fixing the objective):

| objective | W_g | eff-64 top-1 (unif 0.7505) | eff-128 top-1 (unif 0.8330) |
|---|---|---:|---:|
| `sigma` | `uniform` (shipped) | 0.7402 (−1.03) | 0.8364 (+0.34) |
| `sigma2` | `uniform` | 0.7319 (−1.86) | 0.8418 (+0.88) |
| `delta_sigma2` | `uniform` | 0.7441 (−0.63) | 0.8403 (+0.73) |
| **`sigma`** | **`macs`** | **0.7573 (+0.68)** | **0.8398 (+0.68)** |
| `sigma2` | `macs` | 0.7446 (−0.59) | 0.8364 (+0.34) |
| `delta_sigma2` | `macs` | 0.7441 (−0.63) | 0.8364 (+0.34) |

Use this table only for the *ranking within a budget* — it is one calibration
draw (seed 1234), which a 4-seed replication showed to be a high draw at
effective 64 and a low one at effective 128. **The absolutes above do not
replicate; the recommendation below comes from the 4-seed paired result**
(`MP_REPORT.md` §Replication, `results/mp_eval/rep{32,64}_s*`):

| budget | `wuniform` | `wmacs` | paired | verdict |
|---|---:|---:|---:|---|
| eff 64 | 0.7383 (−1.22 vs unif) | 0.7516 (+0.11) | **+1.33** (sd 0.55, 4/4 pos) | use **`macs`** |
| eff 128 | 0.8424 (+0.94 vs unif) | 0.8386 (+0.56) | **−0.38** (sd 0.30, 0/4 pos) | keep **`uniform`** |

**So `--cross-layer-weight` is budget-dependent: `macs` at aggressive budgets
(eff ≤ 64), `uniform` (the default) at mild ones (eff ≥ 128).** The mechanism
predicts this — cost-side favouritism toward the cheap `out_proj` is harmless
while the budget is loose and only becomes a defect once the bill lands on
`mlp_fc1` below its cliff. At eff 64 `macs` closes MP's loss to a **tie** with
uniform SC (+0.11, between-seed sd 0.99 pt); it does not win there.

Do NOT stack `sigma2` on top of `macs`: σ² feeds the already-high-σ `mlp_fc2` and
starves `mlp_fc1`, the operator on its quality cliff — once benefit and cost share
a currency, an extra cliff penalty is over-correction.

Run one cell with `CLW=macs bash cls/mp_cell.sh <tag> ...` (env knobs
`OBJECTIVE` / `CLW`; give such a cell its own tag — the calibration cache is
keyed by tag alone).

### `--variants` — N tables from one σ draw

σ collection does not depend on the objective or `W_g`, so
`--variants all` (or `sigma:uniform,sigma:macs`) exports one table per pair from a
single collection. Use it for any objective comparison: the calibration-sample sd
is 0.33 pt while these effects are ~1 pt, so independent calibrations per variant
would confound the two. `cls/mp_variant_sweep.sh` drives calibrate-once →
evaluate-each; `cls/collect_variants.py` prints the grid and **withholds the
`vs uniform` delta** unless the trace-measured realized budget matches the
uniform comparator within 1.5 cycles.

## Running cls

```bash
# from vit_sc/, on a GPU node
export TORCH_HOME=$PWD/.torch_cache
DATA=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data

python -u cls/eval.py --mode fp --data_root $DATA --max_images 512
python -u cls/eval.py --mode sc --data_root $DATA --max_images 512 \
  --sc_prec 8 --stoc_len 256 --sc_ops mlp_fc1,mlp_fc2,qk,av,proj

bash cls/run_smoke.sh 512 smoke        # FP + 4 SC configs, writes results/smoke/
python tests/test_sc_vit_smoke.py      # no network / no weights needed
```

`--sc_ops` takes names from `{mlp_fc1, mlp_fc2, qk, av, proj}`, or a 5-element
0/1 vector in that order. `--sc_ops_per_block_json` gives per-block control.

## Precision trace

`SC_MP_TRACE=out.json` records every `sc_matmul` (shape, `stoc_len`, op, block)
for the energy model — host-side metadata only, no device sync. `model/sc_vit.py`
already calls `trace.set_context(op, block, unit)` on every SC path, which is
the only integration a consumer needs.

## Compute

Login node is CPU-only; SC needs CUDA (Triton). GPU access is via Slurm — `ssh
gl18XX` only works while you hold an allocation on that node (`pam_slurm_adopt`
rejects otherwise, which looks like an SSH failure). Nodes are RTX PRO 6000
Blackwell, 98 GB. `/home` is NFS-shared with the compute nodes, so one pip
install and one Triton cache serve both.

## Not yet migrated

`det/` and the top-level `experiments/`, plus 5 scripts in `cls/experiments/`
(`eval_custom_sl_map_fpcomp`, `mp_budget_swap_search`, `mp_uniform_vs_fine`,
`qwt_sc_overnight`, `sensitivity_all_ops`) still import the retired stack. They
need `HeadAlignedSCLinear`, `sc_prec_for_stoc_len`, `set_noise_corrections` —
symbols with no scmp_kernels equivalent. To run them unchanged:

```bash
export PYTHONPATH="$PWD/deprecated:$PWD/deprecated/sc:$PYTHONPATH"
```

That reactivates the OLD kernels for those scripts — never mix it into a `cls/`
run. See `deprecated/README.md` for the full API mapping.

Note `sc_prec_for_stoc_len` is conceptually obsolete: the new kernel takes
`stoc_len` directly at `sc_prec=8`, which is the point of the fine-grained
budget knob.
