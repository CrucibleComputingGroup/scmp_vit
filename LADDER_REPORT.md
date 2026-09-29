# Per-group ladders: the monotonicity argument is right, and it buys nothing

**Date:** 2026-07-29 (late) · **Node:** gl1807 · DINOv2 ViT-L/14 + linear head,
ImageNet val seed-0 subset, n=1000, all comparisons PAIRED.

The question: a per-group ladder should be **≥** a shared one, because "every
group picks the same rungs" is inside the per-group feasible set. So the measured
−1.66 pt (`s32_glk` 0.7407 vs `s32_base` 0.7573, MP_REPORT.md) looks like a bug
rather than a property. It is neither.

## 1. The −1.66 pt was never a superset comparison

| | candidate ladder | deployed per group | span | groups reaching 64 |
|---|---|---:|---|---:|
| `s32_base` | 5 rungs `64,48,32,24,16` | all 5 | 64..16 | **96 / 96** |
| `s32_glk` | 8 rungs dense | **3** (93 groups) / 2 (3 groups) | mean 40.7..24.8 | **2 / 96** |

0 of 96 groups' spans cover the baseline's; 94 have a top rung below 64; 36 are
capped at ≤32. The baseline's solution is **not in the restricted cell's feasible
set**, so ≥ was never predicted for that pair — it measured "3 rungs < 5 rungs".

Read off the precision traces, the damage localizes to under 1% of the compute:

| | MACs at sl=64 | `mlp_fc1` top rung | fc1 avg sl | fc2 avg sl |
|---|---:|---:|---:|---:|
| `s32_base` | **0.79%** | 64 | 23.81 | 42.29 |
| `s32_glk` | 0.05% | **48** | 23.54 | 42.90 |

The cross-operator allocation barely moved (both realize 31.99 / 31.96 against a
32.00 target). The entire −1.66 came from removing a ceiling that <1% of MACs
used — the same heavy tail that makes 2% protected channels worth +4.5 pt.

## 2. The clean test: a strict superset, and it is a tie

`{64,48,32,24,16}` is a strict subset of `{64,56,48,40,32,24,20,16}`. Nothing in
the study had ever run the dense ladder unrestricted — of 111 calibrated cells,
109 are 5 rungs / `GLK=0` and 2 are the `GLK=3` ones. Three new cells, all on the
shipped eff-64 recipe (`CLW=macs`, `PCF=0.02`, qk `per_row`, `PRICING=macs`):

| cell | ladder | true SC cycles | true avg sl | top-1 | paired Δ vs `r32_pc` | p |
|---|---|---:|---:|---:|---:|---:|
| `uni_32` | uniform SC | 2587.5T | 32.00 | 0.7830 | — | — |
| `r32_pc` | 5 shared | 2571.9T | 31.81 | **0.8280** | — | — |
| `r32_pcD` | 8 shared (**superset**) | 2569.5T | 31.78 | 0.8220 | −0.60 | 0.52 |
| `r32_pcD_g5` | 5 per-group from 8 | 2566.9T | 31.75 | 0.8210 | −0.70 | 0.44 |
| `r32_pcD_g3` | 3 per-group from 8 | 2580.3T | 31.91 | 0.8210 | −0.70 | 0.48 |

All four within 0.7 pt, every p > 0.4, every cell iso within 0.4%. **Ladder
design is not a lever on this model in either direction.** For scale: protected
channels move the same cell by +4.5 pt.

**`GLK=3` — the exact setting that lost 1.66 pt — is now a tie.** And k=3 and
k=5 give IDENTICAL top-1 (0.8210 both), which the tables explain: the endpoint
rule pins the same occupied endpoints whatever k is, so both cells have the same
ceiling distribution (57 of 144 groups at 64) and differ only in how many middle
rungs they carry. The number of rungs per group, between 3 and 5 and 8, is worth
nothing here.

The superset cell is not ignoring its new freedom — it spends 9.21% of MACs on
rungs 56/40/20 that did not exist before. What it does NOT do is change the
allocation. Restricting to the bulk (non-protected) slice of the four linears:

| cell | bulk avg sl | top rung used | bulk MACs at the 64 ceiling |
|---|---:|---:|---:|
| `r32_pc` | 31.26 | 64 | 2.17% |
| `r32_pcD` | 31.22 | 64 | 0.83% |
| `r32_pcD_g5` | 31.23 | 64 | 0.80% |

Three different rung sets, and the σ-optimal allocation lands within 0.04 cycles
of the same place. Finer within-group resolution converts to zero top-1 because
it converts to almost no change in what the rows actually run at.

## 3. Why ≥ does not transfer: the surrogate is not top-1

Monotonicity is provable for the quantity the allocator minimizes, and it holds
exactly as predicted. MAC-weighted σ-surrogate error from the calibration tables,
at *identical* realized budget:

| cell | rungs | MAC-weighted surrogate error | `expected_flop_avg_stoc_len` |
|---|---:|---:|---:|
| `r32_pc` | 5 | 0.109258 | 31.3839 |
| `r32_pcD` | 8 | **0.108764** (−0.45%) | 31.3839 |

The dense superset **does** find a better optimum of the objective. It does not
find a better top-1. That is the ρ leak, measured elsewhere at ρ(metric,
σ-benefit) = +0.13 … +0.40: a better argmin of the surrogate is not a better
argmin of accuracy.

Note the *size* of the surrogate gain — 0.45% from 3 extra rungs. Ladder
resolution was never the binding constraint, so there was almost nothing for
monotonicity to deliver even in principle. This is the same conclusion
`CLAUDE.md` reached from ρ ("the lever is a better metric, not more rungs"),
now measured directly rather than inferred.

## 4. Per-group ladders can never BE a superset — a structural fact

`_select_group_ladder` was fixed after the −1.66 run to pin both **endpoints of
the group's occupied range** (top = headroom, bottom = the cheap end the budget
is paid from) instead of ranking purely by occupancy. The fix works globally:

| | groups | rungs/group | groups reaching 64 |
|---|---:|---|---:|
| `s32_glk` (occupancy only, k=3) | 96 | 3 | **2** |
| `r32_pcD_g5` (endpoints pinned, k=5) | 144 | mostly 5 | **57** |

But the rule pins the occupied top, **not the ladder top**, and the occupancy it
reads comes from an *unrestricted* λ solve on the dense grid. Any group that puts
no mass on the top rung there loses that ceiling permanently. `mlp_fc1` — 32% of
all MACs — does exactly this: its most common k=5 ladder is `(32,24,20,16)`,
ceiling 32, where the shared ladder let it reach 64.

So **no value of k makes the per-group cell a superset of the shared one.** If
per-group ladders are ever worth keeping, the fix is to force the GLOBAL ladder's
top rung into every group's kept set rather than the group's occupied top, which
would make ≥ structurally valid instead of accidentally violated.

### The ceiling story does not survive its own control

An earlier draft of this section argued that protected channels neutralize the
lost ceiling — `--protect-channel-stoc-len` defaults to the ladder top and the
protected split is its own `sc_matmul`, so for `mlp_fc1` block 0 of `r32_pcD_g5`
the bulk (`d_in` 1004) runs at 16 while the protected 2% (`d_in` 20) runs at
**64, 100%**. That is a true description of the datapath. As an *explanation* of
the −1.66 it is wrong, and the control that kills it is `m32_D_g3_nopc`:
`PCF=0`, k=3, no protected slice anywhere (`mlp_fc1`'s minimum `d_in` is 1024),
7 of 96 groups reach 64 against `m32_base`'s 96 of 96, MACs at the ceiling
0.90% → 0.25%. The cleanest possible reproduction of the old situation.

**It costs −0.80 pt (p=0.47), not −1.66.**

| cell | protection | groups reaching 64 | Δ vs same-recipe shared ladder | p |
|---|---|---:|---:|---:|
| `s32_glk` (old rule, k=3) | none | 2 / 96 | −1.66 | — |
| `r32_pcD_g3` (k=3) | 2% | 57 / 144 | −0.70 | 0.48 |
| `r32_pcD_g5` (k=5) | 2% | 57 / 144 | −0.70 | 0.44 |
| `m32_D_g3_nopc` (k=3) | **none** | **7 / 96** | **−0.80** | 0.47 |

Protection changes the per-group penalty by 0.10 pt. It is not the mechanism.

More importantly, **−1.66 and −0.80 are not statistically distinguishable** — the
paired CIs are roughly ±1.8 pt and overlap heavily. The ceiling narrative was
built to explain a difference between cells that is not there. What the four
measurements support is only this:

* the per-group penalty is **consistently negative and small**: −1.66 / −0.70 /
  −0.70 / −0.80, **4 of 4 negative** (sign test p=0.0625), none individually
  significant, magnitude around −0.5 to −1 pt;
* it is **insensitive to protection, to k (3 vs 5), and to which rungs survive** —
  7/96 groups keeping the ceiling costs the same as 57/144 keeping it;
* meanwhile the σ-surrogate IMPROVES (§3).

That combination — surrogate better, top-1 consistently slightly worse,
insensitive to every structural knob — is what objective mismatch looks like. The
−1.66 was the extreme draw of a small negative effect, not a separate phenomenon
with its own mechanism.

**Two predictions were made before their evals and both failed**, recorded because
they are what ruled the ceiling mechanism out: (i) `mlp_fc1` losing its ceiling at
k=5 was expected to cost accuracy → −0.10 (p=1.00); (ii) with `PCF=0` the
per-group cell was expected to lose again like the old −1.66 → −0.80 (p=0.47), a
tie.

**Consequence for the endpoint fix:** its contribution is still not isolated, but
for a different reason than the earlier draft claimed. Protection is out as the
explanation, so the remaining differences between `s32_glk` and the new cells are
the endpoint rule, qk `per_head`→`per_row`, and a 4-op vs 6-op table — and since
−1.66 and −0.80 are not significantly different, there may be nothing to explain.

## 5. What this settles

* The −1.66 pt in MP_REPORT.md §3 is **not** evidence that per-group < per-row,
  and its magnitude does not reproduce: −0.70 (p=0.48) with protection, −0.80
  (p=0.47) without. The row `per-group ladders | −1.66 | —` should read as one
  draw of a small negative effect, and its empty "metric repaired" column is now
  filled and is a tie.
* **The honest summary of the whole axis is a consistently negative, never
  significant, ~−0.75 pt penalty that no structural knob moves.** 4 of 4
  measurements negative; protection worth 0.10 pt of it; k=3 identical to k=5;
  7/96 groups keeping the ceiling identical to 57/144. Do not attribute it to a
  mechanism — the surrogate says the allocation got BETTER (§3), so the residual
  is the surrogate-to-top-1 gap, which is exactly where ρ=+0.13…+0.40 puts it.
* `GLK=0` remains the right default, but for a different reason than "group
  ladders lose": the entire axis is flat, so the simplest configuration wins on
  parsimony rather than on accuracy.
* The claim "subdividing a broken ranking buys nothing", previously inferred by
  analogy with the escape gate's sign reversal, is now measured directly: the
  subdivision improves the surrogate by 0.45% and top-1 by nothing.

## 6. Reproduce

```bash
bash cls/lane_l_denseladder.sh     # the three cells + ladder-occupancy dump
```

Cells: `results/fig1k/r32_pcD{,_g5,_g3}.json`, tables in `results/mp_calib/`.
