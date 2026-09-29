# The three orderings, and the two accounting bugs that were hiding them

**Date:** 2026-07-29 · **Nodes:** gl1805 + gl1807 (RTX PRO 6000 Blackwell) ·
**Model:** DINOv2 ViT-L/14 + linear head · **Data:** ImageNet val, seed-0 subset,
n=1000, all comparisons PAIRED (McNemar + bootstrap CI on the paired difference)

Target orderings:

1. `MP > uniform`
2. `hybrid MP > hybrid uniform`
3. `hybrid MP > MP`

**All three hold at eff-64, and (3) only becomes visible once the compute
accounting is fixed.** At eff-128 (1) is marginal and (2)/(3) are saturated; at
eff-192 none can be demonstrated by any allocator, for a reason that is not the
allocator's fault.

---

## 1. Why `+10% INT` MP looked worse than plain MP — it was never iso-compute

`--hybrid_config_json` routes 10% of `(op, block)` groups to INT7 and leaves the
remaining 90% **at the same stream length**. The masked groups' SC cycles are
simply not spent, and not redistributed. Measured off the precision traces:

| pair | true SC MACs | true SC cycles spent | top-1 |
|---|---:|---:|---:|
| `r32_pc` → `hyb_mppc_32` | −7.9% | **92.1%** | −0.10 |
| `r64_pc` → `hyb_mppc_64` | −10.0% | **87.9%** | −0.70 |

So the hybrid arm was charged for a budget it never spent, and `hyb_mppc_32`'s
−0.10 pt was really a **win on the energy axis** — the figure just had no axis on
which to show that.

**The fix** is to give the freed cycles back: raise the surviving groups' target
by `1/(1-f)` where `f` is the mask's TRUE MAC-weighted fraction. Note `f` is not
the group fraction — the 14 groups of `mask10_t32` are **7.9%** of MACs, not 10%,
because attention groups are cheap. That puts t32 at 32 → **35** and t64 at
64 → **71**, both still straddled by their ladders.

Result at eff-64, every cell trace-verified within ±0.7% of its reference's true
cycles:

| cell | recipe | true sl | true cyc(T) | top-1 |
|---|---|---:|---:|---:|
| `uni_32` | uniform SC | 32.00 | 2587.5 | 0.7830 |
| `hyb_uni_32iso` | uniform + 10% INT7, re-budgeted | 32.23 | 2606.0 | 0.8150 |
| `r32_pc` | MP + 2% protected channels | 31.81 | 2571.9 | 0.8280 |
| **`hyb_mppc_32iso`** | **MP + PC + 10% INT7, re-budgeted** | **31.95** | **2583.7** | **0.8490** |

The three orderings:

| ordering | comparison | Δ | 95% CI | p |
|---|---|---:|---|---:|
| (1) MP > uniform | `r32_pc` vs `uni_32` | **+4.50** | [+2.20, +6.80] | 1e-4*** |
| (2) hyb MP > hyb uniform | `hyb_mppc_32iso` vs `hyb_uni_32iso` | **+3.40** | [+1.60, +5.20] | 4e-4*** |
| (3) hyb MP > MP | `hyb_mppc_32iso` vs `r32_pc` | **+2.10** | [+0.50, +3.70] | 0.017* |
| — same, NOT iso (the old cell) | `hyb_mppc_32` vs `r32_pc` | −0.10 | [−1.70, +1.50] | 1.00 |

Ordering (3) is entirely an artifact of the accounting: −0.10 (tie) before,
+2.10 (significant) after, with the *same mask and the same recipe*.

**But quote ordering (3) as +1.7, not +2.10.** Replicated over 4 calibration
seeds (n_calib=128, where the calibration-sample sd is ~1 pt — comparable to the
effect itself):

| calib seed | top-1 | Δ vs `r32_pc` | p |
|---|---:|---:|---:|
| 1234 (the cell above) | 0.8490 | +2.10 | 0.017* |
| 7 | 0.8500 | +2.20 | 0.009** |
| 23 | 0.8450 | +1.70 | 0.033* |
| 41 | 0.8370 | +0.90 | 0.34 n.s. |

**mean +1.73, sd 0.59, 4/4 positive**, across-seed 95% CI ≈ [+1.0, +2.4]. The
ORDERING is robust — every draw is positive — but the draw first measured was a
high one and one of four is not individually significant. Orderings (1) and (2)
are 4–7 pt effects and are not at risk from this; (3) is the one to state
carefully.

### Independent replication at n=4000

Fresh calibration at `EXCL=4000` (so the eval set stays disjoint from the
calibration pool) plus 4× the eval images — a new calibration draw AND a new
eval set, so this is a genuine independent replication, not a precision
touch-up. All three cells ISO-verified within ±0.8%:

| ordering | n=1000 | **n=4000, independent draw** |
|---|---|---|
| (1) MP > uniform | +4.50 [+2.20, +6.80] | **+3.65 [+2.57, +4.72]**, p<1e-4 |
| (2) hyb MP > hyb uniform | +3.40 [+1.60, +5.20] | **+2.97 [+2.08, +3.90]**, p<1e-4 |
| (3) hyb MP > MP | +2.10 [+0.50, +3.70] | **+2.08 [+1.23, +2.97]**, p<1e-4 |

CI widths roughly halved as expected, and every ordering survives at p<1e-4.
Ordering (3) lands at +2.08 on an independent draw, bracketed by the n=1000
4-seed mean of +1.73 — so **+1.7 to +2.1 is the defensible range**, and the
ordering itself is no longer in question.

**Headline consequence.** `hyb_mppc_32iso` at 0.8490 is **−0.50 pt from FP
(0.8540), n.s. (p=0.57)**, and −0.50 from INT6 at the same 64 levels. The
SC-vs-INT gap at eff-64 that the previous report called an unavoidable 2.6 pt
(7.1 pt for uniform SC) is now **0.5 pt and statistically indistinguishable from
lossless**.

## 2. What is actually producing the hybrid gain — four controls

### (a) It is NOT just "the SC groups got more cycles"

`r35_pc_overbudget` is the same table at the same raised target 35 with NO mask,
so nothing pays for the raise:

| cell | true cyc(T) | vs `r32_pc` | top-1 | Δ | p |
|---|---:|---:|---:|---:|---:|
| `r32_pc` | 2571.9 | — | 0.8280 | — | — |
| `r35_pc_overbudget` | 2802.0 | **+8.9% OVER** | 0.8340 | +0.60 | 0.56 n.s. |
| `hyb_mppc_32iso` | 2583.7 | +0.5% ISO | 0.8490 | **+2.10** | 0.017* |

**Simply buying 9% more SC cycles gets +0.60 and is not significant. Earning the
same stream length by moving 7.9% of MACs to INT gets +2.10 for free.** The iso
hybrid also beats the over-budget unmasked cell by +1.50 (p=0.082) while spending
8.4% less. So the gain is the INT format on those groups, not the extra cycles.

### (b) It is NOT just "the masked groups got more levels"

At target 35 the SC groups carry 2×35 = 70 levels while INT7 carries 128, so the
masked 8% had ~1.8× everyone else's resolution. Re-running the same mask at
**INT6 = 64 levels, which is BELOW the SC arm's 70**:

| arm | INT7 (128 lv) | INT6 (64 lv) | cost of level parity |
|---|---:|---:|---:|
| MP (`vs r32_pc`) | +2.10* | **+1.70** (p=0.064) | −0.40 |
| uniform (`vs uni_32`) | +3.20** | **+2.30*** (p=0.042) | −0.90 |

Both INT6 arms verify ISO. ~80% of the MP-arm gain survives at a level
*deficit*, so this is a **datapath** result: on those groups a uniform
fixed-point grid beats a stochastic stream of equal-or-more levels.

### (c) The mask RANKING does matter at iso — reversing the earlier conclusion

`maskrand10_t32` is matched to `mask10_t32` in group count and per-op
composition, so its MAC fraction is identical (0.0792) and the same target 35
makes both iso. Only *which* blocks within each op differ.

| arm | random mask | ranked mask | Δ (ranking) | p |
|---|---:|---:|---:|---:|
| MP | 0.8340 | 0.8490 | **+1.50** | 0.063 |
| uniform | 0.8100 | 0.8150 | +0.50 | 0.63 n.s. |

And a random mask alone is worth only +0.60 (n.s.) over no mask on the MP arm.
So on the MP arm **most of the +2.10 is attributable to the ranking**, not to the
dose — the opposite of the pre-iso finding ("−0.30, p=0.82, the probe bought
nothing"). That earlier test was run where both arms were 8% under budget and had
little to resolve. At p=0.063 this is **suggestive, not established**; it is the
single most valuable thing to confirm with more eval.

### (d) Dose: the uniform arm wants more INT, the MP arm peaks at 10%

All cells re-budgeted to their own iso target (5% → 33, 10% → 35, 20% → 39) and
ISO-verified:

| INT fraction | uniform arm | Δ vs `uni_32` | MP arm | Δ vs `r32_pc` |
|---|---:|---:|---:|---:|
| 0% | 0.7830 | — | 0.8280 | — |
| 5% | 0.8080 | +2.50* | 0.8370 | +0.90 n.s. |
| 10% | 0.8150 | +3.20** | **0.8490** | **+2.10*** |
| 20% | 0.8320 | +4.90*** | 0.8420 | +1.40 n.s. |
| 100% (= INT7) | 0.8520 | — | — | — |

The uniform arm rises monotonically toward pure INT7 — at eff-64 on this network,
the more you move off SC the better, which is the pre-existing "INT beats SC at
the aggressive end" caveat showing up as a dose curve. The MP arm instead
**peaks at 10%**: MP has already repaired the fragile groups, so masking more of
them starts removing work MP was handling well, and the ladder gets stretched
(target 39 against a 64-rung top).

## 3. The precision trace over-counts attention MACs by 3.4–3.8×

Found while verifying the above, and it had to be fixed first, because §1 is
entirely an iso-compute argument.

`model/sc_vit.py::_padded_level_gather` pads every `(b, h)` slice out to the
largest per-slice row count at each rung (`M = counts.max()`) so one batched
`sc_matmul` can serve all BH slices instead of a BH-long Python loop. Pad slots
repeat an already-selected row — a zero row would have a degenerate `per_row`
range — and their outputs are discarded by `valid`. Correct results, but **the
kernel ran on them and the trace records the padded shape.**

On `r32_pc`: `qk` MACs come out **3.42×** and `av` **3.76×** the uniform cell's,
every linear matches exactly, raw total SC cycles inflate **+10.7%**, and
`qk`+`av` appear as ~13% of MACs against their true 4%. (It is also why MP cells
run at ~5.4 img/s against uniform's 8.2.)

Padding is a GPU-batching artifact, not an architectural cost: a per-row SC
dispatcher spends exactly `Σ_rows sl(row)`. Weighting each `(op, block)` by its
TRUE MAC count and taking only the stream length from the trace:

| accounting | `r32_pc` realized | vs `uni_32` = 32.00 |
|---|---:|---|
| raw trace MACs | 32.09 | +0.3%, but total cycles +10.7% |
| **true MACs** | **31.81** | **−0.6% — iso, and slightly conservative** |

**The +4.50 headline survives** — but only the corrected accounting shows it was
on budget rather than 10% over.

Both corrections are in `cls/iso_check.py` and `cls/collect_figure.py`
(`--mac-ref`). The collector additionally divides by the FULL model MAC count, so
a masked cell lands at its real cost and **equal x on the figure means
iso-compute** — the old figure placed `hyb_mppc_32` at 64 eff-levels when its
true cost was 58.6.

## 4. Why eff-192 MP "performs so badly" — it doesn't, it is saturated

Not an allocator failure. There is almost nothing left to win:

| budget | uniform SC | FP | headroom | disc/1k | paired CI half | n to resolve |
|---|---:|---:|---:|---:|---:|---:|
| eff-64 | 0.7830 | 0.8540 | **7.1 pt** | 139 | 2.25 pt | **402** |
| eff-128 | 0.8360 | 0.8540 | 1.8 pt | 52 | 1.40 pt | 2,420 |
| eff-192 | 0.8500 | 0.8540 | **0.4 pt** | 26 | 1.00 pt | **25,000** |
| eff-256 | 0.8500 | 0.8540 | 0.4 pt | 22 | 0.90 pt | 20,250 |

`headroom` is the most *any* allocator can win at that budget. At eff-192 it is
0.4 pt and calling an effect that size needs ~**25,000 images**, 25× the current
eval. Every eff-192 number measured (−0.4 to +0.3) is inside noise. Ordering (1)
is therefore not demonstrable at eff-192 by any method — because uniform SC is
*already* at FP there. This retroactively justifies the abandoned 50k plan: the
mild end was never resolvable at n=1000.

### "MP@192 is below MP@128, and below uniform@192" — it is the PROTECTION

Both are correct readings of the old figure and both are ties, but the cause is
specific. Replicated over 4 calibration seeds per cell:

| budget | uniform | MP alone | MP + 2% PC | MP−uni | PC increment |
|---|---:|---:|---:|---:|---:|
| eff-64 | 78.30 | 78.10 | **82.62** ±0.29 | −0.20 | **+4.52** |
| eff-128 | 83.60 | **84.72** ±0.13 | 84.88 ±0.49 | +1.12 | +0.15 |
| eff-192 | 85.00 | **85.22** ±0.10 | 84.88 ±0.38 | +0.22 | **−0.35** |

* **MP@192 vs MP@128:** MP *alone* goes 84.72 → 85.22, **+0.50, 4/4 seeds
  positive — monotone**. MP+PC is 84.88 → 84.88, exactly flat. Neither is worse;
  the old figure plotted the MP+PC arm, hence the flat look.
* **MP@192 vs uniform@192:** MP *alone* is **+0.22 ABOVE**, 4/4 seeds positive
  (+0.30/+0.20/+0.10/+0.30). The protection is what pushes the arm under
  (−0.35, 3/4 seeds negative). None individually significant — eff-192 has 0.4 pt
  of headroom, so nothing there can be.

**Protection's sign flips with budget: +4.52 → +0.15 → −0.35.**

#### Two candidate mechanisms, both tested, both wrong on their own

**(i) Not the protect-stream ratio.** `--protect-channel-stoc-len` defaults to the
ladder ceiling and resolves correctly everywhere (verified on 8 tables: t96
protects at 128, not 64). But that ceiling is the hard grid limit
`2**(sc_prec-1) = 128` under `--halve`, so the ratio of protected stream length to
average budget collapses: 2.04× at eff-64 and eff-128, but only **1.34× at
eff-192** — protection can barely out-run the bulk there while still paying full
price. Plausible, and testable by holding the budget and dialling the ratio by
hand:

| eff-64, budget held at 32 | protect sl | ratio | PC increment |
|---|---:|---:|---:|
| `r32_pc_r137` | 43 | 1.35× | **+3.40** (p=0.0017) |
| `r32_pc_r153` | 48 | 1.51× | +3.40 (p=0.0019) |
| `r32_pc` | 64 | 2.04× | **+4.70** (p<1e-4) |

**Verdict: falsified as the explanation.** At exactly eff-192's own ratio (1.35×),
protection at eff-64 is still **+3.40**, not −0.35. The ratio is worth −1.30 pt of
the −4.87 total swing, i.e. **~27%**; headroom carries the rest. (It is also
threshold-like: 1.35× and 1.51× are identical, and only approaching 2× buys the
last 1.3 pt.) The same sweep at eff-128 is flat noise (−0.40/+0.10/−0.10),
consistent with protection having nothing to do there at any ratio.

**(ii) Not the ρ repair either — and this corrects the earlier mechanism claim.**
`MP_REPORT`/`NOTES_20260729` explained protection's +3.23 as buying twice: the
outlier channels get precision AND the row metric the allocator ranks on becomes
informative (`mlp_fc2` ρ −0.105 → +0.231). Measured across budgets:

| table | protection | `mlp_fc2` ρ |
|---|---|---:|
| `m96_base` (eff-192) | none | **−0.100** |
| `m96_pc` (eff-192) | 2% | **+0.270** |
| `m64_base` (eff-128) | none | −0.090 |
| `m64_pc` (eff-128) | 2% | +0.237 |
| `r32_pc` (eff-64) | 2% | +0.214 |

**Protection repairs ρ just as well at eff-192 as at eff-64.** The ρ repair is
therefore **budget-INDEPENDENT and cannot explain a +4.52 vs −0.35 difference** —
it is necessary but not sufficient. What varies with budget is whether there is
any headroom for a repaired ranking to cash in. At eff-192 the ranking is fixed,
there is nothing to buy with it, and the protected channels still cost budget.

Corollary worth recording because it kills an attractive idea: **ρ cannot be used
as a calibration-time switch for enabling protection.** It reads −0.09…−0.10
without protection at *every* budget, so it would say "enable" everywhere.

#### Protection also costs reproducibility

| | seed sd, eff-128 / eff-192 |
|---|---|
| MP alone | 0.13 / 0.10 pt |
| MP + 2% PC | **0.49 / 0.38 pt** |

~4× the calibration variance, because the protected channel set is itself
estimated from 128 calibration images (per-channel activation amax). So the
eff-128 "indifference" (+0.15) should resolve toward MP alone: same accuracy,
4× the stability.

#### Recipe, and an honest caveat about it

| budget | recipe | why |
|---|---|---|
| eff-64 | **MP + 2% PC** | +4.52, 4/4 seeds |
| eff-128 | **MP alone** | PC worth +0.15 and costs 4× variance |
| eff-192 | **MP alone** | PC is −0.35; MP alone is the arm that sits above uniform |

Mechanism statement: protection pays only where the stream is short enough that
outlier-driven scale coarseness dominates. **Caveat: the threshold — where to
switch PC off — was read off the same eval set the figure reports.** The ρ
diagnostic cannot supply it (above), so an independent validation set is the
honest way to fix the switch point, and that has not been run.

### The allocator misallocation at eff-192 was real, and irrelevant

The diagnosis was half right. Every winning t32 cell ran `CLW=macs`; t64 and t96
never did (they kept the `uniform` default from a 2026-07-25 recommendation that
predates per_row qk, attention in the table, and protected channels).
`CLW=macs` fixed the allocation exactly as predicted:

| op | %MACs | `uni_96` | `r96_base` (W_g=uniform) | `m96_base` (W_g=macs) | Δ |
|---|---:|---:|---:|---:|---:|
| `mlp_fc1` | 32% | 96.0 | **64.8** | **73.4** | +8.6 |
| `out_proj` | 8% | 96.0 | **127.1** | 103.6 | −23.5 |
| `qk` | 2% | 96.0 | 115.0 | 85.2 | −29.8 |

The cheap `out_proj` was pinned at the 127 ladder ceiling and `mlp_fc1` (32% of
MACs) starved to 64.8 cycles; `CLW=macs` un-pinned it and returned 8.6 cycles.
**And top-1 did not move**: `m96_base` 0.8530 vs `r96_base` 0.8510 (n.s.),
`m96_pc` 0.8480 vs `r96_pc` 0.8460 (p=0.86); same at eff-128 (`m64_pc` 0.8480 vs
`r64_pc` 0.8520, p=0.56).

**The allocation defect was real, and fixing it bought nothing, because at that
budget there was nothing to buy.** The mechanism prediction held; the consequence
prediction did not. `CLW=macs` is still worth adopting at the mild budgets for a
different reason — it makes ONE recipe (`CLW=macs, PCF=0.02`) correct across all
three budgets, which the figure needs in order not to be per-budget
cherry-picking.

## 5. Where each ordering stands, by budget

| budget | (1) MP > uni | (2) hyb MP > hyb uni | (3) hyb MP > MP |
|---|---|---|---|
| **eff-64** | **+4.50***, p=1e-4 | **+3.40***, p=4e-4 | **+1.7 mean (4/4 seeds), +2.10 on the reported draw** |
| eff-128 | +1.2…+1.6, one of four cells reaches p<0.05 | +0.1, saturated | +0.0, saturated |
| eff-192 | 0.4 pt headroom — not resolvable at any recipe | no mask built | no mask built |

At eff-128 the iso hybrid gives `hyb_mppc_64iso` 0.8480 vs `m64_pc` 0.8480 —
exactly nothing, consistent with 1.8 pt of headroom already ~70% consumed by
uniform SC.

### Why the hybrid-MP curve DIPS from eff-64 to eff-128 on the figure

`hyb_mppc_32iso` 0.8490 (eff-64) against `hyb_mppc_64iso` 0.8480 (eff-128) reads
as "double the compute, slightly worse". Three separate causes, none a real
regression:

1. **It is a tie.** Paired: −0.10 pt, CI [−1.60, +1.40], **p=1.00**, 59
   discordant. Both sit within 0.6 pt of FP, so there is no headroom to rise into.
2. **The eff-64 point plotted is a high calibration draw.** Over 4 seeds that
   cell means 0.8453 (sd 0.59); the plotted draw is 0.8490, **+0.37 above its own
   mean**. Against the seed-mean, eff-64 → eff-128 is **+0.27 — monotone again**.
   The eff-128 point had only one draw, so it never got the same treatment.
3. **The eff-128 hybrid is still 3.1% UNDER budget.** The `1/(1-f)` raise to 71
   under-shot: realized true sl 67.79 against `m64_pc`'s 63.05, i.e. 4942.3T vs
   5098.4T cycles. The **uniform** t64 arm landed at +0.0%, so this is specific to
   the MP arm.

Cause 3 is a limitation of the rule itself, worth stating: **`1/(1-f)` assumes the
masked groups were allocated at the average stream length.** That is exact for a
uniform arm and false for an MP one — the t64 mask happened to remove groups whose
MP allocation was ABOVE average, so dropping them pulled the mean down further
than the raise pushed it up. At t32 the same rule landed at +0.5% by luck of which
groups the mask took. **The general fix is to iterate the target against the
trace-measured cost rather than trusting the closed form**; here that gives
71 × 1.0316 = **73**.

**Confirmed.** `hyb_mppc_64iso2` (target 73, 4 seeds) lands at **+0.0% ISO**
against `m64_pc` — realized true sl 63.06 vs 63.05:

| hybrid MP | 4 calibration seeds | mean | sd |
|---|---|---:|---:|
| eff-64 (target 35) | 0.8490 / 0.8500 / 0.8450 / 0.8370 | **0.8453** | 0.59 |
| eff-128 (target **73**) | 0.8550 / 0.8490 / 0.8470 / 0.8480 | **0.8497** | 0.36 |

**eff-64 → eff-128 is +0.45 on seed-means: monotone.** The dip was an artifact of
plotting one high draw against one under-budget draw.

And the eff-128 hybrid over its unmasked reference is +0.70 / +0.10 / −0.10 /
+0.00 across those seeds — **mean +0.18, none significant.** So ordering (3) is
genuinely saturated at eff-128, now established on 4 draws rather than inferred
from one.

## 6. Caveats

* **Iso-SC-cycle counts the INT work as free**, the optimistic bound for a hybrid
  arm. Quantified model-free: the +2.10 is fully eaten only if one INT7 MAC costs
  **158 SC cycles**; at 7 (bit-serial) the net is +1.98, at 16 it is +1.87. The
  repo has **no energy model**, so that break-even is offered as a conversion for
  the reader, not as an SC-vs-INT energy claim.
* **The dose curve monotonically favours INT on the uniform arm**, so this axis
  cannot be used to select the hybrid fraction, and at eff-64 pure INT7 (0.8520)
  still beats every SC configuration measured. The hybrid claim is "given an SC
  datapath, a small INT escape hatch buys a lot at equal SC energy", not "SC beats
  INT". The SC-favourable regime remains eff-192, where SC matches INT8 at 25%
  fewer levels on a budget fixed-point cannot express.
* Ordering (3)'s ranking attribution (§2c) is at p=0.063. Suggestive only.
* eff-192 conclusions are bounded by headroom, not method. Do not read those rows
  as evidence against MP.
* Marginal Wilson intervals at n=1000 are ~±2.4 pt and must not be quoted against
  these paired results.

## Files added / changed

| path | what |
|---|---|
| `cls/iso_check.py` | **new** — true-MAC compute accounting; `--pairs` verifies iso, `--int-cycle-cost` prices INT |
| `cls/lane_a_isohybrid.sh` | **new** — the re-budgeted hybrid cells |
| `cls/lane_b_clwmacs.sh` | **new** — `CLW=macs` at t64/t96 |
| `cls/lane_d_seeds.sh` | **new** — 4-seed replication of the headline |
| `cls/lane_e_dose_control.sh` | **new** — ranked-vs-random control and 5/10/20% dose, at iso |
| `cls/lane_f_power.sh` | **new** — independent n=4000 replication (fresh tables at `EXCL=4000`) |
| `cls/lane_g_levelmatch.sh` | **new** — level-matched INT6 control |
| `cls/lane_h_decomp.sh` | **new** — MP at the raised target, unmasked (over-budget probe) |
| `cls/finalize_iso.sh` | **new** — regenerates every table above in one pass |
| `cls/collect_figure.py` | true-MAC x axis, charged against the full model |
| `cls/mp_cell2.sh` | `EXCL` env knob for the calibration/eval exclusion |
| `cls/lane_i_eff128fix.sh` | **new** — eff-128 hybrid at the iterated target 73, 4 seeds |
| `figs/plot_figure.py` | corrected axis; seed-mean points; **error bars are now calibration spread, not marginal Wilson** |
| `results/hybrid/mask10_t32_int6.json` | **new** — level-matched control mask |
| `CLAUDE.md` | both accounting bugs documented |
