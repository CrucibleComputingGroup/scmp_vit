# The three orderings across all three budgets — the complete 3×3

**Date:** 2026-07-29/30 · **Nodes:** gl1805 + gl1807 · DINOv2 ViT-L/14 + linear
head · ImageNet val, seed-0 subset, all comparisons PAIRED (McNemar exact +
bootstrap CI on the paired difference) · every cell trace-verified iso-compute.

Target orderings: **(1) MP > uniform · (2) hybrid MP > hybrid uniform ·
(3) hybrid MP > MP**, at eff-64 / eff-128 / eff-192.

> **This supersedes ISO_REPORT.md §5.** Two things changed: the eff-192 hybrid
> cells now EXIST (they were "no mask built"), and the eff-128 numbers were
> re-measured at n=4000, where ordering (1) SHRANK from +1.30 to +0.50. The
> eff-64 column is unchanged and is the only column that survives.

## 1. The answer

| budget | (1) MP > uni | (2) hyb MP > hyb uni | (3) hyb MP > MP |
|---|---|---|---|
| **eff-64** | **+4.50** p=1e-4 *** | **+3.40** p=4e-4 *** | **+2.10** p=0.017 * |
| eff-64, independent n=4000 | **+3.65** p<1e-4 *** | **+2.97** p<1e-4 *** | **+2.08** p<1e-4 *** |
| eff-128, n=1000 | +1.30 p=0.060 | +0.00 p=1.00 | −0.20 p=0.87 |
| eff-128, **n=4000** | **+0.50** p=0.198 | see §4 | see §4 |
| eff-192 | +0.30 p=0.72 | **−0.20** p=0.81 | **−0.10** p=1.00 |

**All three orderings hold, with 2–4.5 pt margins and an independent n=4000
replication, at eff-64 only.** At eff-128 all three are ties. At eff-192 two of
the three have NEGATIVE point estimates. Nothing above eff-64 is demonstrable,
and §3 shows that is a property of the benchmark rather than of the method.

Cells, for reference:

| budget | uniform | MP | hybrid uniform | hybrid MP |
|---|---|---|---|---|
| eff-64 | `uni_32` 0.7830 | `r32_pc` 0.8280 | `hyb_uni_32iso` 0.8150 | `hyb_mppc_32iso` 0.8490 |
| eff-128 | `uni_64` 0.8360 | `m64_base` 0.8490 | `hyb_uni_64iso` 0.8470 | `hyb_mp_64iso2` 0.8470 |
| eff-192 | `uni_96` 0.8500 | `m96_base` 0.8530 | `hyb_uni_96iso` **0.8540** | `hyb_mp_96iso` 0.8520 |

FP = 0.8540. INT6 = 0.8540, INT7 = INT8 = 0.8520.

## 2. eff-192: the two negative signs are structural, not a method failure

`hyb_uni_96iso` scores **0.8540 — exactly FP — while costing 0.34% LESS than
`uni_96`**. Ordering (2)'s comparator is already lossless, so (2) cannot be
positive at this budget by any allocator. This is stronger than the "0.4 pt of
headroom" argument in ISO_REPORT §4: it is not that the room is small, it is that
the arm MP has to beat is at the ceiling.

The mechanism is worth stating on its own: **at eff-192, a 10% INT7 escape hatch
alone lifts uniform SC from 0.8500 to FP.** Whatever MP was going to repair, a
much cheaper mechanism has already repaired it. This is the ISO_REPORT dose curve
("the uniform arm rises monotonically toward pure INT") in its extreme form.

Iso audit, so the negatives are not an accounting artifact — spread 0.37%:

| cell | true SC cycles | true avg sl | top-1 |
|---|---:|---:|---:|
| `uni_96` | 7762.4T | 96.00 | 0.8500 |
| `m96_base` | 7753.4T | 95.89 | 0.8530 |
| `hyb_uni_96iso` | 7736.2T | 107.00 | **0.8540** |
| `hyb_mp_96iso` | 7764.6T | 107.39 | 0.8520 |
| `hyb_mppc_96iso` | 7739.9T | 107.05 | 0.8470 |

## 3. The statistics lesson: n=1000 effects under ~1.5 pt do not survive

Ordering (1) at eff-128 was **+1.30 (p=0.060) at n=1000** and **+0.50
(p=0.198) at n=4000** on an independent draw with a fresh `EXCL=4000` table. The
interval tightened 2× and the effect halved.

**The 4-seed replications in ISO_REPORT do not protect against this.** All four
seeds score the SAME 1000 images, so they control calibration-sample noise and
not eval-subset bias. Going to n=4000 moved uniform +0.50 and MP −0.30 — both
toward the null.

Consequence, and it decides what is publishable: **in this project an effect
below ~1.5 pt measured at n=1000 is not a result, even at 4/4 positive seeds.**
The eff-64 column (2–4.5 pt, plus an independent n=4000 replication) is
unaffected. Every eff-128 and eff-192 number is in the discounted tier.

## 4. eff-128: which recipe, and why it does not matter

Two hybrid-MP arms exist at eff-128, differing only in protected channels:

| cell | PCF | target | top-1 | (2) vs `hyb_uni_64iso` | (3) vs `m64_base` |
|---|---:|---:|---:|---:|---:|
| `hyb_mp_64iso2` | 0 | 73 | 0.8470 | +0.00 (p=1.00) | −0.20 (p=0.87) |
| `hyb_mppc_64iso2` | 0.02 | 73 | 0.8550 | +0.80 | +0.60 |

The 0.8550 looks like protection being worth +0.80 on a masked arm against +0.15
on an unmasked one. **It is not.** That cell has 4 calibration seeds —
0.8550/0.8490/0.8470/0.8480, mean **0.8498**, sd 0.36 — and the 0.8550 draw sits
+0.52 pt above its own mean. On seed-means the PC increment is **+0.28 pt**, the
same order as the unmasked +0.15, and the single-draw paired test is +0.80
[−0.30, +1.90] p=0.23.

So the ISO_REPORT recipe (**PC on at eff-64, off above**) stands, and it stands
in the hybrid arm too. Either way eff-128's (2) and (3) are ties.

Protected channels at eff-192, both arms, confirming the sign does not flip back
once a mask is present:

| eff-192 | unmasked | + 10% INT mask |
|---|---:|---:|
| MP alone | 0.8530 | 0.8520 |
| MP + 2% PC | 0.8480 | 0.8470 |
| **PC increment** | **−0.50** (p=0.40) | **−0.50** (p=0.30) |

## 5. The `1/(1-f)` iso rule needs iteration, and now iterates itself

A masked group LEAVES the SC budget rather than shrinking it, so a hybrid cell
must be re-budgeted to `T/(1-f)` — with `f` the **MAC-weighted** fraction, not
the group fraction (attention groups are cheap: 10% of groups is 7.9% of MACs at
t32, 9.8% at t64, 10.6% at t96).

The closed form assumes the masked groups sat at the average stream length. That
is exact for a uniform arm and false for an MP one. Measured error of the closed
form, per budget:

| budget | `f` | closed-form target | realized vs unmasked ref | needed iteration? |
|---|---:|---:|---:|---|
| t32 | 0.0792 | 35 | +0.5% | no |
| t64 | 0.0983 | 71 | **−2.3%** | yes → **73** |
| t96 | 0.1058 | 107 | +0.14% | no |

`cls/sc_lib.sh::mp_hybrid_iso` now measures the realized cost off the trace and
re-calibrates at `T × 100/pct` when it is off by more than 1.5%. On a PCF=0 arm
against a different reference and with no hardcoded value, it independently
re-derived **73** — the target that had previously been found by hand for the
PCF=0.02 arm. Only t64 needs the second pass; whether a budget needs one depends
on whether its mask happened to remove groups allocated above or below average.

## 6. What to do about eff-128 and eff-192

Not a method problem, so not fixable by the allocator. The headroom is:

| budget | uniform SC | FP | headroom | n needed to resolve it |
|---|---:|---:|---:|---:|
| eff-64 | 0.7830 | 0.8540 | **7.1 pt** | 402 |
| eff-128 | 0.8360 | 0.8540 | 1.8 pt | 2,420 |
| eff-192 | 0.8500 | 0.8540 | **0.4 pt** | ~25,000 |

ImageNet val has 50k images total, so eff-192 is out of reach even in principle
once you account for needing several cells. **Getting all nine claims requires a
different operating point, not more compute** — the direct route is a backbone
with less redundancy (ViT-B/14 rather than ViT-L/14), which costs a new download,
a new FP anchor, and recalibrating every table. NOT STARTED, pending a decision.

## 7. Reproduce

```bash
bash cls/sweep_orderings.sh                  # all 3 budgets, n=1000
N=4000 bash cls/sweep_orderings.sh           # power run (needs EXCL=4000 tables)
bash cls/lane_k_eff192hybrid.sh              # the eff-192 hybrid arm from scratch
bash cls/lane_m_eff128power.sh               # eff-128 (2)/(3) at n=4000
```

`cls/sweep_orderings.sh`'s header documents every parameter and which measurement
fixed it. New files this round: `cls/sc_lib.sh`, `cls/sweep_orderings.sh`,
`cls/mask_mac_frac.py`, `cls/lane_k_eff192hybrid.sh`,
`cls/lane_l_denseladder.sh`, `cls/lane_m_eff128power.sh`,
`cls/chain_1807_sweep.sh`, `results/hybrid/mask{05,10,20}_t96.json`,
`LADDER_REPORT.md`.
