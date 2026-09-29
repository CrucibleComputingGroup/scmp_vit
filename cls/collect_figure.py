#!/usr/bin/env python3
"""Collect every cell for the accuracy-vs-precision figure onto one levels axis.

Classifies each eval JSON into an arm from the fields ``cls/eval.py`` records,
so no side manifest can drift out of sync with the runs:

    fp                                  -> fp
    mode=int                            -> int_symm / int_asymm
    mode=sc, mp=False, no hybrid        -> sc          (uniform SC)
    mode=sc, mp=True,  no hybrid        -> mp          (per-row MP SC)
    mode=sc, mp=False, hybrid           -> sc_hyb      (uniform SC + INT mask)
    mode=sc, mp=True,  hybrid           -> mp_hyb      (MP SC + INT mask)

x is the REALIZED MAC-weighted average stream length from the precision trace,
never the nominal budget: the table only predicts the average, and a cell may
only be compared with one it actually matches in compute.

    python cls/collect_figure.py --out results/figure/summary_all.csv \
        results/int_vs_sc results/mp_eval results/hybrid_eval

CAVEAT the collector cannot enforce: hybrid cells move work OFF the SC budget
into INT, so their x (an SC-only average) understates total compute. A hybrid
cell is comparable to another cell carrying the SAME mask, and to a non-hybrid
cell only on the energy axis. Rows carry ``hybrid_frac`` so a plot can keep the
arms separate.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os


def wilson_halfwidth(p, n, z=1.96):
    if not n:
        return float("nan")
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c + half - p, p - (c - half))


def realized_avg(trace_path, ref=None, ref_total=None):
    """SC cost of one cell, in cycles per MAC. Returns (x, per_op, share, sc_only).

    Two corrections over a plain average of the trace's own numbers, both of
    which move cells on this figure's x axis:

    1. TRUE vs PADDED MACs. ``model/sc_vit.py::_padded_level_gather`` pads every
       (b, h) slice out to the largest per-slice row count at each rung so one
       batched kernel serves all BH slices. Pad slots repeat an already-selected
       row and are masked out of the result, but the kernel ran on them and the
       trace records the padded shape. Measured on ``r32_pc``: qk MACs come out
       3.42x and av 3.76x the uniform cell's, inflating raw total cycles +10.7%.
       Padding is a GPU-batching artifact -- a per-row SC dispatcher spends
       exactly ``sum_rows sl(row)`` -- so each (op, block) is weighted by its
       TRUE MAC count from ``ref`` and the trace supplies only the stream length.

    2. HYBRID CELLS ARE CHARGED AGAINST THE WHOLE MODEL. A masked group makes no
       ``sc_matmul`` call, so it is ABSENT from the trace and an SC-only average
       silently renormalizes over a smaller pool: ``hyb_mppc_32`` and ``r32_pc``
       both report ~32 while the former spent 7.9% fewer cycles. Dividing by the
       FULL model MAC count (``ref_total``) instead puts a cell at its real cost,
       so two cells at the same x are genuinely iso-compute and a mask that buys
       its INT groups with SC cycles it never spent lands to the LEFT.

    The INT work itself is counted as free, which is the optimistic bound for a
    hybrid arm; ``cls/iso_check.py --int-cycle-cost`` prices it.
    """
    with open(trace_path) as f:
        groups = json.load(f).get("groups", [])
    acc, per_op_acc = {}, {}
    for g in groups:
        k = (g.get("op"), g.get("block"))
        e = acc.setdefault(k, [0.0, 0.0])
        e[0] += g.get("macs", 0) * (g.get("stoc_len") or 0)
        e[1] += g.get("macs", 0)
    cycles, macs_present = 0.0, 0.0
    for k, (num, den) in acc.items():
        if den <= 0:
            continue
        w = ref.get(k, den) if ref else den
        sl = num / den
        cycles += w * sl
        macs_present += w
        e = per_op_acc.setdefault(k[0], [0.0, 0.0])
        e[0] += w * sl
        e[1] += w
    total = ref_total if ref_total else macs_present
    return (cycles / total if total else float("nan"),
            {k: (v[0] / v[1] if v[1] else float("nan"))
             for k, v in per_op_acc.items()},
            {k: (v[1] / total if total else 0.0) for k, v in per_op_acc.items()},
            cycles / macs_present if macs_present else float("nan"))


def hybrid_int_levels(mask_path, ref, ref_total):
    """MAC-weighted levels contributed by the INT-masked groups.

    Returns ``f * 2**int_bits`` where ``f`` is the TRUE MAC-weighted fraction the
    mask removes from SC -- not the group fraction, which differs (attention
    groups are cheap: 10% of groups is 7.9% of MACs at t32, 10.9% at t128).

    Added to the SC-only average, this puts a hybrid cell at the level count it
    actually carries. With the INT width matched to the SC level count the two
    contributions reconstruct the unmasked cell's x exactly: at eff-256,
    0.891*256 + 0.109*256 = 256.
    """
    if not os.path.exists(mask_path):
        print(f"  WARN {mask_path}: missing, hybrid cell placed at its SC-only x")
        return 0.0
    with open(mask_path) as f:
        m = json.load(f)
    int_levels = 2 ** int(m.get("int_bits", 7))
    masked = 0.0
    for op, blocks in m.get("schedule", {}).items():
        for blk, kind in enumerate(blocks):
            if kind == "int":
                masked += ref.get((op, blk), 0)
    return (masked / ref_total) * int_levels if ref_total else 0.0


def true_mac_map(trace_path):
    """{(op, block): macs} from an UNPADDED uniform cell, plus the total."""
    with open(trace_path) as f:
        groups = json.load(f).get("groups", [])
    out = {}
    for g in groups:
        k = (g.get("op"), g.get("block"))
        out[k] = out.get(k, 0) + g.get("macs", 0)
    return out, sum(out.values())


def classify(d):
    mode = d.get("mode")
    if mode == "fp":
        return "fp"
    if mode == "int":
        return "int_symm" if d.get("int_sym") else "int_asymm"
    if mode != "sc":
        return None
    hyb = d.get("hybrid_frac") is not None
    mp = bool(d.get("mp"))
    return ("mp_hyb" if (mp and hyb) else "sc_hyb" if hyb
            else "mp" if mp else "sc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results/figure/summary_all.csv")
    ap.add_argument("--only", default="",
                    help="Comma-separated cell tags to keep. The MP arm has "
                         "many configurations; the figure must show ONE "
                         "recipe across budgets, not the per-budget winner — "
                         "picking the best cell per budget on the same 1000 "
                         "images the figure reports is selection on the test "
                         "set. Ablations stay in the notes.")
    ap.add_argument("--mac-ref", default="results/fig1k/uni_32_trace.json",
                    help="trace of a uniform (unpadded, unmasked) cell, used for "
                         "TRUE per-(op, block) MAC counts and the full-model "
                         "total. Any uniform cell works — MAC counts are "
                         "architecture-determined and identical across budgets.")
    args = ap.parse_args()
    keep_only = {t.strip() for t in args.only.split(",") if t.strip()}

    if not os.path.exists(args.mac_ref):
        raise SystemExit(f"--mac-ref {args.mac_ref}: not found. Without it, MP "
                         "cells are placed at padded-MAC x values and hybrid "
                         "cells at an SC-only average; both are wrong.")
    REF, REF_TOTAL = true_mac_map(args.mac_ref)

    rows = []
    for dirname in args.dirs:
        for p in sorted(glob.glob(os.path.join(dirname, "*.json"))):
            base = os.path.basename(p)
            if base.endswith("_trace.json") or base.startswith("summary"):
                continue
            with open(p) as f:
                d = json.load(f)
            if "top1" not in d:
                continue
            arm = classify(d)
            if arm is None:
                continue
            tag = base[:-5]
            if keep_only and tag not in keep_only:
                continue
            levels = d.get("levels")
            realized = ""
            if arm in ("sc", "mp", "sc_hyb", "mp_hyb"):
                tr = p[:-5] + "_trace.json"
                if not os.path.exists(tr):
                    print(f"  SKIP {tag}: no trace, cannot place on the x axis")
                    continue
                avg, per_op, share, sc_only = realized_avg(tr, REF, REF_TOTAL)
                realized = avg
                levels = 2 * avg if d.get("halve") else avg
                # A masked group is ABSENT from the trace, so the SC average
                # above counts it as ZERO levels. It is not zero -- it runs INT
                # at 2**int_bits levels. On a "representable levels" axis that
                # omission drags a hybrid cell left by f * 2**int_bits and makes
                # it look like a lower-precision cell rather than an equal-
                # precision one, which is the whole point of matching the INT
                # width to the SC level count (INT6 at eff-64 = 64 = 2*32).
                #
                # `realized_avg_sl` keeps the SC-only number, because THAT is the
                # cost. levels = precision, realized = cost; they differ for a
                # hybrid cell and the figure needs the first one.
                if d.get("hybrid_config"):
                    levels += hybrid_int_levels(d["hybrid_config"], REF, REF_TOTAL)
            rows.append(dict(
                name=tag, mode=arm, levels=levels, realized_avg_sl=realized,
                n=d.get("n"), top1=d.get("top1"), top5=d.get("top5"),
                img_s=d.get("img_per_s"),
                sym=d.get("int_sym"), hybrid_frac=d.get("hybrid_frac"),
                qk=d.get("qk_granularity"),
                per_op=(per_op if realized != "" else {}),
                share=(share if realized != "" else {})))

    if not rows:
        print("no cells found")
        return
    rows.sort(key=lambda r: (r["mode"], -(r["levels"] or 0)))

    hdr = (f'{"cell":<26}{"arm":<9}{"eff.lv":>8}{"realized":>10}'
           f'{"n":>7}{"top1":>9}{"±95%":>8}{"qk":>10}{"mask":>7}')
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        hw = wilson_halfwidth(r["top1"], r["n"] or 0)
        lv = f'{r["levels"]:.1f}' if r["levels"] is not None else "-"
        rz = f'{r["realized_avg_sl"]:.2f}' if r["realized_avg_sl"] != "" else "-"
        mk = f'{100 * r["hybrid_frac"]:.0f}%' if r["hybrid_frac"] else "-"
        print(f'{r["name"]:<26}{r["mode"]:<9}{lv:>8}{rz:>10}{r["n"] or 0:>7}'
              f'{r["top1"]:>9.4f}{hw:>8.3f}{str(r["qk"] or "-"):>10}{mk:>7}')

    print("\nper-op realized avg stoc_len (share of SC MACs):")
    for r in rows:
        if not r["per_op"]:
            continue
        parts = " ".join(f'{op}={r["per_op"][op]:.0f}({100 * r["share"][op]:.0f}%)'
                         for op in sorted(r["per_op"], key=lambda o: -r["share"][o]))
        print(f'  {r["name"]:<26}{parts}')

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name", "mode", "levels", "realized_avg_sl", "n", "top1", "top5",
            "img_s", "sym", "hybrid_frac", "qk"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\nwrote {args.out}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
