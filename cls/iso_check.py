#!/usr/bin/env python3
"""Trace-verified compute accounting for cell-vs-cell comparisons.

The precision trace only records ops that actually ran on SC. A hybrid cell's
masked groups never call ``sc_matmul``, so they are ABSENT from its trace --
which makes the realized MAC-weighted average stream length (what
``collect_figure.py`` puts on the x axis) an average over a SMALLER pool. Two
cells can therefore both report "realized 32.0" while one spent 8% fewer total
cycles than the other.

That is exactly what happened to the 2026-07-29 hybrid arms: ``hyb_mppc_32``
spent 91.6% of ``r32_pc``'s SC cycles and ``hyb_mppc_64`` 87.9% of
``r64_pc``'s, and both were plotted at the same x as their unmasked reference.

The quantity that IS comparable is total SC cycles, ``sum_g macs_g * sl_g``.
This script reports it per cell and, for a pair, the mismatch in percent.

    python cls/iso_check.py results/fig1k --pairs uni_32:hyb_uni_32,r32_pc:hyb_mppc_32
    python cls/iso_check.py results/fig1k --cells uni_32,r32_pc,m64_pc

``--int-cycle-cost c`` additionally prices each masked MAC at ``c`` SC-cycle
equivalents, so a hybrid cell can be charged for the INT work it moved off the
SC budget instead of getting it free. The reference pool for "masked MACs" is
the paired unmasked cell, whose trace covers all groups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cell_cost(trace_path: Path):
    """(sc_macs, sc_cycles, realized_avg_sl) from one precision trace.

    RAW accounting: sums the trace's own MAC counts. For any cell with qk/av in
    the MP table this OVER-counts attention -- see ``true_macs``.
    """
    groups = json.loads(trace_path.read_text()).get("groups", [])
    macs = sum(g.get("macs", 0) for g in groups)
    cycles = sum(g.get("macs", 0) * (g.get("stoc_len") or 0) for g in groups)
    return macs, cycles, (cycles / macs if macs else float("nan"))


def true_macs(trace_path: Path) -> dict:
    """{(op, block): macs} from an UNPADDED reference cell (uniform SC).

    A uniform cell issues exactly one kernel call per (op, block), so its MAC
    counts are the architecture's true ones.
    """
    groups = json.loads(trace_path.read_text()).get("groups", [])
    out: dict = {}
    for g in groups:
        out[(g.get("op"), g.get("block"))] = out.get((g.get("op"), g.get("block")), 0) \
            + g.get("macs", 0)
    return out


def corrected_cost(trace_path: Path, ref: dict):
    """Cost with attention priced at TRUE MACs instead of padded ones.

    ``model/sc_vit.py::_padded_level_gather`` pads every (b, h) slice out to the
    LARGEST per-slice row count at each rung (``M = counts.max()``) so one
    batched ``sc_matmul`` can serve all BH slices instead of a BH-long Python
    loop. The pad slots repeat an already-selected row and their outputs are
    discarded by ``valid`` -- correct, but the kernel still ran on them, and the
    trace records the padded tensor shape. Measured on ``r32_pc``: qk MACs come
    out 3.42x and av 3.76x the uniform cell's, inflating total SC cycles by
    +10.7% even though the per-row ALLOCATION is on budget.

    Padding is a GPU-batching artifact, not an architectural cost -- a per-row
    SC dispatcher spends exactly ``sum_rows sl(row)``. So for any energy or
    iso-compute claim, weight each (op, block) by its TRUE MAC count and use the
    trace only for the stream length that group ran at. Linears are unaffected
    (their MACs match the uniform cell exactly); this only moves attention, which
    is 4% of true MACs -- small, but it is the difference between "MP overspent
    its budget by 10%" and "MP came in 0.6% under".

    Returns (true_macs_present, corrected_cycles, corrected_avg_sl, pad_ratio).
    """
    groups = json.loads(trace_path.read_text()).get("groups", [])
    # macs-weighted mean stream length WITHIN each (op, block)
    acc: dict = {}
    for g in groups:
        k = (g.get("op"), g.get("block"))
        e = acc.setdefault(k, [0.0, 0.0])
        e[0] += g.get("macs", 0) * (g.get("stoc_len") or 0)
        e[1] += g.get("macs", 0)
    macs_present = 0.0
    cycles = 0.0
    for k, (num, den) in acc.items():
        if den <= 0 or k not in ref:
            continue
        macs_present += ref[k]
        cycles += ref[k] * (num / den)
    raw_macs = sum(g.get("macs", 0) for g in groups)
    return (macs_present, cycles,
            (cycles / macs_present if macs_present else float("nan")),
            (raw_macs / macs_present if macs_present else float("nan")))


def load(d: Path, tag: str, ref: dict | None = None):
    p = d / f"{tag}_trace.json"
    if not p.exists():
        return None
    macs, cycles, avg = cell_cost(p)
    meta_p = d / f"{tag}.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    r = dict(tag=tag, macs=macs, cycles=cycles, avg=avg,
             top1=meta.get("top1"), masked=meta.get("hybrid_frac") is not None)
    if ref:
        tm, tc, ta, pad = corrected_cost(p, ref)
        r.update(true_macs=tm, true_cycles=tc, true_avg=ta, pad=pad)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--cells", default="", help="comma-separated tags to list")
    ap.add_argument("--pairs", default="",
                    help="comma-separated ref:cell pairs to check for iso-compute")
    ap.add_argument("--int-cycle-cost", type=float, default=0.0,
                    help="SC-cycle equivalents charged per masked (INT) MAC. "
                         "0 = INT counted free, the optimistic bound for a "
                         "hybrid arm.")
    ap.add_argument("--tol", type=float, default=1.5,
                    help="percent mismatch in total cycles still called iso")
    ap.add_argument("--mac-ref", default="uni_32",
                    help="uniform (unpadded) cell supplying TRUE per-(op, block) "
                         "MAC counts. Any uniform cell works; they are "
                         "architecture-determined and identical across budgets.")
    args = ap.parse_args()

    ref_trace = args.dir / f"{args.mac_ref}_trace.json"
    if not ref_trace.exists():
        raise SystemExit(f"--mac-ref {args.mac_ref}: no trace at {ref_trace}")
    REF = true_macs(ref_trace)

    if args.cells:
        rows = [load(args.dir, t.strip(), REF)
                for t in args.cells.split(",") if t.strip()]
        print(f"{'cell':<18}{'top-1':>8}{'true cyc(T)':>13}{'true sl':>9}"
              f"{'raw sl':>8}{'pad':>7}{'mask':>6}")
        print("-" * 69)
        for r in rows:
            if r is None:
                continue
            t1 = r['top1'] if r['top1'] is not None else float('nan')
            print(f"{r['tag']:<18}{t1:>8.4f}{r['true_cycles']/1e12:>13.1f}"
                  f"{r['true_avg']:>9.2f}{r['avg']:>8.2f}{r['pad']:>6.2f}x"
                  f"{('yes' if r['masked'] else '-'):>6}")
        print("\ntrue sl / true cyc weight each (op, block) by its unpadded MAC count; "
              "raw sl is the trace's own\nnumber. pad = raw MACs / true MACs — "
              "1.00x for uniform cells, ~1.10x once qk/av take MP\ntables, from "
              "the batched-dispatch padding in _padded_level_gather.")
        print()

    if args.pairs:
        c = args.int_cycle_cost
        print(f"iso-compute check on TRUE (unpadded) cycles   "
              f"(INT priced at {c:g} SC-cycle equivalents/MAC)")
        hdr = (f"{'reference':<16}{'cell':<18}{'ref cyc(T)':>12}"
               f"{'cell cyc(T)':>13}{'mismatch':>11}{'verdict':>9}")
        print(hdr); print("-" * len(hdr))
        for pair in args.pairs.split(","):
            pair = pair.strip()
            if not pair:
                continue
            ref_t, cell_t = pair.split(":")
            ref, cell = load(args.dir, ref_t, REF), load(args.dir, cell_t, REF)
            if ref is None or cell is None:
                miss = ref_t if ref is None else cell_t
                print(f"{ref_t:<16}{cell_t:<18}  missing trace for {miss}")
                continue
            # True MACs the mask moved off SC, priced at c cycles each.
            masked_macs = max(ref["true_macs"] - cell["true_macs"], 0)
            cell_total = cell["true_cycles"] + c * masked_macs
            pct = ((cell_total / ref["true_cycles"] - 1.0) * 100
                   if ref["true_cycles"] else float("nan"))
            verdict = "ISO" if abs(pct) <= args.tol else ("OVER" if pct > 0 else "UNDER")
            print(f"{ref_t:<16}{cell_t:<18}{ref['true_cycles']/1e12:>12.1f}"
                  f"{cell_total/1e12:>13.1f}{pct:>+10.1f}%{verdict:>9}")
        print(f"\nverdict is against a +-{args.tol:g}% tolerance. UNDER means the "
              "cell spent LESS than its reference,\nso any accuracy deficit it "
              "shows is at least partly bought back in energy.")


if __name__ == "__main__":
    main()
