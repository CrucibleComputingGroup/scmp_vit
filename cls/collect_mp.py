"""Collect MP cells onto the same levels axis as the INT-vs-SC sweep.

An MP cell's x-position is its **realized** MAC-weighted average stream length,
read from the precision trace — not the nominal budget. The calibration table
only predicts the average; the deployed per-row dispatch is what actually
spends cycles, and qk/av run uniform outside the MP table. Comparing an MP cell
against a uniform cell at the nominal budget when the realized one differs
would be comparing across compute budgets.

    python cls/collect_mp.py [results/mp_eval] [results/int_vs_sc/summary.csv]

Writes ``<mp_dir>/summary_mp.csv`` with the same columns the INT/SC collector
emits, plus the realized/target budget pair so any drift is visible.
"""
import csv
import glob
import json
import math
import os
import sys


def wilson_halfwidth(p, n, z=1.96):
    if not n:
        return float("nan")
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c + half - p, p - (c - half))


def realized_avg_stoc_len(trace_path):
    """MAC-weighted mean stream length over every SC matmul in the run."""
    with open(trace_path) as f:
        payload = json.load(f)
    groups = payload.get("groups", [])
    num = sum(g.get("macs", 0) * (g.get("stoc_len") or 0) for g in groups)
    den = sum(g.get("macs", 0) for g in groups)
    per_op = {}
    for g in groups:
        op = g.get("op")
        e = per_op.setdefault(op, [0.0, 0.0])
        e[0] += g.get("macs", 0) * (g.get("stoc_len") or 0)
        e[1] += g.get("macs", 0)
    return (num / den if den else float("nan"),
            {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in per_op.items()},
            {k: (v[1] / den if den else 0.0) for k, v in per_op.items()})


def main(mp_dir="results/mp_eval", ref_csv="results/int_vs_sc/summary.csv"):
    uniform = {}
    fp = None
    if os.path.exists(ref_csv):
        with open(ref_csv) as f:
            for r in csv.DictReader(f):
                if r["mode"] == "fp":
                    fp = float(r["top1"])
                elif r["mode"] == "sc" and r["levels"]:
                    uniform[int(r["levels"])] = float(r["top1"])

    rows = []
    for p in sorted(glob.glob(os.path.join(mp_dir, "*.json"))):
        if p.endswith("_trace.json"):
            continue
        tag = os.path.basename(p)[:-5]
        d = json.load(open(p))
        tr = p[:-5] + "_trace.json"
        if not os.path.exists(tr):
            print(f"  {tag}: no trace — skipped (cannot place on the x axis)")
            continue
        avg, per_op, share = realized_avg_stoc_len(tr)
        halve = bool(d.get("halve"))
        levels = 2 * avg if halve else avg
        rows.append(dict(
            name=tag, mode="mp", levels=levels, realized_avg_sl=avg,
            n=d.get("n"), top1=d.get("top1"), top5=d.get("top5"),
            img_s=d.get("img_per_s"), per_op=per_op, share=share))

    if not rows:
        print("no MP cells found")
        return

    rows.sort(key=lambda r: -r["levels"])
    hdr = (f'{"cell":<14}{"eff.levels":>11}{"realized":>10}{"top1":>9}'
           f'{"±95%":>8}{"vs uniform":>12}{"vs FP":>9}')
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        hw = wilson_halfwidth(r["top1"], r["n"] or 0)
        # Compare against the uniform cell at the NEAREST measured level, and
        # say which one, so a budget mismatch cannot hide inside the delta.
        near = (min(uniform, key=lambda L: abs(L - r["levels"]))
                if uniform else None)
        du = (f'{100 * (r["top1"] - uniform[near]):+.2f} @{near}'
              if near is not None else "-")
        dfp = f'{100 * (r["top1"] - fp):+.2f}' if fp is not None else "-"
        print(f'{r["name"]:<14}{r["levels"]:>11.1f}{r["realized_avg_sl"]:>10.2f}'
              f'{r["top1"]:>9.4f}{hw:>8.3f}{du:>12}{dfp:>9}')

    print("\nper-op realized avg stoc_len (share of MACs):")
    for r in rows:
        parts = " ".join(
            f'{op}={r["per_op"][op]:.0f}({100 * r["share"][op]:.0f}%)'
            for op in sorted(r["per_op"], key=lambda o: -r["share"][o]))
        print(f'  {r["name"]:<14}{parts}')

    out = os.path.join(mp_dir, "summary_mp.csv")
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["name", "mode", "levels", "realized_avg_sl", "n",
                    "top1", "top5", "img_s"])
        for r in rows:
            w.writerow([r["name"], "mp", f'{r["levels"]:.2f}',
                        f'{r["realized_avg_sl"]:.3f}', r["n"],
                        r["top1"], r["top5"], r["img_s"]])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
