"""Compare objective/W_g variants of ONE sigma draw against uniform SC.

    python cls/collect_variants.py [TAG_PREFIX ...]

Reads the cells `cls/mp_variant_sweep.sh` wrote (`results/mp_eval/<tag>__<obj>_w<W_g>.json`
plus their traces) and the uniform-SC / FP reference from
`results/int_vs_sc/summary.csv`.

Two things this does that `collect_mp.py` does not:

* Groups by (tag, variant) and prints the objective x W_g grid, so the cells
  that share a sigma draw are read as one experiment rather than N independent
  points.
* Refuses to quote a `vs uniform` delta unless the MP cell's trace-measured
  realized budget matches the uniform cell's stream length to within
  `--tol` cycles. An iso-compute claim has to be checked, not assumed; qk/av run
  uniform outside the MP table and can drag the realized average off target.
"""
import csv
import glob
import json
import math
import os
import sys

# The pair each objective/W_g grid is read against.
CONTROL = "sigma_wuniform"          # == act_global, the shipped config
ORDER = ["sigma_wuniform", "sigma2_wuniform", "delta_sigma2_wuniform",
         "sigma_wmacs", "sigma2_wmacs", "delta_sigma2_wmacs"]


def wilson_halfwidth(p, n, z=1.96):
    if not n:
        return float("nan")
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c + half - p, p - (c - half))


def realized(trace_path):
    """MAC-weighted mean stream length + per-op breakdown over every SC matmul."""
    payload = json.load(open(trace_path))
    groups = payload.get("groups", [])
    num = sum(g.get("macs", 0) * (g.get("stoc_len") or 0) for g in groups)
    den = sum(g.get("macs", 0) for g in groups)
    per = {}
    for g in groups:
        e = per.setdefault(g.get("op"), [0.0, 0.0])
        e[0] += g.get("macs", 0) * (g.get("stoc_len") or 0)
        e[1] += g.get("macs", 0)
    return (num / den if den else float("nan"),
            {k: v[0] / v[1] for k, v in per.items() if v[1]},
            {k: v[1] / den for k, v in per.items()} if den else {})


def load_reference(path="results/int_vs_sc/summary.csv"):
    """{halved stream length -> uniform top-1}, plus the FP ceiling."""
    uniform, fp = {}, None
    if not os.path.exists(path):
        return uniform, fp
    for r in csv.DictReader(open(path)):
        if r["mode"] == "fp":
            fp = float(r["top1"])
        elif r["mode"] == "sc" and r.get("stoc_len"):
            uniform[float(r["stoc_len"])] = float(r["top1"])
    return uniform, fp


def main(*tags, mp_dir="results/mp_eval", tol=1.5):
    uniform, fp = load_reference()
    cells = {}
    for p in sorted(glob.glob(os.path.join(mp_dir, "*__*.json"))):
        if p.endswith("_trace.json") or "summary" in p:
            continue
        name = os.path.basename(p)[:-5]
        tag, _, variant = name.partition("__")
        if tags and tag not in tags:
            continue
        if variant not in ORDER:
            continue
        tr = p[:-5] + "_trace.json"
        if not os.path.exists(tr):
            print(f"  {name}: no trace — skipped (cannot verify iso-compute)")
            continue
        d = json.load(open(p))
        avg, per_op, share = realized(tr)
        cells.setdefault(tag, {})[variant] = dict(
            top1=d["top1"], top5=d["top5"], n=d["n"], realized=avg,
            halve=bool(d.get("halve")), per_op=per_op, share=share)

    if not cells:
        print("no variant cells found")
        return

    for tag in sorted(cells):
        grid = cells[tag]
        ctrl = grid.get(CONTROL)
        base_sl = ctrl["realized"] if ctrl else next(iter(grid.values()))["realized"]
        # Uniform comparator: the measured uniform cell at the same stream length.
        near = min(uniform, key=lambda L: abs(L - base_sl)) if uniform else None
        ok = near is not None and abs(near - base_sl) <= tol
        eff = 2 * base_sl if next(iter(grid.values()))["halve"] else base_sl
        print(f"\n=== {tag} | realized ~{base_sl:.1f} halved cycles "
              f"(eff {eff:.0f} levels) | n={next(iter(grid.values()))['n']} ===")
        if ok:
            print(f"    uniform SC @ stoc_len {near:g}: {uniform[near]:.4f}"
                  + (f"   FP: {fp:.4f}" if fp is not None else ""))
        else:
            print(f"    NO iso-compute uniform comparator within {tol} cycles of "
                  f"{base_sl:.1f} (nearest measured: {near}) — deltas vs uniform "
                  f"withheld.")

        hdr = (f'{"variant":<22}{"top1":>8}{"±95%":>8}{"realized":>10}'
               f'{"vs ctrl":>9}{"vs unif":>9}   per-op avg stoc_len')
        print(hdr)
        print("-" * 72)
        for v in ORDER:
            c = grid.get(v)
            if c is None:
                continue
            hw = wilson_halfwidth(c["top1"], c["n"])
            dc = (f'{100 * (c["top1"] - ctrl["top1"]):+.2f}'
                  if ctrl and v != CONTROL else "—")
            du = (f'{100 * (c["top1"] - uniform[near]):+.2f}' if ok else "—")
            ops = " ".join(
                f'{o.replace("mlp_","").replace("_proj","")}={c["per_op"][o]:.0f}'
                for o in sorted(c["per_op"], key=lambda o: -c["share"].get(o, 0)))
            drift = ("!" if abs(c["realized"] - base_sl) > 0.5 else " ")
            print(f'{v:<22}{c["top1"]:>8.4f}{hw:>8.3f}'
                  f'{c["realized"]:>9.2f}{drift}{dc:>9}{du:>9}   {ops}')
        print("    'vs ctrl' is noise-free w.r.t. calibration sampling: every "
              "variant shares one sigma draw.")
        if any(abs(c["realized"] - base_sl) > 0.5 for c in grid.values()):
            print("    ! = realized budget drifted >0.5 cyc from the control; "
                  "that row is not iso-compute with the rest.")


if __name__ == "__main__":
    main(*sys.argv[1:])
