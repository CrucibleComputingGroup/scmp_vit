"""Collect INT-vs-SC sweep JSONs into a table on the shared levels axis.

    OUT_DIR=results/int_vs_sc python cls/collect_int_vs_sc.py

Both families are placed on the same x-axis, "number of representable levels":
INT-b has 2**b; a halved bipolar SC stream of length L carries 2L
(wu-hpca2022). So int8 and halved-SC-128 both sit at 256, int7 and halved-64 at
128, int6 and halved-32 at 64 — and halved-96 (192) is a point fixed-point
cannot express.
"""
import glob
import json
import math
import os
import sys


def wilson_halfwidth(p, n, z=1.96):
    """Half-width of the 95% Wilson interval — honest error bars for top-1."""
    if not n:
        return float("nan")
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c + half - p, p - (c - half))


def main(out_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        d = json.load(open(f))
        name = os.path.basename(f)[:-5]
        rows.append({
            "name": name,
            "mode": d.get("mode"),
            "levels": d.get("levels"),
            "n": d.get("n"),
            "top1": d.get("top1"),
            "top5": d.get("top5"),
            "img_s": d.get("img_per_s"),
            "stoc_len": d.get("stoc_len"),
            "int_bits": d.get("int_bits"),
            "sym": d.get("int_sym"),
        })

    fp = next((r for r in rows if r["mode"] == "fp"), None)
    order = {"fp": 0, "int": 1, "sc": 2}
    rows.sort(key=lambda r: (order.get(r["mode"], 9),
                             -(r["levels"] or 0), r["name"]))

    hdr = (f'{"config":<16}{"mode":<6}{"levels":>8}{"top1":>9}{"±95%":>8}'
           f'{"top5":>9}{"vs FP":>9}{"img/s":>9}')
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["top1"] is None:
            continue
        hw = wilson_halfwidth(r["top1"], r["n"] or 0)
        delta = ("" if fp is None or r["mode"] == "fp"
                 else f'{100 * (r["top1"] - fp["top1"]):+.2f}')
        print(f'{r["name"]:<16}{r["mode"]:<6}{str(r["levels"] or "-"):>8}'
              f'{r["top1"]:>9.4f}{hw:>8.3f}{r["top5"]:>9.4f}'
              f'{delta:>9}{r["img_s"]:>9.1f}')

    print()
    print("Matched-precision pairs (same levels => same effective grid):")
    print(f'{"levels":>8}  {"INT symm":>9} {"INT asymm":>10} {"uniform SC":>11}'
          f'  {"SC - INTsymm":>13}')
    by = {}
    for r in rows:
        if r["levels"] is None or r["mode"] == "fp":
            continue
        key = r["levels"]
        slot = ("sc" if r["mode"] == "sc"
                else "int_symm" if r["sym"] else "int_asymm")
        by.setdefault(key, {})[slot] = r["top1"]
    for lv in sorted(by, reverse=True):
        e = by[lv]
        s, a, c = e.get("int_symm"), e.get("int_asymm"), e.get("sc")
        d = f"{100 * (c - s):+.2f}" if (s is not None and c is not None) else "-"
        f2 = lambda v: f"{v:.4f}" if v is not None else "-"
        print(f'{lv:>8}  {f2(s):>9} {f2(a):>10} {f2(c):>11}  {d:>13}')
    if fp:
        print(f'\nFP ceiling: {fp["top1"]:.4f} (n={fp["n"]})')

    csv = os.path.join(out_dir, "summary.csv")
    with open(csv, "w") as fh:
        cols = ["name", "mode", "levels", "n", "top1", "top5", "img_s",
                "stoc_len", "int_bits", "sym"]
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r[c] is None else str(r[c])
                              for c in cols) + "\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT_DIR", "."))
