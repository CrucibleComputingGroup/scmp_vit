#!/usr/bin/env python3
"""Paired top-1 comparisons between cells evaluated on the SAME images.

At n=1000 the marginal Wilson half-width is about +-2.6 pt at p~0.78, wider
than the 1-3 pt effects this study is about, so two cells whose marginal
intervals overlap can still differ reliably. Every cell scores the same images
in the same loader order and dumps per-image top-1 hits (``<tag>_hits.npy``),
which makes the PAIRED comparison available: only the discordant images carry
information, and there are far fewer of them than n.

Reports, per pair, McNemar's exact test (b ~ Binomial(b+c, 1/2) under the null)
plus the paired difference and a bootstrap CI over images.

    python cls/paired_stats.py results/fig1k --base uni_32 \
        --against r32_base,r32_pc,r32_pc_esc

Do NOT quote a marginal Wilson interval as evidence against a paired result
here; at this n it is answering a different question.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load(d: Path, tag: str):
    hits = d / f"{tag}_hits.npy"
    meta = d / f"{tag}.json"
    if not hits.exists():
        raise SystemExit(f"missing {hits} — rerun that cell with --out_json set")
    h = np.load(hits).astype(bool)
    m = json.loads(meta.read_text()) if meta.exists() else {}
    return h, m


def compare(a, b, n_boot=10000, seed=0):
    """McNemar exact + bootstrap CI on the paired top-1 difference (b - a)."""
    if a.shape != b.shape:
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
    n = a.size
    b_only = int((~a & b).sum())      # b right, a wrong
    a_only = int((a & ~b).sum())      # a right, b wrong
    disc = b_only + a_only
    p = (stats.binomtest(b_only, disc, 0.5).pvalue if disc else 1.0)
    diff = (b.mean() - a.mean()) * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = (b[idx].mean(axis=1) - a[idx].mean(axis=1)) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=n, diff=diff, lo=lo, hi=hi, p=p,
                b_only=b_only, a_only=a_only, disc=disc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--base", required=True,
                    help="reference cell every other cell is differenced against")
    ap.add_argument("--against", required=True, help="comma-separated tags")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    base, bmeta = load(args.dir, args.base)
    print(f"base = {args.base}   top-1 {base.mean():.4f}   n={base.size}"
          + (f"   realized_sl~{bmeta.get('stoc_len')}" if bmeta else ""))
    hdr = (f'{"cell":<20}{"top-1":>8}{"Δ pt":>8}{"95% CI":>18}'
           f'{"p(McNemar)":>12}{"disc":>7}')
    print(hdr); print("-" * len(hdr))
    for tag in [t.strip() for t in args.against.split(",") if t.strip()]:
        try:
            h, _m = load(args.dir, tag)
        except SystemExit as e:
            print(f'{tag:<20}  {e}')
            continue
        r = compare(base, h, args.n_boot)
        star = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else \
               "*" if r["p"] < 0.05 else ""
        ci = "[{:+.2f}, {:+.2f}]".format(r["lo"], r["hi"])
        print("{:<20}{:>8.4f}{:>+8.2f}{:>18}{:>11.4f}{:<3}{:>5}".format(
            tag, h.mean(), r["diff"], ci, r["p"], star, r["disc"]))
    print("\ndisc = images the two cells disagree on; only those carry "
          "information in the paired test.")


if __name__ == "__main__":
    main()
