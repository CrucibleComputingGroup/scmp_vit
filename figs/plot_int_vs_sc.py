"""Accuracy-vs-precision figure: INT fixed-point vs uniform SC on DINOv2 ViT-L/14.

x = number of representable levels, the axis on which the two families are
comparable: INT-b has 2**b; a halved bipolar SC stream of length L carries 2L
(wu-hpca2022 sign-magnitude identity). So int8 and halved-SC-128 both sit at
256, int7 and halved-64 at 128, int6 and halved-32 at 64 — and **192
(halved-96) is a point fixed-point cannot express**, which is the SC
granularity argument.

Reads ``results/<tag>/summary.csv`` (written by cls/collect_int_vs_sc.py) so
the figure never carries hardcoded numbers.

    python figs/plot_int_vs_sc.py [results/int_vs_sc]

Emits ``<tag>_light.png`` and ``<tag>_dark.png``.

Palette: dataviz categorical slots 1 (blue) / 2 (orange), with slot 3 (aqua)
reserved for MP SC so adding that series later repaints nothing. Both modes
validated with scripts/validate_palette.js — light CVD ΔE 24.7, dark 26.8
(floor 8), normal-vision 33.6 / 31.8 (floor 15). NOT scmp_cnn's INT violet:
blue↔violet collapses on the dark surface (normal-vision ΔE 9.8, a hard fail).
Identity is never colour-alone — distinct markers plus a legend plus direct
end labels.
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

THEMES = {
    "light": dict(sc="#2a78d6", integer="#eb6834", mp="#1baf7a",
                  surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", grid="#e1e0d9"),
    "dark": dict(sc="#3987e5", integer="#d95926", mp="#199e70",
                 surface="#1a1a19", ink="#f5f4f0", ink2="#c9c7c1",
                 muted="#8b8985", grid="#333330"),
}


def wilson_halfwidth(p, n, z=1.96):
    if not n:
        return 0.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c + half - p, p - (c - half))


def load(tag_dir, mp_csv=None):
    rows = []
    with open(Path(tag_dir) / "summary.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    # MP cells are placed by their REALIZED MAC-weighted budget (written by
    # cls/collect_mp.py), not by the nominal target — an MP point may only sit
    # beside a uniform point it actually matches in compute.
    if mp_csv and Path(mp_csv).exists():
        with open(mp_csv) as f:
            for r in csv.DictReader(f):
                rows.append(r)

    def num(r, k, cast=float):
        v = r.get(k, "")
        return cast(v) if v not in ("", None) else None

    fp = None
    series = {"int_symm": [], "int_asymm": [], "sc": [], "mp": []}
    n_images = None
    for r in rows:
        mode = r["mode"]
        top1 = num(r, "top1")
        n = num(r, "n", int)
        if top1 is None:
            continue
        n_images = n_images or n
        if mode == "fp":
            fp = top1
            continue
        # MP cells carry a FRACTIONAL level count (their realized
        # MAC-weighted budget), so this must not be cast to int.
        lv = num(r, "levels", float)
        if lv is None:
            continue
        pt = (lv, top1, wilson_halfwidth(top1, n or 0))
        if mode == "mp":
            series.setdefault("mp", []).append(pt)
        elif mode == "sc":
            series["sc"].append(pt)
        elif r.get("sym", "").lower() in ("true", "1"):
            series["int_symm"].append(pt)
        else:
            series["int_asymm"].append(pt)
    for v in series.values():
        v.sort()
    return fp, series, n_images


# Which INT curve(s) to draw. Asymmetric dominates symmetric at every level on
# this model (and by 52 pt at int4, where symmetric spends half its 16 codes on
# a sign range post-GELU/post-softmax tensors never occupy), so the default
# shows the better scheme only; the full symm-vs-asymm split lives in the table.
INT_SPEC = {
    "best": lambda t: [("int_asymm", "INT (asymmetric)", t["integer"], "s", "-")],
    "symm": lambda t: [("int_symm", "INT symmetric", t["integer"], "s", "-")],
    "both": lambda t: [
        ("int_symm", "INT symmetric", t["integer"], "s", "-"),
        ("int_asymm", "INT asymmetric", t["integer"], "D", (0, (4, 3)))],
}


def plot(tag_dir, mode="light", int_series="best", mp_csv=None):
    t = THEMES[mode]
    fp, series, n_images = load(tag_dir, mp_csv)
    tag = Path(tag_dir).name
    if int_series not in INT_SPEC:
        raise ValueError(f"int_series must be one of {sorted(INT_SPEC)}")
    # Only the series actually drawn may drive the axis range.
    drawn = {"sc", "mp"} | {s[0] for s in INT_SPEC[int_series](t)}
    series = {k: v for k, v in series.items() if k in drawn}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": t["muted"],
        "axes.linewidth": 0.8,
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["ink"],
        "axes.labelcolor": t["ink2"],
        "xtick.color": t["ink2"],
        "ytick.color": t["ink2"],
    })

    # int4-symmetric collapses ~55 pt below everything else. Plotting it on one
    # continuous axis squashes the region the figure is actually about, so the
    # y-axis is broken: every point is still shown, the break is marked, and
    # both segments keep a linear scale.
    all_y = [p[1] for v in series.values() for p in v] + (
        [fp] if fp is not None else [])
    lo_pts = [y for y in all_y if y < 0.6]
    hi_pts = [y for y in all_y if y >= 0.6]
    broken = bool(lo_pts)

    if broken:
        fig, (ax, ax_lo) = plt.subplots(
            2, 1, figsize=(7.8, 6.0), sharex=True,
            gridspec_kw=dict(height_ratios=[5, 1], hspace=0.08))
        panels = [ax, ax_lo]
    else:
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        ax_lo = None
        panels = [ax]

    spec = [("sc", "uniform SC (halved, prec 8)", t["sc"], "o", "-")]
    spec += INT_SPEC[int_series](t)
    if series.get("mp"):
        spec.append(("mp", "per-row MP SC (iso-compute)", t["mp"], "^", "-"))

    for p in panels:
        if fp is not None:
            p.axhline(fp, ls=(0, (5, 4)), lw=1.4, color=t["muted"], zorder=1)
        for key, label, color, marker, ls in spec:
            pts = series.get(key) or []
            if not pts:
                continue
            xs = [q[0] for q in pts]
            ys = [q[1] for q in pts]
            es = [q[2] for q in pts]
            alpha = 0.8 if (key == "int_asymm" and int_series == "both") else 1.0
            p.errorbar(xs, ys, yerr=es, fmt="none", ecolor=color,
                       elinewidth=1.0, capsize=2.5, alpha=0.5 * alpha, zorder=2)
            p.plot(xs, ys, linestyle=ls, lw=2.0, color=color, alpha=alpha,
                   zorder=3, marker=marker, ms=8, mfc=color,
                   mec=t["surface"], mew=2.0,
                   label=(label if p is ax else None))
        p.grid(True, axis="y", color=t["grid"], lw=0.8, zorder=0)
        p.set_axisbelow(True)
        p.spines["top"].set_visible(False)
        p.spines["right"].set_visible(False)

    pad = 0.012
    ax.set_ylim(min(hi_pts) - 0.035, max(hi_pts) + pad)
    ax.text(0.998, fp, f"FP {fp:.3f} ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9.5, color=t["ink2"])
    if broken:
        ax_lo.set_ylim(min(lo_pts) - 0.03, max(lo_pts) + 0.03)
        ax_lo.set_yticks([round(min(lo_pts), 2)])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)
        # Break marks on both facing edges.
        kw = dict(transform=fig.transFigure, color=t["muted"], lw=1.0,
                  clip_on=False)
        for a, frac in ((ax, 0), (ax_lo, 1)):
            pos = a.get_position()
            y = pos.y0 if frac == 0 else pos.y1
            for x0 in (pos.x0, pos.x1):
                fig.add_artist(plt.Line2D([x0 - 0.008, x0 + 0.008],
                                          [y - 0.012, y + 0.012], **kw))

    # The SC-only budget: a level count fixed-point has no format for. Carried
    # by the axis tick rather than a leader line — at 192 every offset for a
    # callout either crosses the SC curve or lands on its error bar.
    sc_pts = {p[0]: p[1] for p in series.get("sc", [])}
    int_lv = {p[0] for k, v in series.items()
              if k not in ("sc", "mp") for p in v}

    bottom = ax_lo if broken else ax
    bottom.set_xscale("log", base=2)
    ticks = sorted({int(round(v)) for v in list(sc_pts) + list(int_lv)})
    bottom.set_xticks(ticks)
    bottom.set_xticklabels([f"{v}\n(SC only)" if v not in int_lv else str(v)
                            for v in ticks])
    for lab in bottom.get_xticklabels():
        if "SC only" in lab.get_text():
            lab.set_color(t["sc"])
    bottom.minorticks_off()
    bottom.set_xlabel("representable levels   "
                      "(INT: $2^{bits}$   ·   SC: 2 × halved stream length)")
    ax.set_ylabel("ImageNet top-1")
    ax.yaxis.set_label_coords(-0.085, 0.42)

    leg = ax.legend(frameon=False, loc="lower right", fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    # Bottom margin has to clear the two-line "(SC only)" tick label as well as
    # the axis title; top is reserved for the figure-level title block.
    fig.tight_layout(rect=(0, 0.045, 1, 0.9))
    title = ("INT fixed-point vs stochastic computing — uniform and per-row MP"
             if series.get("mp") else
             "INT fixed-point vs uniform stochastic computing")
    fig.text(0.012, 0.965, title, fontsize=13, color=t["ink"], va="top")
    sub = (f"DINOv2 ViT-L/14 + linear head · ImageNet val ({n_images} imgs) "
           f"· all 5 op types × 24 blocks, no skipping")
    fig.text(0.012, 0.918, sub, fontsize=9.5, color=t["muted"], va="top")
    if series.get("mp"):
        fig.text(0.012, 0.884,
                 "MP @128 levels = mean of 4 calibration seeds (sd 0.33 pt); "
                 "MP @64 and @192 are single draws, not replicated",
                 fontsize=8.5, color=t["muted"], va="top")
    out = HERE / f"{tag}_{mode}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "int_vs_sc"
    which = sys.argv[2] if len(sys.argv) > 2 else "best"
    mp = sys.argv[3] if len(sys.argv) > 3 else "results/mp_eval/summary_mp.csv"
    for m in ("light", "dark"):
        plot(d, m, which, mp)
