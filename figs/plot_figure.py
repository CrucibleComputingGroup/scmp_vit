"""Accuracy-vs-precision figure: INT vs SC, uniform vs per-row MP, ± INT mask.

Extends ``plot_int_vs_sc.py`` from 3 series to 5 without adding a hue.

x = number of representable levels, the axis on which INT and SC are
comparable: INT-b has 2**b; a halved bipolar SC stream of length L carries 2L
(wu-hpca2022 sign-magnitude identity). So int8 and halved-SC-128 both sit at
256, int7 and halved-64 at 128 — and **192 (halved-96) is a point fixed-point
cannot express**, which is the SC granularity argument.

ENCODING. Hue carries the ALGORITHM (INT / uniform SC / per-row MP SC);
linestyle + marker fill carry whether a 10% INT mask is applied (solid+filled =
no mask, dashed+hollow = masked). Two new hues would have meant re-validating a
5-slot palette against the CVD floor; a mask is a MODIFIER of an algorithm, not
a fourth entity, so composite encoding is also the more honest mapping. The
3-hue palette is unchanged and still passes scripts/validate_palette.js
(light: worst adjacent ΔE 9.2 deutan / 27.6 normal; dark: 9.4 / 26.5). The
light-mode green sits at 2.74:1 contrast, a WARN that obligates relief — met by
the legend, the direct end labels, and the companion CSV table.

Reads the CSV written by ``cls/collect_figure.py``, which places every SC cell
at its REALIZED MAC-weighted stream length from the precision trace.

    python figs/plot_figure.py results/figure/summary_all.csv

Emits ``figure_light.png`` and ``figure_dark.png``.
"""
import csv
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

# (csv mode, legend label, theme colour key, marker, linestyle, filled)
SPEC = [
    ("sc",       "uniform SC",              "sc",      "o", "-",          True),
    ("int_asymm", "INT (asymmetric)",       "integer", "s", "-",          True),
    ("mp",       "per-row MP SC",           "mp",      "^", "-",          True),
    ("sc_hyb",   "uniform SC + 10% INT",    "sc",      "o", (0, (5, 3)),  False),
    ("mp_hyb",   "MP SC + 10% INT",         "mp",      "^", (0, (5, 3)),  False),
]


SEED_SUFFIX = re.compile(r"_s\d+$")

# Every cell is DRAWN at the nominal budget it targets, not at its trace-measured
# level count. The residuals being suppressed are 0.5-2.9%, which on a log2 axis
# is a few pixels -- encoding them in x does not communicate them, it just makes
# five arms look jittery and invites the reader to compare positions that are not
# resolvable. The two that are MATERIAL are stated in the caption instead, which
# is where a 2.9% fact can actually be read:
#
#   eff-96  hybrids sit 2.9% HIGH  -- INT7's 128 levels exceed the SC arm's 96,
#                                     so those 10% of groups carry 1.33x the
#                                     precision. The one budget where the INT
#                                     width could not be level-matched.
#   eff-128 hyb MP sits 2.5% LOW   -- the t64 mask removes mlp_fc2, which MP had
#                                     allocated 88 cycles = 176 levels; INT7's
#                                     128 is a downgrade for those groups.
#
# The exact realized values stay in the companion CSV (`levels` column), and the
# paired tests the claims rest on never used x at all.
NOMINAL_X = [16, 64, 96, 128, 256]


def snap_x(v):
    return min(NOMINAL_X, key=lambda n: abs(n - v))


def load(csv_path):
    """Collapse calibration-seed replicates into mean ± sd per cell.

    ERROR BARS ARE CALIBRATION SPREAD, NOT EVAL SAMPLING. This figure used to
    draw marginal Wilson intervals (±2.4 pt at n=1000), which was actively
    misleading: every claim the figure makes is a comparison BETWEEN arms at the
    same x, those are tested paired, and the paired intervals are ~6x tighter
    than the marginal ones. A marginal bar answers a question nobody is asking
    here and visually swamps all five arms.

    What a reader should actually worry about for an MP or hybrid point is
    reproducibility across calibration draws: at n_calib=128 that spread is
    ~0.4-0.6 pt and it is the reason the eff-64 hybrid point looked like it beat
    eff-128 (its plotted draw sat +0.37 above its own 4-seed mean). So cells
    named ``<base>_s<seed>`` are averaged with ``<base>`` and the bar is ±1 sd
    over those draws.

    Uniform SC and INT arms get NO bar, and that is not an omission: they have no
    calibration step, so their calibration spread is genuinely zero. Their eval
    sampling error is shared with every other cell on the figure and cancels in
    the paired comparisons.
    """
    fp, n_images = None, None
    groups = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            mode = r["mode"]
            if not r.get("top1"):
                continue
            top1 = float(r["top1"])
            n = int(r["n"]) if r.get("n") else 0
            if mode == "fp":
                fp = top1
                n_images = n_images or n
                continue
            n_images = n_images or n
            if mode not in {k for k, *_ in SPEC} or not r.get("levels"):
                continue
            base = SEED_SUFFIX.sub("", r["name"])
            g = groups.setdefault((mode, base), {"x": [], "y": []})
            g["x"].append(float(r["levels"]))
            g["y"].append(top1)

    series = {k: [] for k, *_ in SPEC}
    for (mode, _base), g in groups.items():
        ys, xs = g["y"], g["x"]
        mean_y = sum(ys) / len(ys)
        mean_x = sum(xs) / len(xs)
        sd = (statistics.stdev(ys) if len(ys) > 1 else 0.0)
        series[mode].append((snap_x(mean_x), mean_y, sd, len(ys)))
    for v in series.values():
        v.sort()
    return fp, series, n_images


def plot(csv_path, mode="light"):
    t = THEMES[mode]
    fp, series, n_images = load(csv_path)

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

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    if fp is not None:
        ax.axhline(fp, ls=(0, (5, 4)), lw=1.4, color=t["muted"], zorder=1)

    for key, label, ckey, marker, ls, filled in SPEC:
        pts = series.get(key) or []
        if not pts:
            continue
        color = t[ckey]
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        # Bars only where there is more than one calibration draw. A zero-length
        # bar on a single-draw point would read as "measured to be tiny" rather
        # than "not measured".
        rep = [(q[0], q[1], q[2]) for q in pts if q[3] > 1]
        if rep:
            ax.errorbar([q[0] for q in rep], [q[1] for q in rep],
                        yerr=[q[2] for q in rep], fmt="none", ecolor=color,
                        elinewidth=1.2, capsize=3.0, alpha=0.7, zorder=2)
        ax.plot(xs, ys, linestyle=ls, lw=2.0, color=color, zorder=3,
                marker=marker, ms=8,
                mfc=(color if filled else t["surface"]),
                mec=(t["surface"] if filled else color),
                mew=2.0, label=label)

    all_y = [p[1] + p[2] for v in series.values() for p in v] + \
            [p[1] - p[2] for v in series.values() for p in v] + (
        [fp] if fp is not None else [])
    ax.set_ylim(min(all_y) - 0.02, max(all_y) + 0.012)
    if fp is not None:
        ax.text(0.998, fp, f"FP {fp:.3f} ", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9.5, color=t["ink2"])

    ax.grid(True, axis="y", color=t["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # SC-only budgets: level counts fixed-point has no format for. Marked on the
    # tick itself — at 192 any callout offset either crosses the SC curve or
    # lands on its error bar.
    int_lv = {round(p[0]) for p in series.get("int_asymm", [])}
    # Ticks live on the NOMINAL budget grid, not on each cell's realized value.
    # Cells are placed at their trace-measured x (an MP cell lands at 127.9 or
    # 125.0, not 128), which is the honest position — but turning every
    # realized value into its own tick produced overlapping "125 128" and
    # "191 192" labels. Snap each SC budget to the nominal rung it targets;
    # the exact realized numbers are in the companion CSV.
    # 96 belongs here for the same reason 192 does: it is a level count no INT
    # format expresses, and it is where all three orderings hold significantly
    # at n=10000. Omitting it snapped the eff-96 cells onto the 64 tick (96 is
    # equidistant from 64 and 128, and min() breaks the tie toward the first),
    # so the axis had no label at the run's most informative budget.
    NOMINAL = [16, 64, 96, 128, 192, 256]
    sc_lv = set()
    for k in ("sc", "mp", "sc_hyb", "mp_hyb"):
        for p in series.get(k, []):
            sc_lv.add(min(NOMINAL, key=lambda v: abs(v - p[0])))
    ax.set_xscale("log", base=2)
    ticks = sorted(int_lv | sc_lv)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v}\n(SC only)" if v not in int_lv else str(v)
                        for v in ticks])
    for lab in ax.get_xticklabels():
        if "SC only" in lab.get_text():
            lab.set_color(t["sc"])
    ax.minorticks_off()
    ax.set_xlabel("representable levels   "
                  "(INT: $2^{bits}$   ·   SC: 2 × halved stream length)")
    ax.set_ylabel("ImageNet top-1")

    # Two-block legend: hue = algorithm, linestyle = mask. Spelling the second
    # channel out beats making the reader infer it from five entries.
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, frameon=False, loc="lower right",
                    fontsize=9.5, ncol=1, handlelength=2.6)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    ax.add_artist(leg)
    style_key = [
        Line2D([], [], color=t["muted"], lw=2.0, ls="-", marker="o", ms=7,
               mfc=t["muted"], mec=t["surface"], mew=1.6, label="pure SC"),
        Line2D([], [], color=t["muted"], lw=2.0, ls=(0, (5, 3)), marker="o",
               ms=7, mfc=t["surface"], mec=t["muted"], mew=1.6,
               label="+ 10% INT mask"),
    ]
    # Left-centre is the one region no series occupies (INT is the only curve
    # left of 64, and it sits high). Upper-left collides with the FP rule.
    # Lower left, not the old (0.015, 0.62): with INT4 on the curve the INT arm
    # climbs steeply across the left of the panel and ran straight through the
    # style key. Below y~0.79 and left of the 64 tick nothing is plotted -- the
    # SC arms all start at 64.
    leg2 = ax.legend(handles=style_key, frameon=False, loc="lower left",
                     fontsize=9, handlelength=2.6,
                     bbox_to_anchor=(0.015, 0.02))
    for txt in leg2.get_texts():
        txt.set_color(t["muted"])

    fig.tight_layout(rect=(0, 0.045, 1, 0.772))
    fig.text(0.012, 0.985,
             "INT fixed-point vs stochastic computing — uniform, per-row MP, "
             "and hybrid SC/INT", fontsize=13, color=t["ink"], va="top")
    fig.text(0.012, 0.943,
             f"DINOv2 ViT-L/14 + linear head · ImageNet val ({n_images:,} imgs) "
             f"· all 5 op types × 24 blocks", fontsize=9.5, color=t["muted"],
             va="top")
    fig.text(0.012, 0.912,
             "x = MAC-weighted representable levels from the trace, on TRUE "
             "(unpadded) MACs — so equal x means equal precision.",
             fontsize=8.5, color=t["muted"], va="top")
    # Caption lines are hand-wrapped: fig.text does not wrap, so an over-long
    # string runs off the canvas and is silently cropped at render. Keep each
    # under ~110 characters at fontsize 8.5 on a 8.4in figure.
    fig.text(0.012, 0.884,
             "Masked cells run 10% of groups on INT at the SAME stream length, "
             "INT width matched to the SC level count",
             fontsize=8.5, color=t["muted"], va="top")
    fig.text(0.012, 0.860,
             "(INT6/7/8) and counted at those levels; they also spend 6–11% "
             "less. Points drawn at nominal levels — two differ:",
             fontsize=8.5, color=t["muted"], va="top")
    fig.text(0.012, 0.836,
             "eff-96 hybrids carry 1.33× (INT7=128 vs SC 96); eff-128 hybrid MP "
             "carries 0.975× (mlp_fc2 176→128 levels).",
             fontsize=8.5, color=t["muted"], va="top")
    # Say what the bars ARE, per cell, rather than asserting a replication that
    # a given CSV may not contain. load() sets sd=0 for a single draw, so a
    # blanket "mean of 4 seeds" caption is false on any un-replicated run and
    # invents a precision claim the figure cannot support.
    n_rep = max((k for *_, k in sum(series.values(), [])), default=1)
    fig.text(0.012, 0.812,
             ("MP points are the mean of up to %d calibration seeds, bar = ±1 sd."
              % n_rep) if n_rep > 1 else
             "One calibration draw per MP cell, so no calibration bar; "
             "the eff-64 hybrid's 4-seed sd was 0.6 pt at n=1000.",
             fontsize=8.5, color=t["muted"], va="top")
    fig.text(0.012, 0.788,
             "MP recipe: 2% protected channels at eff-64 (+4.5 pt); off above "
             "it (+0.3 pt at eff-96, below the +0.5 pt switch threshold).",
             fontsize=8.5, color=t["muted"], va="top")

    out = HERE / f"{OUT_STEM}_{mode}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    path = (sys.argv[1] if len(sys.argv) > 1
            else ROOT / "results" / "figure" / "summary_all.csv")
    # Output stem is a parameter so a new study does not overwrite the previous
    # figure. The n=1000 figure stays on disk as provenance for the reports that
    # cite it -- silently replacing it would leave those reports describing an
    # image that no longer exists.
    OUT_STEM = sys.argv[2] if len(sys.argv) > 2 else "figure"
    for m in ("light", "dark"):
        plot(path, m)
