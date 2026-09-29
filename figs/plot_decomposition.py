"""What actually bought the accuracy, at one fixed compute budget.

The levels-vs-accuracy figure answers "how do the families compare". This one
answers "which change was worth what" at a fixed budget:

    uniform SC (qk per_head)      the configuration vit_sc shipped
      -> uniform SC (qk per_row)  a SCOPE fix — no MP, helps uniform on its own
      -> MP, channel axis only    split the outlier input channels off
      -> MP, channel + row axis   add per-row dispatch

Steps 2 and 3 are both mixed precision. They are split because the split IS the
finding: per-row dispatch alone ties uniform SC (−0.2) and channel protection
alone gives +1.9, but together they give +4.5 — super-additive, because
protection also repairs the row metric the allocator ranks on.

Form: a cumulative step chart with the increment labelled on each transition.
One measure, one hue (the SC blue from the shared palette) — the stages are an
ORDERED sequence, not categories, so hue would encode nothing. The FP ceiling
is a reference rule, and every bar is directly labelled, which is also the
relief the light-mode contrast WARN on this palette requires.

    python figs/plot_decomposition.py results/fig1k 32
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

THEMES = {
    "light": dict(sc="#2a78d6", up="#1baf7a", down="#eb6834",
                  surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", grid="#e1e0d9"),
    "dark": dict(sc="#3987e5", up="#199e70", down="#d95926",
                 surface="#1a1a19", ink="#f5f4f0", ink2="#c9c7c1",
                 muted="#8b8985", grid="#333330"),
}

# (tag template, label). Order is the causal chain, not a ranking.
#
# Steps 3 and 4 are BOTH mixed precision — protected channels are MP along the
# channel axis, per-row dispatch is MP along the row axis. They are split here
# because the split is the finding (neither is worth much alone; the pair is
# super-additive), not because protection is somehow outside MP.
STAGES = [
    ("uni_head_{t}", "uniform SC\nqk per_head"),
    ("uni_{t}",      "+ qk per_row\n(scope fix, still uniform)"),
    # uniprot_31, not uniprot_32: protecting 2% at stoc_len 64 on top of a base
    # of 32 realizes 32.60, i.e. 1.9% OVER budget, which would flatter this
    # step. Base 31 realizes 31.62 — under the uniform comparator — so the
    # increment shown here is the conservative one.
    ("uniprot_31",   "MP: channel\nprotection only"),
    ("r{t}_pc",      "MP: protection\n+ per-row"),
]


def load(d: Path, t: int):
    out = []
    for tmpl, label in STAGES:
        p = d / f"{tmpl.format(t=t)}.json"
        if not p.exists():
            print(f"  missing {p.name} — stage dropped")
            continue
        j = json.loads(p.read_text())
        out.append((label, float(j["top1"]), j.get("n"), p.stem))
    fp = d / "fp.json"
    fpv = json.loads(fp.read_text())["top1"] if fp.exists() else None
    return out, fpv


def plot(d: Path, t: int, mode="light"):
    th = THEMES[mode]
    stages, fpv = load(d, t)
    if len(stages) < 2:
        print("not enough stages to plot")
        return None

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11, "axes.edgecolor": th["muted"], "axes.linewidth": 0.8,
        "figure.facecolor": th["surface"], "axes.facecolor": th["surface"],
        "savefig.facecolor": th["surface"], "text.color": th["ink"],
        "axes.labelcolor": th["ink2"], "xtick.color": th["ink2"],
        "ytick.color": th["ink2"],
    })
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    xs = list(range(len(stages)))
    ys = [s[1] for s in stages]

    if fpv is not None:
        ax.axhline(fpv, ls=(0, (5, 4)), lw=1.4, color=th["muted"], zorder=1)
        ax.text(0.998, fpv, f"FP {fpv:.3f} ", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9.5, color=th["ink2"])

    ax.bar(xs, ys, width=0.56, color=th["sc"], zorder=3,
           edgecolor=th["surface"], linewidth=2.0)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.0015, f"{y:.4f}", ha="center", va="bottom",
                fontsize=10, color=th["ink"], zorder=4)

    # Increment between consecutive stages, coloured by sign. This is the
    # number the figure exists to show.
    for i in range(1, len(stages)):
        d_pt = (ys[i] - ys[i - 1]) * 100
        col = th["up"] if d_pt >= 0 else th["down"]
        ytop = max(ys[i], ys[i - 1])
        ax.annotate("", xy=(i, ytop + 0.010), xytext=(i - 1, ytop + 0.010),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.8),
                    zorder=4)
        ax.text(i - 0.5, ytop + 0.0125, f"{d_pt:+.2f} pt", ha="center",
                va="bottom", fontsize=10.5, color=col, fontweight="bold",
                zorder=4)

    lo = min(ys) - 0.030
    ax.set_ylim(lo, max(max(ys), fpv or 0) + 0.028)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in stages], fontsize=10)
    ax.set_ylabel("ImageNet top-1")
    ax.grid(True, axis="y", color=th["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    n = stages[0][2]
    fig.tight_layout(rect=(0, 0, 1, 0.855))
    fig.text(0.012, 0.985,
             f"What bought the accuracy — SC at effective {2 * t} levels",
             fontsize=13, color=th["ink"], va="top")
    fig.text(0.012, 0.940,
             f"DINOv2 ViT-L/14 + linear head · ImageNet val ({n} imgs) · "
             f"cumulative, left to right", fontsize=9.5, color=th["muted"],
             va="top")
    fig.text(0.012, 0.907,
             "Step 1 is a quantization-SCOPE fix and helps uniform SC on its "
             "own. Steps 2-3 are both MP — channel axis, then row axis.",
             fontsize=8.5, color=th["muted"], va="top")
    fig.text(0.012, 0.879,
             "Increments are paired differences on identical images. Realized "
             "budgets: 32.00 / 32.00 / 31.62 / 32.09 halved cycles.",
             fontsize=8.5, color=th["muted"], va="top")

    out = HERE / f"decomposition_t{t}_{mode}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "fig1k")
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    for m in ("light", "dark"):
        plot(d, t, m)
