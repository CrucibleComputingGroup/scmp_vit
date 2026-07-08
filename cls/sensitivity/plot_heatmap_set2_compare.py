"""Compare Set2-seeded sequential palettes (light → dark) on the cls
sensitivity heat-map. Set2 itself is qualitative, so we expand each Set2
hue into a continuous gradient via sns.light_palette(...).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
PRECS = [6, 7, 8]
JSONS = [HERE / f"sensitivity_per_operator_real_sc_p{p}.json" for p in PRECS]

# Set2 8-color palette (ColorBrewer)
SET2 = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
        "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"]
SET2_NAMES = ["mint", "orange", "lavender",
              "pink", "lime", "yellow", "tan", "grey"]


def matrify(d, key="l2"):
    ops = list(d["config"]["op_names"])
    n_blocks = int(d["config"]["n_blocks"])
    M = np.full((len(ops), n_blocks), np.nan)
    for r in d["grid"]:
        oi = ops.index(r["op"])
        bi = int(r["block"])
        v = r.get(key)
        if v is not None:
            M[oi, bi] = v
    return ops, M


def main():
    data = [json.load(open(p)) for p in JSONS]
    ops_ref, _ = matrify(data[0])
    n_blocks = data[0]["config"]["n_blocks"]
    Ms = [matrify(d)[1] for d in data]
    mean_per_op = np.mean([np.nanmean(M, axis=1) for M in Ms], axis=0)
    order = np.argsort(mean_per_op)[::-1]
    ops_plot = [ops_ref[i] for i in order]
    Ms = [M[order] for M in Ms]
    pretty = {
        "mlp_fc2": r"$\mathrm{MLP}_{fc2}$",
        "mlp_fc1": r"$\mathrm{MLP}_{fc1}$",
        "qkv_proj": r"$W_{QKV}$",
        "out_proj": r"$W_{O}$",
        "qk":  r"$Q\!\cdot\! K$",
        "av":  r"$A\!\cdot\! V$",
    }
    op_labels = [pretty.get(op, op) for op in ops_plot]
    vmax = max(np.nanmax(M) for M in Ms)
    vmin = 0.0

    # Build cmaps: 8 Set2-seeded sequential
    cmaps_named = []
    for name, hex_ in zip(SET2_NAMES, SET2):
        cm = sns.light_palette(hex_, as_cmap=True, reverse=False)
        cmaps_named.append((f"Set2-{name}", cm))

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 8,
    })

    n_cmaps = len(cmaps_named)
    fig, axes = plt.subplots(
        n_cmaps, 3, figsize=(11.5, 1.7 * n_cmaps),
        gridspec_kw={"wspace": 0.10, "hspace": 0.55},
    )

    for ri, (cm_name, cmap) in enumerate(cmaps_named):
        for ci, (M, p) in enumerate(zip(Ms, PRECS)):
            ax = axes[ri, ci]
            ax.imshow(M, aspect="auto", cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation="nearest")
            if ri == 0:
                ax.set_title(rf"$L = {1<<p}$ (p={p})")
            if ci == 0:
                ax.set_ylabel(cm_name, fontsize=10, fontweight="bold")
                ax.set_yticks(range(len(op_labels)))
                ax.set_yticklabels(op_labels)
            else:
                ax.set_yticks([])
            xt = list(range(0, n_blocks, 4))
            if (n_blocks - 1) not in xt:
                xt.append(n_blocks - 1)
            ax.set_xticks(xt)
            if ri == n_cmaps - 1:
                ax.set_xlabel("block index")
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", length=2)

    out_png = HERE / "figs" / "cmap_compare_set2.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
