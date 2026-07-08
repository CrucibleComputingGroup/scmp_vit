"""Plot per-(op, block) SC sensitivity heat-maps for cls at p={6,7,8}.

1x3 panel, shared L2 colormap, ops sorted by mean L2 across the three L.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PRECS = [6, 7, 8]
JSONS = [HERE / f"sensitivity_per_operator_real_sc_p{p}.json" for p in PRECS]


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

    # Build matrices
    Ms = [matrify(d)[1] for d in data]

    # Order ops by mean L2 across the three precisions, ascending so the
    # hottest op (mlp_fc2) sits at the top of the heatmap.
    mean_per_op = np.mean([np.nanmean(M, axis=1) for M in Ms], axis=0)
    order = np.argsort(mean_per_op)[::-1]  # descending → top row hottest
    ops_plot = [ops_ref[i] for i in order]
    Ms = [M[order] for M in Ms]

    # Pretty op labels
    pretty = {
        "mlp_fc2": r"$\mathrm{MLP}_{fc2}$",
        "mlp_fc1": r"$\mathrm{MLP}_{fc1}$",
        "qkv_proj": r"$W_{QKV}$",
        "out_proj": r"$W_{O}$",
        "qk":  r"$Q\!\cdot\! K$",
        "av":  r"$A\!\cdot\! V$",
    }
    op_labels = [pretty.get(op, op) for op in ops_plot]

    # Shared color scale across all three panels.
    vmax = max(np.nanmax(M) for M in Ms)
    vmin = 0.0
    cmap = mpl.colormaps["rocket_r"] if "rocket_r" in mpl.colormaps \
        else mpl.colormaps["magma_r"]

    # Figure: 1x3 panels + a single shared colorbar to the right.
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(11.5, 2.4),
        gridspec_kw={"wspace": 0.18},
        constrained_layout=False,
    )

    ims = []
    for ax, M, p in zip(axes, Ms, PRECS):
        im = ax.imshow(
            M, aspect="auto", cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        ims.append(im)
        ax.set_title(rf"$L = {1<<p}$ (p={p})")
        ax.set_xlabel("block index")
        # x ticks at 0, 4, 8, ..., n_blocks-1
        xt = list(range(0, n_blocks, 4))
        if (n_blocks - 1) not in xt:
            xt.append(n_blocks - 1)
        ax.set_xticks(xt)
        ax.set_yticks(range(len(op_labels)))
        ax.set_yticklabels(op_labels)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2)
        # Subtle grid
        ax.set_xticks(np.arange(-0.5, n_blocks, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(op_labels), 1), minor=True)
        ax.grid(which="minor", color="0.5", linewidth=0.3, alpha=0.25)
        ax.tick_params(which="minor", length=0)

    # Hide y tick labels on the 2nd/3rd panels (shared op axis)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    # Shared colorbar
    fig.subplots_adjust(left=0.07, right=0.92, top=0.86, bottom=0.20)
    cax = fig.add_axes([0.935, 0.20, 0.012, 0.66])
    cb = fig.colorbar(ims[-1], cax=cax)
    cb.set_label("backbone-feature L2 distance vs FP", rotation=90, labelpad=8)

    out_pdf = HERE / "figs" / "sensitivity_heatmap.pdf"
    out_png = HERE / "figs" / "sensitivity_heatmap.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")

    # Console summary
    for p, M in zip(PRECS, Ms):
        nz_max = np.nanmax(M)
        nz_p99 = np.nanpercentile(M, 99)
        print(f"p={p} (L={1<<p}): max L2={nz_max:.3f}  p99={nz_p99:.3f}")


if __name__ == "__main__":
    main()
