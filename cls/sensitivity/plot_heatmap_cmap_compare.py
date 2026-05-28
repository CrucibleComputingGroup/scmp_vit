"""Render the sensitivity heatmap under several light-to-dark colormaps so
the user can pick one. Output: figs/cmap_compare.png."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PRECS = [6, 7, 8]
JSONS = [HERE / f"sensitivity_per_operator_real_sc_p{p}.json" for p in PRECS]

import seaborn as sns  # noqa: F401  (registers seaborn colormaps)

CMAPS = ["rocket_r", "mako_r", "flare", "crest", "YlGnBu", "magma_r"]


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

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 8,
    })

    n_cmaps = len(CMAPS)
    fig, axes = plt.subplots(
        n_cmaps, 3, figsize=(11.5, 1.8 * n_cmaps),
        gridspec_kw={"wspace": 0.10, "hspace": 0.55},
    )

    for ri, cm_name in enumerate(CMAPS):
        cmap = mpl.colormaps[cm_name]
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

    out_png = HERE / "figs" / "cmap_compare.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
