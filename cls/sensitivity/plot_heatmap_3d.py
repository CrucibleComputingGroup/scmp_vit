"""3D bar version of the sensitivity heat-map: 1x3 panels at L=64/128/256.

Each bar is one (op, block) pair; height encodes backbone-feature L2 vs FP.
Bars are colored by height with a shared colormap and z-axis range so the
three panels are directly comparable.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

HERE = Path(__file__).resolve().parent
PRECS = [6, 7, 8]
JSONS = [HERE / f"sensitivity_per_operator_real_sc_p{p}.json" for p in PRECS]

CMAP = "YlGnBu"


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

    # Order ops so the *most sensitive* row sits at the BACK of the 3D scene
    # (largest y), with quiet ops in the front. This way the tall bars don't
    # occlude the short ones.
    mean_per_op = np.mean([np.nanmean(M, axis=1) for M in Ms], axis=0)
    order = np.argsort(mean_per_op)  # ascending: quiet → loud
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
    cmap = mpl.colormaps[CMAP]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
    })

    fig = plt.figure(figsize=(13.5, 4.0))

    n_ops = len(ops_plot)
    bar_w = 0.85  # along x (block axis)
    bar_d = 0.7   # along y (op axis)

    axes = []
    for ci, (M, p) in enumerate(zip(Ms, PRECS)):
        ax = fig.add_subplot(1, 3, ci + 1, projection="3d")
        axes.append(ax)

        # Build flat arrays of bar (x, y, z, dx, dy, dz, color)
        xs, ys, zs, dz, colors = [], [], [], [], []
        for oi in range(n_ops):
            for bi in range(n_blocks):
                v = M[oi, bi]
                if not np.isfinite(v):
                    continue
                xs.append(bi)
                ys.append(oi)
                zs.append(0.0)
                dz.append(max(v, 1e-6))
                colors.append(cmap(norm(v)))
        xs = np.asarray(xs) - bar_w / 2.0
        ys = np.asarray(ys) - bar_d / 2.0
        zs = np.asarray(zs)
        dz = np.asarray(dz)
        dx = np.full_like(xs, bar_w)
        dy = np.full_like(ys, bar_d)

        ax.bar3d(xs, ys, zs, dx, dy, dz,
                 color=colors, shade=True,
                 edgecolor="0.3", linewidth=0.15)

        ax.set_title(rf"$L = {1<<p}$ (p={p})", pad=2)
        ax.set_xlabel("block index", labelpad=4)
        ax.set_xlim(-0.5, n_blocks - 0.5)
        ax.set_ylim(-0.5, n_ops - 0.5)
        ax.set_zlim(0, vmax * 1.02)
        xt = list(range(0, n_blocks, 4))
        if (n_blocks - 1) not in xt:
            xt.append(n_blocks - 1)
        ax.set_xticks(xt)
        ax.set_yticks(range(n_ops))
        ax.set_yticklabels(op_labels)
        if ci == 0:
            ax.set_zlabel("L2 vs FP", labelpad=2)
        else:
            ax.set_zticklabels([])

        # Camera angle: look from top-front-right corner
        ax.view_init(elev=22, azim=-58)
        # Quiet panes
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        ax.grid(True, linewidth=0.3, alpha=0.3)
        # Tighten box aspect: x is long (24 blocks), y is short (6 ops)
        ax.set_box_aspect((3.2, 1.0, 1.1))

    fig.subplots_adjust(left=0.03, right=0.93, top=0.95, bottom=0.05,
                        wspace=0.05)

    # Shared colorbar
    cax = fig.add_axes([0.945, 0.18, 0.012, 0.66])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("backbone-feature L2 distance vs FP",
                 rotation=90, labelpad=8)

    out_pdf = HERE / "figs" / "sensitivity_heatmap_3d.pdf"
    out_png = HERE / "figs" / "sensitivity_heatmap_3d.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
