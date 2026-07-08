"""3D surface version of the sensitivity heat-map.

The (op, block) grid is bilinearly upsampled and plotted as a smooth surface
with viridis_r colormap; high-sensitivity (op, block) cells protrude as
peaks above an otherwise flat plane. Three panels for L=64/128/256 share
the z-axis and colormap so they are directly comparable.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import zoom

HERE = Path(__file__).resolve().parent
PRECS = [6, 7, 8]
JSONS = [HERE / f"sensitivity_per_operator_real_sc_p{p}.json" for p in PRECS]

# Upsampling factors along (op, block). Cubic spline on a 6 x 24 grid → 48 x 96.
UP_OP = 8
UP_BLK = 4
CMAP = "viridis_r"


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

    # Order ops so quiet rows sit in front and the loudest sits at the back
    # (y=n_ops-1). With the current azim, the back of the scene is on the
    # right and the front is on the left, so quiet ops render on the left
    # (high visual priority) and loud peaks recede to the right.
    mean_per_op = np.mean([np.nanmean(M, axis=1) for M in Ms], axis=0)
    order = np.argsort(mean_per_op)  # ascending: quiet → loud
    ops_plot = [ops_ref[i] for i in order]
    Ms = [M[order] for M in Ms]
    n_ops = len(ops_plot)

    # Math labels need \mathbf{...} — fontweight="bold" alone does not bold
    # mathtext.
    pretty = {
        "mlp_fc2": "FC2",
        "mlp_fc1": "FC1",
        "qkv_proj": r"$\mathbf{W_{QKV}}$",
        "out_proj": r"$\mathbf{W_{O}}$",
        "qk":  r"$\mathbf{Q\!\cdot\! K}$",
        "av":  r"$\mathbf{A\!\cdot\! V}$",
    }
    op_labels = [pretty.get(op, op) for op in ops_plot]

    vmax = max(np.nanmax(M) for M in Ms)
    vmin = 0.0
    cmap = mpl.colormaps[CMAP]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "font.weight": "normal",
    })
    fig = plt.figure(figsize=(19.5, 7.0))

    # Build upsampled coordinate grids in *original* index space so axis
    # labels stay aligned to op names and block indices.
    big_y = np.linspace(0.0, n_ops - 1.0, n_ops * UP_OP)
    big_x = np.linspace(0.0, n_blocks - 1.0, n_blocks * UP_BLK)
    XX, YY = np.meshgrid(big_x, big_y)

    axes = []
    for ci, (M, p) in enumerate(zip(Ms, PRECS)):
        ax = fig.add_subplot(1, 3, ci + 1, projection="3d")
        axes.append(ax)

        # Cubic-spline upsample. Clip negative undershoot to keep the floor
        # at zero.
        Z = zoom(M, zoom=(UP_OP, UP_BLK), order=3, mode="nearest")
        Z = np.clip(Z, 0.0, None)

        ax.plot_surface(
            XX, YY, Z,
            cmap=cmap, norm=norm,
            rcount=80, ccount=160,
            linewidth=0, antialiased=True, shade=True,
        )

        ax.set_title(rf"Length = {1<<p}", fontsize=15, fontweight="bold",
                     y=0.92, pad=0)
        ax.set_xlim(0, n_blocks - 1)
        ax.set_ylim(0, n_ops - 1)
        ax.set_zlim(0, vmax * 1.02)
        # Drop the last tick (block 23) — it collides with the y-axis op
        # labels in the corner under the 3D perspective.
        xt = list(range(0, n_blocks, 4))
        ax.set_xticks(xt)
        ax.set_yticks(range(n_ops))
        ax.set_yticklabels(op_labels, fontweight="bold")
        ax.set_xlabel("block index", labelpad=18, fontweight="bold")
        ax.set_zticklabels([])
        # Orthographic projection keeps the z-axis line visually vertical
        # rather than tilted under perspective foreshortening.
        ax.set_proj_type("ortho")
        for t in ax.get_xticklabels():
            t.set_fontweight("bold")
        # Push tick labels off the surface so they don't collide with each
        # other or with the axis label under the heavy 3D foreshortening.
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", pad=4)

        ax.view_init(elev=22, azim=-44)
        # Quiet panes
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_alpha(0.0)
        ax.grid(True, linewidth=0.3, alpha=0.3)
        ax.set_box_aspect((2.5, 1.7, 1.1))

    # Restore a small figure margin, then trim each subplot's own L/R
    # whitespace by widening every axes' rect into the inter-panel gap
    # (without overlapping). The figure margin stays nonzero.
    fig.subplots_adjust(left=0.02, right=0.93, top=0.96, bottom=0.05,
                        wspace=0.04)
    expand = -0.005  # figure-fraction; negative = each subplot slightly smaller
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 - expand, pos.y0,
                         pos.width + 2 * expand, pos.height])

    cax = fig.add_axes([0.945, 0.18, 0.012, 0.66])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("backbone-feature L2 distance vs FP",
                 rotation=90, labelpad=10, fontsize=12, fontweight="bold")
    cb.ax.tick_params(labelsize=11)
    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")

    out_pdf = HERE / "figs" / "sensitivity_surface_3d.pdf"
    out_png = HERE / "figs" / "sensitivity_surface_3d.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
