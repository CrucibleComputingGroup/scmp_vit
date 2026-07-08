"""Sweep three knobs on the 3D sensitivity surface to help pick a final
look. Produces three comparison PNGs:

  figs/surface_views.png       — view angles
  figs/surface_smoothness.png  — spline upsampling factor
  figs/surface_sizes.png       — figure size (single PNG per size)

Use these to pick (elev, azim, up_op, up_blk, fig size); they all feed into
the final plot_heatmap_3d_surface.py.
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


def _prep():
    data = [json.load(open(p)) for p in JSONS]
    ops_ref, _ = matrify(data[0])
    n_blocks = data[0]["config"]["n_blocks"]
    Ms = [matrify(d)[1] for d in data]
    mean_per_op = np.mean([np.nanmean(M, axis=1) for M in Ms], axis=0)
    order = np.argsort(mean_per_op)  # ascending y = quiet → loud
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
    return Ms, op_labels, n_blocks, vmax


def _draw_panel(ax, Z_big, n_blocks, op_labels, vmax,
                elev, azim, label_prefix=None,
                show_y_labels=True, show_z_label=True):
    n_ops = len(op_labels)
    big_y = np.linspace(0.0, n_ops - 1.0, Z_big.shape[0])
    big_x = np.linspace(0.0, n_blocks - 1.0, Z_big.shape[1])
    XX, YY = np.meshgrid(big_x, big_y)
    cmap = mpl.colormaps[CMAP]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    ax.plot_surface(XX, YY, Z_big, cmap=cmap, norm=norm,
                    rcount=80, ccount=160,
                    linewidth=0, antialiased=True, shade=True)
    ax.set_xlim(0, n_blocks - 1)
    ax.set_ylim(0, n_ops - 1)
    ax.set_zlim(0, vmax * 1.02)
    xt = list(range(0, n_blocks, 4))
    if (n_blocks - 1) not in xt:
        xt.append(n_blocks - 1)
    ax.set_xticks(xt)
    ax.set_yticks(range(n_ops))
    if show_y_labels:
        ax.set_yticklabels(op_labels)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("block index", labelpad=2)
    if show_z_label:
        ax.set_zlabel("L2 vs FP", labelpad=4)
    else:
        ax.set_zticklabels([])
    ax.view_init(elev=elev, azim=azim)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.set_box_aspect((3.2, 1.0, 1.1))
    if label_prefix is not None:
        ax.text2D(-0.05, 0.55, label_prefix, transform=ax.transAxes,
                  fontsize=10, fontweight="bold",
                  rotation=90, va="center", ha="right")


def upsample(M, up_op, up_blk):
    Z = zoom(M, zoom=(up_op, up_blk), order=3, mode="nearest")
    return np.clip(Z, 0.0, None)


def fig_views():
    Ms, op_labels, n_blocks, vmax = _prep()
    Zs = [upsample(M, 16, 8) for M in Ms]

    views = [
        ("top-down\nelev 45 / azim -60", 45, -60),
        ("balanced\nelev 24 / azim -62", 24, -62),
        ("low side-on\nelev 12 / azim -55", 12, -55),
        ("from-rear\nelev 24 / azim -120", 24, -120),
    ]

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 7})
    n_rows = len(views)
    fig = plt.figure(figsize=(13.0, 3.4 * n_rows))
    for ri, (lbl, elev, azim) in enumerate(views):
        for ci, (Z, p) in enumerate(zip(Zs, PRECS)):
            ax = fig.add_subplot(n_rows, 3, ri * 3 + ci + 1, projection="3d")
            _draw_panel(ax, Z, n_blocks, op_labels, vmax,
                        elev=elev, azim=azim,
                        label_prefix=lbl if ci == 0 else None,
                        show_y_labels=(ci == 0),
                        show_z_label=(ci == 0))
            if ri == 0:
                ax.set_title(rf"$L = {1<<p}$ (p={p})")
    out = HERE / "figs" / "surface_views.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_smoothness():
    Ms, op_labels, n_blocks, vmax = _prep()

    # (label, up_op, up_blk)
    smoothings = [
        ("raw\n(no upsample)", 1, 1),
        ("light\n(8 x 4)", 8, 4),
        ("default\n(16 x 8)", 16, 8),
        ("very smooth\n(32 x 16)", 32, 16),
    ]

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 7})
    n_rows = len(smoothings)
    fig = plt.figure(figsize=(13.0, 3.4 * n_rows))
    for ri, (lbl, uo, ub) in enumerate(smoothings):
        Zs = [upsample(M, uo, ub) for M in Ms]
        for ci, (Z, p) in enumerate(zip(Zs, PRECS)):
            ax = fig.add_subplot(n_rows, 3, ri * 3 + ci + 1, projection="3d")
            _draw_panel(ax, Z, n_blocks, op_labels, vmax,
                        elev=24, azim=-62,
                        label_prefix=lbl if ci == 0 else None,
                        show_y_labels=(ci == 0),
                        show_z_label=(ci == 0))
            if ri == 0:
                ax.set_title(rf"$L = {1<<p}$ (p={p})")
    out = HERE / "figs" / "surface_smoothness.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_sizes():
    Ms, op_labels, n_blocks, vmax = _prep()
    Zs = [upsample(M, 16, 8) for M in Ms]

    sizes = [
        ("small", (11.5, 3.4)),
        ("medium", (14.0, 4.4)),
        ("large", (17.0, 5.5)),
    ]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "xtick.labelsize": 7,
                         "ytick.labelsize": 8})
    for name, sz in sizes:
        fig = plt.figure(figsize=sz)
        for ci, (Z, p) in enumerate(zip(Zs, PRECS)):
            ax = fig.add_subplot(1, 3, ci + 1, projection="3d")
            _draw_panel(ax, Z, n_blocks, op_labels, vmax,
                        elev=24, azim=-62,
                        show_y_labels=True,
                        show_z_label=(ci == 0))
            ax.set_title(rf"$L = {1<<p}$ (p={p})")
        fig.subplots_adjust(left=0.02, right=0.96, top=0.93, bottom=0.05,
                            wspace=0.04)
        out = HERE / "figs" / f"surface_size_{name}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    fig_views()
    fig_smoothness()
    fig_sizes()
