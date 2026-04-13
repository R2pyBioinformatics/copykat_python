"""Enhanced heatmap visualization (R source: heatmap.3.R)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, to_rgba
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

__all__ = ["heatmap3"]


def _colors_to_rgba(arr: np.ndarray) -> np.ndarray:
    """Convert an array of string color names to an RGBA float array."""
    if arr.dtype.kind in ("U", "S", "O"):  # string or object dtype
        shape = arr.shape
        flat = arr.ravel()
        rgba = np.array([to_rgba(c) for c in flat])
        return rgba.reshape(*shape, 4)
    return arr


def heatmap3(
    x: np.ndarray,
    *,
    row_cluster: bool = True,
    col_cluster: bool = False,
    dist_func: str = "euclidean",
    link_method: str = "ward",
    col_side_colors: np.ndarray | None = None,
    row_side_colors: np.ndarray | None = None,
    cmap: str | Any = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    breaks: np.ndarray | None = None,
    dendrogram_: str = "row",
    key: bool = True,
    figsize: tuple[float, float] = (10, 10),
    save_path: str | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Create an enhanced heatmap with optional dendrograms and side color bars.

    Parameters
    ----------
    x : np.ndarray
        Data matrix (rows x columns).
    row_cluster : bool
        Whether to cluster rows.
    col_cluster : bool
        Whether to cluster columns.
    dist_func : str
        Distance metric for clustering (e.g., ``"euclidean"``).
    link_method : str
        Linkage method (e.g., ``"ward"``).
    col_side_colors : np.ndarray | None
        Color array for column annotations. Shape (n_cols,) or (n_rows_annot, n_cols).
    row_side_colors : np.ndarray | None
        Color array for row annotations. Shape (n_rows,) or (n_rows, n_cols_annot).
    cmap : str or Colormap
        Colormap for the heatmap.
    vmin : float | None
        Minimum value for color scaling.
    vmax : float | None
        Maximum value for color scaling.
    breaks : np.ndarray | None
        Custom color breakpoints. If provided, overrides vmin/vmax.
    dendrogram_ : str
        Which dendrograms to show: ``"row"``, ``"column"``, ``"both"``, ``"none"``.
    key : bool
        Whether to show a color key.
    figsize : tuple[float, float]
        Figure size in inches.
    save_path : str | None
        If provided, save figure to this path.
    show : bool
        Whether to display the figure.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``"row_order"`` and ``"col_order"`` containing
        the row and column ordering arrays, and ``"row_dendrogram"`` and
        ``"col_dendrogram"`` containing linkage matrices if clustering was performed.
    """
    data = x.copy()
    n_rows, n_cols = data.shape

    result: dict[str, Any] = {"row_order": np.arange(n_rows), "col_order": np.arange(n_cols)}

    # Determine layout
    show_row_dend = row_cluster and dendrogram_ in ("row", "both", "r")
    show_col_dend = col_cluster and dendrogram_ in ("column", "both", "c")
    has_col_side = col_side_colors is not None
    has_row_side = row_side_colors is not None

    # Cluster
    row_link = None
    col_link = None
    if row_cluster:
        row_dist = pdist(data, metric=dist_func)
        row_link = linkage(row_dist, method=link_method)
        row_dend = dendrogram(row_link, no_plot=True)
        row_order = np.array(row_dend["leaves"])
        data = data[row_order, :]
        result["row_order"] = row_order
        result["row_dendrogram"] = row_link

    if col_cluster:
        col_dist = pdist(data.T, metric=dist_func)
        col_link = linkage(col_dist, method=link_method)
        col_dend = dendrogram(col_link, no_plot=True)
        col_order = np.array(col_dend["leaves"])
        data = data[:, col_order]
        result["col_order"] = col_order
        result["col_dendrogram"] = col_link

    # Set up figure grid
    fig = plt.figure(figsize=figsize)
    n_grid_rows = 1 + int(show_col_dend) + int(has_col_side) + int(key)
    n_grid_cols = 1 + int(show_row_dend) + int(has_row_side)

    heights = []
    if show_col_dend:
        heights.append(0.8)
    if has_col_side:
        heights.append(0.15)
    heights.append(5)  # main heatmap
    if key:
        heights.append(0.3)

    widths = []
    if show_row_dend:
        widths.append(1.0)
    if has_row_side:
        widths.append(0.15)
    widths.append(5)  # main heatmap

    gs = GridSpec(len(heights), len(widths), figure=fig,
                  height_ratios=heights, width_ratios=widths,
                  hspace=0.02, wspace=0.02)

    row_idx = 0
    col_base = 0

    # Column dendrogram
    if show_col_dend and col_link is not None:
        ax_cdend = fig.add_subplot(gs[row_idx, len(widths) - 1])
        dendrogram(col_link, ax=ax_cdend, no_labels=True, color_threshold=0)
        ax_cdend.axis("off")
        row_idx += 1

    # Column side colors
    if has_col_side:
        ax_cside = fig.add_subplot(gs[row_idx, len(widths) - 1])
        csc = col_side_colors
        if csc.ndim == 1:
            csc = csc.reshape(1, -1)
        elif csc.shape[1] != n_cols and csc.shape[0] == n_cols:
            # R convention: (n_cols, n_tracks) — transpose to (n_tracks, n_cols)
            csc = csc.T
        if col_cluster:
            csc = csc[:, result["col_order"]]
        csc = _colors_to_rgba(csc)
        ax_cside.imshow(csc, aspect="auto", interpolation="none")
        ax_cside.set_xticks([])
        ax_cside.set_yticks([])
        row_idx += 1

    # Row dendrogram
    heatmap_row = row_idx
    if show_row_dend and row_link is not None:
        ax_rdend = fig.add_subplot(gs[heatmap_row, 0])
        dendrogram(row_link, ax=ax_rdend, orientation="left", no_labels=True,
                   color_threshold=0)
        ax_rdend.axis("off")
        col_base = 1

    # Row side colors
    if has_row_side:
        ax_rside = fig.add_subplot(gs[heatmap_row, col_base])
        rsc = row_side_colors
        if rsc.ndim == 1:
            rsc = rsc.reshape(-1, 1)
        elif rsc.shape[0] != n_rows and rsc.shape[1] == n_rows:
            # R convention: (n_tracks, n_rows) — transpose to (n_rows, n_tracks)
            rsc = rsc.T
        if row_cluster:
            rsc = rsc[result["row_order"], :]
        rsc = _colors_to_rgba(rsc)
        ax_rside.imshow(rsc, aspect="auto", interpolation="none")
        ax_rside.set_xticks([])
        ax_rside.set_yticks([])
        col_base += 1

    # Main heatmap
    ax_heat = fig.add_subplot(gs[heatmap_row, col_base])
    if breaks is not None:
        norm = Normalize(vmin=breaks[0], vmax=breaks[-1])
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax_heat.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                        interpolation="none")
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])

    # Color key
    if key:
        ax_key = fig.add_subplot(gs[-1, col_base])
        plt.colorbar(im, cax=ax_key, orientation="horizontal")
        row_idx += 1

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return result
