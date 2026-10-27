# Copyright 2026 the squares-in-squares-vis authors
# SPDX-License-Identifier: Apache-2.0
"""Static matplotlib previews (for notebooks whose saved view cannot run JavaScript, e.g. Kaggle): draw(viewer, n)
and draw_scores(viewer). Needs matplotlib (`pip install squares-in-squares-vis[preview]`)."""
from __future__ import annotations

from .viewer import COLORS, PackingViewer, corners, grid_score


def draw(viewer: PackingViewer, n: int, size: float = 6.0, show: bool = True):
    """One packing: squares coloured by angle class, bounding square of side L. Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    rows, e = viewer.packings[n], viewer.entry(n)
    c, L, (x0, y0) = corners(rows), viewer.L[n], e["o"]
    rot, near = set(e["r"]), set(e["q"])
    cls = ["rotated" if k in rot else "near" if k in near else "aligned" for k in range(len(rows))]
    fig, ax = plt.subplots(figsize=(size, size))
    for key, label in (("aligned", "axis-aligned"), ("near", "nearly aligned (< 1e-3 rad)"), ("rotated", "rotated")):
        idx = [k for k, g in enumerate(cls) if g == key]
        if idx:
            ax.add_collection(PolyCollection(c[idx], facecolors=COLORS[key], edgecolors="white", linewidths=0.4,
                                             alpha=0.9, label=f"{label} ({len(idx)})"))
    ax.plot([x0, x0 + L, x0 + L, x0, x0], [y0, y0, y0 + L, y0 + L, y0], color=COLORS["box"], lw=1.2,
            label="bounding square")
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.02)
    ax.set_title(f"n = {n}   L = {L:.12f}   score_n = {viewer.score(n):.12f}   rotated: {len(rot)} (+{len(near)} near)",
                 fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if show:
        plt.show()
    return fig


def draw_scores(viewer: PackingViewer, show: bool = True):
    """score_n over n with the grid ceiling. Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt

    ns = sorted(viewer.L)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(ns, [grid_score(n) for n in ns], color="#b0afa9", lw=1, marker="." if len(ns) < 50 else None,
            label="grid ceiling")
    ax.plot(ns, [viewer.score(n) for n in ns], color=COLORS["aligned"], lw=1.2, marker="o" if len(ns) < 50 else None,
            label="score_n")
    ax.set_xlabel("n")
    ax.set_ylabel("score_n = L/√n − 1")
    ax.grid(alpha=0.3)
    ax.set_title(f"{viewer.src.name}: total {viewer.total():.6f} (overlaps not checked)", fontsize=10)
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if show:
        plt.show()
    return fig
