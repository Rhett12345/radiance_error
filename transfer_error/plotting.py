# -*- coding: utf-8 -*-
"""
Nature-figure multi-panel output:  1 figure, 2×2 bidirectional heatmaps.

Panel layout:
  A — Reflective A→B   (fy4a source → fy4b target,  rel_err_mean %)
  B — Reflective B→A   (fy4b source → fy4a target,  rel_err_mean %)
  C — IR A→B           (fy4a source → fy4b target,  dTb_mean K)
  D — IR B→A           (fy4b source → fy4a target,  dTb_mean K)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors

from .config import get_wavelength_um

cmap = get_cmap("GnBu")
cmap_clipped = mcolors.LinearSegmentedColormap.from_list(
    "GnBu_clip", cmap(np.linspace(0.2, 1.0, 256))
)

def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        # ===== 全局字体 =====
        "font.size": 10,              # 原来 7.5
        # ===== 坐标轴标题 =====
        "axes.titlesize": 12,         # 原来 8
        "axes.titleweight": "bold",
        # ===== x/y label =====
        "axes.labelsize": 11,         # 原来 7.5
        # ===== 坐标刻度 =====
        "xtick.labelsize": 9,         # 原来 6.5
        "ytick.labelsize": 9,         # 原来 6.5
        # ===== 图例 =====
        "legend.fontsize": 6,       # 原来 5.8
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.3,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _ch_short(ch: str) -> str:
    """Very short: 'ch01'."""
    return ch.split("_")[1]


def _sat_tag(ch: str) -> str:
    return ch.split("_")[0]




# ---------------------------------------------------------------------------
# Panel A / B: reflective heatmap (bidirectional)
# ---------------------------------------------------------------------------

def _panel_refl_heatmap(ax, results: List[dict], sat_source: str,
                        direction_label: str) -> None:
    refl = [r for r in results
            if r["ch_type"] == "reflective" and _sat_tag(r["channel"]) == sat_source]
    if not refl:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return

    channels = sorted(set(r["channel"] for r in refl),
                      key=lambda c: get_wavelength_um(c))
    perts = sorted(set(r["perturbation"] for r in refl))

    n_ch = len(channels)
    n_pt = len(perts)
    mat = np.full((n_ch, n_pt), np.nan)
    for i, ch in enumerate(channels):
        for j, pt in enumerate(perts):
            row = [r for r in refl
                   if r["channel"] == ch and r["perturbation"] == pt]
            if row:
                mat[i, j] = row[0]["rel_err_mean"]

    x_edges = np.arange(n_pt + 1) - 0.5
    y_edges = np.arange(n_ch + 1) - 0.5
    im = ax.pcolormesh(x_edges, y_edges, mat, cmap=cmap_clipped,
                       edgecolors="black", linewidth=0.6, antialiased=False)

    ax.set_xticks(range(n_pt))
    ax.set_xticklabels([f"{p * 100:.1f}%" for p in perts], rotation=45, ha="right")
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([_ch_short(c) for c in channels])
    ax.set_xlabel("Perturbation fraction")
    ax.set_title(f"Reflective  {direction_label}  —  rel_err_mean (%)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    for i in range(n_ch):
        for j in range(n_pt):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if v > 0.5 * np.nanmax(mat) else "black")


# ---------------------------------------------------------------------------
# Panel C / D: IR heatmap (bidirectional)
# ---------------------------------------------------------------------------

def _panel_ir_heatmap(ax, results: List[dict], sat_source: str,
                      direction_label: str) -> None:
    ir_list = [r for r in results
               if r["ch_type"] == "ir" and _sat_tag(r["channel"]) == sat_source]
    if not ir_list:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return

    channels = sorted(set(r["channel"] for r in ir_list),
                      key=lambda c: get_wavelength_um(c))
    perts = sorted(set(r["perturbation"] for r in ir_list))

    n_ch = len(channels)
    n_pt = len(perts)
    mat = np.full((n_ch, n_pt), np.nan)
    for i, ch in enumerate(channels):
        for j, pt in enumerate(perts):
            row = [r for r in ir_list
                   if r["channel"] == ch and r["perturbation"] == pt]
            if row:
                mat[i, j] = row[0]["dTb_mean"]

    x_edges = np.arange(n_pt + 1) - 0.5
    y_edges = np.arange(n_ch + 1) - 0.5
    im = ax.pcolormesh(x_edges, y_edges, mat, cmap=cmap_clipped,
                       edgecolors="black", linewidth=0.6, antialiased=False)

    ax.set_xticks(range(n_pt))
    ax.set_xticklabels([f"{p:.1f}" for p in perts], rotation=45, ha="right")
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([_ch_short(c) for c in channels])
    ax.set_xlabel("$\\Delta K$  (K)")
    ax.set_title(f"IR  {direction_label}  —  dTb_mean (K)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    for i in range(n_ch):
        for j in range(n_pt):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if v > 0.5 * np.nanmax(mat) else "black")


# ---------------------------------------------------------------------------
# Full figure builder
# ---------------------------------------------------------------------------

def make_all_figures(results: List[dict], output_dir: str) -> List[str]:
    """Generate one 2×2 bidirectional heatmap figure.

    Layout:
      (A) Reflective A→B   (B) Reflective B→A
      (C) IR A→B           (D) IR B→A
    """
    _configure_style()

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.5))
    ((ax_a, ax_b), (ax_c, ax_d)) = axes

    _panel_refl_heatmap(ax_a, results, "fy4a", "A→B")
    _panel_refl_heatmap(ax_b, results, "fy4b", "B→A")
    _panel_ir_heatmap(ax_c, results, "fy4a", "A→B")
    _panel_ir_heatmap(ax_d, results, "fy4b", "B→A")

    fig.tight_layout()
    out = Path(output_dir) / "fig_sensitivity_bidirectional.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")
    return [str(out)]
