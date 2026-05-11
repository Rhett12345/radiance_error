# -*- coding: utf-8 -*-
"""
Nature-figure multi-panel output:  2 figures (fy4a / fy4b), each 2×2.

Panel layout per satellite:
  A — Reflective line plot   (perturbation %  vs  rel_err_mean %,  p95 band, 1:1 ref)
  B — IR line plot           (ΔK  vs  dTb_mean,  p95 band, 1:1 ref)
  C — Reflective heatmap     (channel × perturbation,  colour = rel_err_mean %)
  D — IR heatmap             (channel × ΔK,  colour = dTb_mean)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from .config import get_wavelength_um

# ---------------------------------------------------------------------------
# Palette — one colour per channel index, consistent across panels
# ---------------------------------------------------------------------------
_CH_COLOURS = [
    "#2166AC", "#D6604D", "#4DAF4A", "#984EA3",
    "#FF7F00", "#377EB8", "#E41A1C", "#4D4D4D",
    "#A65628", "#F781BF", "#66C2A5", "#FC8D62",
    "#8DA0CB", "#E78AC3", "#A6D854",
]


def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 5.8,
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


def _ch_label(ch: str) -> str:
    """Short label: 'FY-4A ch01 (0.47µm)'."""
    sat = ch.split("_")[0].upper().replace("FY4", "FY-4")
    ch_id = ch.split("_")[1]
    try:
        lam = get_wavelength_um(ch)
        return f"{sat} {ch_id} ({lam:g}µm)"
    except KeyError:
        return f"{sat} {ch_id}"


def _ch_short(ch: str) -> str:
    """Very short: 'ch01'."""
    return ch.split("_")[1]


def _sat_tag(ch: str) -> str:
    return ch.split("_")[0]


# ---------------------------------------------------------------------------
# Panel A: reflective line plot
# ---------------------------------------------------------------------------

def _panel_refl_line(ax, results: List[dict], satellite: str) -> None:
    refl = [r for r in results
            if r["ch_type"] == "reflective" and _sat_tag(r["channel"]) == satellite]
    if not refl:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return

    channels = sorted(set(r["channel"] for r in refl),
                      key=lambda c: get_wavelength_um(c))

    for idx, ch in enumerate(channels):
        pts = sorted([r for r in refl if r["channel"] == ch],
                     key=lambda r: r["perturbation"])
        x = [p["perturbation"] * 100 for p in pts]    # fraction → %
        y = [p["rel_err_mean"] for p in pts]
        color = _CH_COLOURS[idx % len(_CH_COLOURS)]

        ax.plot(x, y, color=color, linewidth=0.9, marker="o", markersize=2.5,
                label=_ch_label(ch))

    # 1:1 reference
    xmax = max(p["perturbation"] * 100 for p in refl)
    ax.plot([0, xmax], [0, xmax], color="#888888", linewidth=0.6,
            linestyle="--", label="1:1")

    ax.set_xlabel("Perturbation fraction  (%)")
    ax.set_ylabel("Mean relative output error  (%)")
    ax.set_title("A  Reflective channels")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.tick_params(direction="in")


# ---------------------------------------------------------------------------
# Panel B: IR line plot
# ---------------------------------------------------------------------------

def _panel_ir_line(ax, results: List[dict], satellite: str) -> None:
    ir_list = [r for r in results
               if r["ch_type"] == "ir" and _sat_tag(r["channel"]) == satellite]
    if not ir_list:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return

    channels = sorted(set(r["channel"] for r in ir_list),
                      key=lambda c: get_wavelength_um(c))

    for idx, ch in enumerate(channels):
        pts = sorted([r for r in ir_list if r["channel"] == ch],
                     key=lambda r: r["perturbation"])
        x = [p["perturbation"] for p in pts]          # ΔK
        y = [p["dTb_mean"] for p in pts]
        color = _CH_COLOURS[idx % len(_CH_COLOURS)]

        ax.plot(x, y, color=color, linewidth=0.9, marker="s", markersize=2.5,
                label=_ch_label(ch))

    # 1:1 reference
    xmax = max(p["perturbation"] for p in ir_list)
    ax.plot([0, xmax], [0, xmax], color="#888888", linewidth=0.6,
            linestyle="--", label="1:1")

    ax.set_xlabel("Input perturbation  $\\Delta K$  (K)")
    ax.set_ylabel("Mean  $|\\delta T_b|$  (K)")
    ax.set_title("B  Infrared channels")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.tick_params(direction="in")


# ---------------------------------------------------------------------------
# Panel C: reflective heatmap
# ---------------------------------------------------------------------------

def _panel_refl_heatmap(ax, results: List[dict], satellite: str) -> None:
    refl = [r for r in results
            if r["ch_type"] == "reflective" and _sat_tag(r["channel"]) == satellite]
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
    im = ax.pcolormesh(x_edges, y_edges, mat, cmap="YlOrRd",
                       edgecolors="black", linewidth=0.6, antialiased=False)

    ax.set_xticks(range(n_pt))
    ax.set_xticklabels([f"{p * 100:.1f}%" for p in perts], rotation=45, ha="right")
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([_ch_short(c) for c in channels])
    ax.set_xlabel("Perturbation fraction")
    ax.set_title("C  Reflective — rel_err_mean (%)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    # Annotate cells
    for i in range(n_ch):
        for j in range(n_pt):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.2, color="white" if v > 0.5 * np.nanmax(mat) else "black")


# ---------------------------------------------------------------------------
# Panel D: IR heatmap
# ---------------------------------------------------------------------------

def _panel_ir_heatmap(ax, results: List[dict], satellite: str) -> None:
    ir_list = [r for r in results
               if r["ch_type"] == "ir" and _sat_tag(r["channel"]) == satellite]
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
    im = ax.pcolormesh(x_edges, y_edges, mat, cmap="YlOrRd",
                       edgecolors="black", linewidth=0.6, antialiased=False)

    ax.set_xticks(range(n_pt))
    ax.set_xticklabels([f"{p:.1f}" for p in perts], rotation=45, ha="right")
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([_ch_short(c) for c in channels])
    ax.set_xlabel("$\\Delta K$  (K)")
    ax.set_title("D  IR — dTb_mean (K)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    for i in range(n_ch):
        for j in range(n_pt):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=5.2,
                        color="black" if v > 0.5 * np.nanmax(mat) else "black")


# ---------------------------------------------------------------------------
# Full figure builder
# ---------------------------------------------------------------------------

def make_figure(results: List[dict], satellite: str, output_dir: str) -> str:
    """Build one 2×2 multi-panel figure for the given satellite.

    Returns path to the saved PNG.
    """
    _configure_style()

    sat_label = satellite.upper().replace("FY4", "FY-4")
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))
    ((ax_a, ax_b), (ax_c, ax_d)) = axes

    _panel_refl_line(ax_a, results, satellite)
    _panel_ir_line(ax_b, results, satellite)
    _panel_refl_heatmap(ax_c, results, satellite)
    _panel_ir_heatmap(ax_d, results, satellite)

    fig.suptitle(f"Transfer-model sensitivity  —  {sat_label} source channels",
                 fontsize=9.5, fontweight="bold", y=1.01)

    fig.tight_layout()
    out = Path(output_dir) / f"fig_sensitivity_{satellite}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def make_all_figures(results: List[dict], output_dir: str) -> List[str]:
    """Generate both figures (fy4a, fy4b)."""
    saved: List[str] = []
    for sat in ["fy4a", "fy4b"]:
        path = make_figure(results, sat, output_dir)
        if path:
            saved.append(path)
    print(f"  saved {len(saved)} figures")
    return saved
