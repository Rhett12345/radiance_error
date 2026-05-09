# -*- coding: utf-8 -*-
"""
Result visualisation:  sensitivity-curve plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from .config import OUTPUT_DPI


# ---------- helpers ---------------------------------------------------------

def _configure_style() -> None:
    """Apply a clean, readable matplotlib style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "grid.color": "#BBBBBB",
        "grid.linewidth": 0.4,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": OUTPUT_DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _channel_label(ch: str) -> str:
    """Short label e.g. 'A-ch01 (0.47 µm)'."""
    from .config import get_wavelength_um
    sat = ch.split("_")[0].upper().replace("FY4", "FY-4")
    ch_id = ch.split("_")[1]
    try:
        lam = get_wavelength_um(ch)
        return f"{sat} {ch_id}  ({lam:g} µm)"
    except KeyError:
        return f"{sat} {ch_id}"


# ---------- reflective-channel plot -----------------------------------------

def plot_reflective(results: List[dict], output_path: str) -> str:
    """Plot *rel_err_mean (%)* vs *perturbation fraction (%)* for reflective channels.

    Each channel is drawn as a separate line with markers.
    """
    _configure_style()

    # Filter & sort
    refl = [r for r in results if r["ch_type"] == "reflective"]
    if not refl:
        print("No reflective-channel results to plot.")
        return ""

    channels = sorted(set(r["channel"] for r in refl))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for ch in channels:
        pts = sorted(
            [r for r in refl if r["channel"] == ch],
            key=lambda r: r["perturbation"],
        )
        x_vals = [p["perturbation"] * 100 for p in pts]   # fraction → %
        y_vals = [p["rel_err_mean"] for p in pts]
        ax.plot(x_vals, y_vals, marker="o", markersize=4, linewidth=1.5,
                label=_channel_label(ch))

    ax.set_xlabel("Perturbation fraction  (%)")
    ax.set_ylabel("Mean relative output error  (%)")
    ax.set_title("Reflective channels — transfer-model sensitivity")
    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", which="both", direction="in")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=OUTPUT_DPI)
    plt.close(fig)
    return str(out)


# ---------- IR-channel plot -------------------------------------------------

def plot_ir(results: List[dict], output_path: str) -> str:
    """Plot *dTb_mean (K)* vs *ΔK* for infrared channels.

    Each channel is drawn as a separate line with markers.
    """
    _configure_style()

    ir_list = [r for r in results if r["ch_type"] == "ir"]
    if not ir_list:
        print("No IR-channel results to plot.")
        return ""

    channels = sorted(set(r["channel"] for r in ir_list))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for ch in channels:
        pts = sorted(
            [r for r in ir_list if r["channel"] == ch],
            key=lambda r: r["perturbation"],
        )
        x_vals = [p["perturbation"] for p in pts]    # already in K
        y_vals = [p["dTb_mean"] for p in pts]
        ax.plot(x_vals, y_vals, marker="s", markersize=4, linewidth=1.5,
                label=_channel_label(ch))

    ax.set_xlabel("Input perturbation  $\\Delta K$  (K)")
    ax.set_ylabel("Mean $\\delta T_b$  (K)")
    ax.set_title("Infrared channels — transfer-model sensitivity")
    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", which="both", direction="in")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=OUTPUT_DPI)
    plt.close(fig)
    return str(out)
