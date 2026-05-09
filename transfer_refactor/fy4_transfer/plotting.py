# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from .channel_catalog import channel_title
from .model import FitResult, finite_xy


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#555555",
        "axes.grid": True,
        "grid.color": "#AAAAAA",
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def equation_text(fit: FitResult) -> str:
    if fit.model_type == "linear":
        b0 = fit.intercept
        sign = "+" if b0 >= 0 else "−"
        return f"$y = {fit.coef1:.4f}x {sign} {abs(b0):.4f}$"

    c1 = fit.coef1
    c2 = fit.coef2 if fit.coef2 is not None else 0.0
    b0 = fit.intercept
    def fmt(v: float, var: str) -> str:
        sign = "+" if v >= 0 else "−"
        return f" {sign} {abs(v):.4f}{var}"
    return f"$y = {c2:.4f}x^2{fmt(c1, 'x')}{fmt(b0, '')}$"


def plot_regression(
    x,
    y,
    fit: FitResult,
    source_ch: str,
    target_ch: str,
    direction: str,
    plot_dir: str,
) -> str | None:
    x_arr, y_arr = finite_xy(x, y)
    if len(y_arr) < 2:
        return None

    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    x_min, x_max = float(x_arr.min()), float(x_arr.max())
    margin = (x_max - x_min) * 0.05 if x_max > x_min else max(abs(x_min) * 0.05, 1.0)
    x_range = np.linspace(x_min - margin, x_max + margin, 300).reshape(-1, 1)
    y_range = fit.model.predict(x_range)
    y_pred = fit.model.predict(x_arr)
    sigma = float(np.std(y_arr - y_pred))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(
        x_range.ravel(), y_range - sigma, y_range + sigma,
        color="#D64B4B", alpha=0.10, label="$\\pm 1\\sigma$ band"
    )
    ax.scatter(
        x_arr.ravel(), y_arr,
        s=5, alpha=0.6, color="black", edgecolors="none",
        rasterized=(len(y_arr) > 2000), zorder=3, label="samples"
    )
    ax.plot(x_range, y_range, color="#FFAAAA", linewidth=1.2, zorder=4, label=f"fitted ({fit.model_type})")

    ref_min = min(float(x_arr.min()), float(y_arr.min()))
    ref_max = max(float(x_arr.max()), float(y_arr.max()))
    ax.plot([ref_min, ref_max], [ref_min, ref_max], color="#888888", linewidth=1.0, linestyle=":", zorder=2, label="1:1")

    ax.text(
        0.04, 0.97,
        f"{equation_text(fit)}\n$R = {fit.r:.5f}$\n$\\sigma = {fit.residual_std:.5f}$\n$n = {fit.n:,}$",
        transform=ax.transAxes, fontsize=17, va="top", ha="left", color="#2B2B2B",
    )

    ax.set_xlabel(f"{source_ch.upper()} Radiance  (W·m⁻²·sr⁻¹·μm⁻¹)", labelpad=6)
    ax.set_ylabel(f"{target_ch.upper()} Radiance  (W·m⁻²·sr⁻¹·μm⁻¹)", labelpad=6)
    ax.set_title(channel_title(source_ch), pad=12)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-2, 4))
    ax.legend(loc="lower right", frameon=False, fontsize=16, markerscale=1.4)

    out = Path(plot_dir) / f"{source_ch}_to_{target_ch}_{direction}.png"
    fig.savefig(out, dpi=600, facecolor="white")
    plt.close(fig)
    return str(out)
