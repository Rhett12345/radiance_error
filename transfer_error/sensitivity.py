# -*- coding: utf-8 -*-
"""
Monte-Carlo sensitivity experiment.

Vectorised MC:  add noise to source-channel radiance → run transfer model →
compute per-channel error statistics (mean, p95).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import (
    N_MC,
    PERTURBATION_DK,
    PERTURBATION_FRAC,
    RANDOM_SEED,
    channel_type,
    get_wavelength_um,
)
from .physics import brightness_temperature, dL_dTb, jacobian_radiance


# ---------- transfer model --------------------------------------------------

def apply_model(x: np.ndarray, coeff1: float, coeff2: float | None,
                intercept: float) -> np.ndarray:
    """Evaluate  y = coeff2·x² + coeff1·x + intercept."""
    y = coeff1 * x + intercept
    if coeff2 is not None:
        y += coeff2 * x * x
    return y


# ---------- single-channel MC -----------------------------------------------

def _run_one_channel(
    x_src: np.ndarray,
    lam_um: float,
    ch_type: str,
    coeff1: float,
    coeff2: float | None,
    intercept: float,
    perturbation: float,
    rng: np.random.Generator,
) -> dict:
    """Monte Carlo for one channel × perturbation combination.

    Returns
    -------
    dict
        dy_mean, dy_p95, rel_err_mean, rel_err_p95,
        dTb_mean, dTb_p95 (NaN for reflective).
    """
    n_scenes = len(x_src)
    y_clean = apply_model(x_src, coeff1, coeff2, intercept)  # (N_scenes,)

    # --- noise sigma ----------------------------------------------------
    if ch_type == "reflective":
        sigma = perturbation * x_src
    else:
        Tb_typ = brightness_temperature(np.atleast_1d(np.mean(x_src)), lam_um)[0]
        sigma = np.full(n_scenes, abs(dL_dTb(Tb_typ, lam_um)) * perturbation)

    sigma = np.maximum(sigma, 1e-12)

    # --- Monte Carlo (vectorised) ---------------------------------------
    noise = rng.normal(0.0, sigma, size=(N_MC, n_scenes))
    x_noisy = x_src[np.newaxis, :] + noise
    x_noisy = np.maximum(x_noisy, 1e-10)

    y_noisy = apply_model(x_noisy, coeff1, coeff2, intercept)  # (N_MC, N_scenes)
    dy = np.abs(y_noisy - y_clean[np.newaxis, :])               # (N_MC, N_scenes)

    dy_mean = float(np.mean(dy))
    dy_p95  = float(np.percentile(dy, 95))
    mean_y_clean = float(np.mean(y_clean))
    rel_err_mean = (dy_mean / mean_y_clean * 100.0) if mean_y_clean > 0 else np.nan

    # rel_err_p95 uses the scene-average y_clean as denominator
    # (per-scene p95 normalisation would be noisier)
    rel_err_p95 = (dy_p95 / mean_y_clean * 100.0) if mean_y_clean > 0 else np.nan

    # --- delta-Tb (IR only) ---------------------------------------------
    dTb_mean: float = np.nan
    dTb_p95: float  = np.nan
    if ch_type == "ir":
        Tb_noisy = brightness_temperature(y_noisy, lam_um)
        Tb_clean = brightness_temperature(y_clean, lam_um)
        dTb = np.abs(Tb_noisy - Tb_clean[np.newaxis, :])
        dTb_mean = float(np.nanmean(dTb))
        dTb_p95  = float(np.nanpercentile(dTb, 95))

    return {
        "dy_mean": dy_mean,
        "dy_p95": dy_p95,
        "rel_err_mean": rel_err_mean,
        "rel_err_p95": rel_err_p95,
        "dTb_mean": dTb_mean,
        "dTb_p95": dTb_p95,
    }


# ---------- main experiment ------------------------------------------------

def run_experiment(
    radiance_data: Dict[str, np.ndarray],
    coeff_list: List[dict],
) -> List[dict]:
    """Run the full sensitivity experiment.

    Returns
    -------
    list of dict
        One per channel × perturbation.
        Keys: channel, ch_type, perturbation, perturbation_label,
              dy_mean, dy_p95, rel_err_mean, rel_err_p95,
              dTb_mean, dTb_p95.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    results: List[dict] = []

    for coeff in coeff_list:
        src_ch = coeff["source_ch"]

        if src_ch not in radiance_data:
            print(f"  skip {src_ch} — not found in radiance data")
            continue
        try:
            lam_um = get_wavelength_um(src_ch)
            cht = channel_type(src_ch)
        except (KeyError, ValueError) as exc:
            print(f"  skip {src_ch} — {exc}")
            continue

        x_jac = jacobian_radiance(radiance_data[src_ch], lam_um)

        pert_grid = PERTURBATION_FRAC if cht == "reflective" else PERTURBATION_DK
        pert_label = "pert_frac" if cht == "reflective" else "delta_K"

        for pert in pert_grid:
            stats = _run_one_channel(
                x_src=x_jac,
                lam_um=lam_um,
                ch_type=cht,
                coeff1=coeff["coeff1"],
                coeff2=coeff["coeff2"],
                intercept=coeff["intercept"],
                perturbation=pert,
                rng=rng,
            )
            results.append({
                "channel": src_ch,
                "ch_type": cht,
                "perturbation": pert,
                "perturbation_label": pert_label,
                **stats,
            })

    return results
