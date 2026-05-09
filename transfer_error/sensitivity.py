# -*- coding: utf-8 -*-
"""
Monte-Carlo sensitivity experiment.

Core logic:  add noise to source-channel radiance → run transfer model →
measure output uncertainty.
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
    """Evaluate the transfer model.

    ``y = coeff2·x² + coeff1·x + intercept``  (quadratic if coeff2 is given,
    otherwise linear).
    """
    y = coeff1 * x + intercept
    if coeff2 is not None:
        y += coeff2 * x * x
    return y


# ---------- Monte-Carlo runner for a single channel ------------------------

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
    """Run N_MC iterations for a single channel × perturbation combination.

    Parameters
    ----------
    x_src : ndarray, shape (N_scenes,)
        Jacobian-transformed source radiances.
    lam_um : float
        Channel wavelength in µm.
    ch_type : {'reflective', 'ir'}
    coeff1, coeff2, intercept : float
        Transfer-model parameters.
    perturbation : float
        Fraction (reflective) or ΔK (IR).
    rng : Generator
        Seeded NumPy random generator.

    Returns
    -------
    dict with keys dy_mean, dy_p95, rel_err_mean, dTb_mean, dTb_p95.
    """
    n_scenes = len(x_src)
    y_clean = apply_model(x_src, coeff1, coeff2, intercept)  # (N_scenes,)

    # --- noise sigma ----------------------------------------------------
    if ch_type == "reflective":
        sigma = perturbation * x_src          # scene-dependent, shape (N_scenes,)
    else:
        # IR: constant sigma based on typical Tb of the mean scene
        Tb_typ = brightness_temperature(
            np.atleast_1d(np.mean(x_src)), lam_um
        )[0]
        sigma_val = abs(dL_dTb(Tb_typ, lam_um)) * perturbation
        sigma = np.full(n_scenes, sigma_val)

    # Clip sigma at a sensible floor to avoid degenerate noise
    sigma = np.maximum(sigma, 1e-12)

    # --- Monte Carlo -----------------------------------------------------
    # noise: (N_MC, N_scenes)
    noise = rng.normal(0.0, sigma, size=(N_MC, n_scenes))
    x_noisy = x_src[np.newaxis, :] + noise
    x_noisy = np.maximum(x_noisy, 1e-10)      # radiance must be > 0

    y_noisy = apply_model(x_noisy, coeff1, coeff2, intercept)  # (N_MC, N_scenes)
    dy = np.abs(y_noisy - y_clean[np.newaxis, :])               # (N_MC, N_scenes)

    dy_mean = float(np.mean(dy))
    dy_p95  = float(np.percentile(dy, 95))
    mean_y_clean = float(np.mean(y_clean))
    rel_err_mean = (dy_mean / mean_y_clean * 100.0) if mean_y_clean > 0 else np.nan

    # --- delta-Tb (IR only) ---------------------------------------------
    dTb_mean: float = np.nan
    dTb_p95: float  = np.nan
    if ch_type == "ir":
        Tb_noisy = brightness_temperature(y_noisy, lam_um)
        Tb_clean = brightness_temperature(y_clean, lam_um)  # (N_scenes,)
        dTb = np.abs(Tb_noisy - Tb_clean[np.newaxis, :])
        dTb_mean = float(np.nanmean(dTb))
        dTb_p95  = float(np.nanpercentile(dTb, 95))

    return {
        "dy_mean": dy_mean,
        "dy_p95": dy_p95,
        "rel_err_mean": rel_err_mean,
        "dTb_mean": dTb_mean,
        "dTb_p95": dTb_p95,
    }


# ---------- main experiment ------------------------------------------------

def run_experiment(
    radiance_data: Dict[str, np.ndarray],
    coeff_list: List[dict],
) -> List[dict]:
    """Run the full sensitivity experiment.

    Parameters
    ----------
    radiance_data : dict
        channel_name → 1-D array of radiance (original units).
    coeff_list : list of dict
        Transfer coefficients from :func:`~io_utils.load_coefficients`.

    Returns
    -------
    list of dict
        One dict per channel × perturbation combination.
        Keys: channel, ch_type, perturbation, dy_mean, dy_p95,
              rel_err_mean, dTb_mean, dTb_p95.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    results: List[dict] = []

    for coeff in coeff_list:
        src_ch = coeff["source_ch"]

        # --- guard: skip if no radiance data for this source channel ----
        if src_ch not in radiance_data:
            print(f"  skip {src_ch} — not found in radiance data")
            continue

        try:
            lam_um = get_wavelength_um(src_ch)
            cht = channel_type(src_ch)
        except (KeyError, ValueError) as exc:
            print(f"  skip {src_ch} — {exc}")
            continue

        # Jacobian transform: original → W·m⁻²·sr⁻¹·µm⁻¹
        x_orig = radiance_data[src_ch]
        x_jac = jacobian_radiance(x_orig, lam_um)

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
