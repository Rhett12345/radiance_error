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
    S_t: float = 1.0,
    Q_t: float = 0.0,
    S_r: float = 1.0,
    Q_r: float = 0.0,
    residual_std: float = 0.0,
) -> dict:
    """Monte Carlo for one channel × perturbation combination.

    Parameters
    ----------
    S_t, Q_t : float
        Target-channel Planck correction:  T_b,0 = (T_b,t - Q_t) / S_t.
    S_r, Q_r : float
        Reference-channel Planck correction:  T_b,r = S_r · T_b,1 + Q_r.
    residual_std : float
        Transfer-model residual std (in target-radiance units).
        Added as independent Gaussian noise after model evaluation,
        representing model-fitting uncertainty.

    Returns
    -------
    dict
        dy_mean, dy_p95, dy_std, dy_rms, rel_err_mean, rel_err_p95,
        dTb_mean, dTb_p95, dTb_std (NaN for reflective).
    """
    n_scenes = len(x_src)
    y_clean = apply_model(x_src, coeff1, coeff2, intercept)  # (N_scenes,)

    # --- noise sigma ----------------------------------------------------
    if ch_type == "reflective":
        sigma = perturbation * x_src
    else:
        # Per-pixel BT from source radiance → per-pixel dL/dT
        Tb0 = brightness_temperature(x_src, lam_um)            # (n_scenes,)
        dL_dT_local = np.abs(dL_dTb(Tb0, lam_um))             # (n_scenes,)
        # σ_{T_b,0} = σ_{T_b,t} / S_t   →   σ_rad = |dL/dT| · (perturbation / S_t)
        sigma = dL_dT_local * (perturbation / S_t)

    sigma = np.maximum(sigma, 1e-12)

    # --- Monte Carlo (vectorised) ---------------------------------------
    noise = rng.normal(0.0, sigma, size=(N_MC, n_scenes))
    x_noisy = x_src[np.newaxis, :] + noise
    x_noisy = np.maximum(x_noisy, 1e-10)

    y_noisy = apply_model(x_noisy, coeff1, coeff2, intercept)  # (N_MC, N_scenes)

    # Superimpose transfer-model residual uncertainty (independent of input noise)
    if residual_std > 0:
        y_noisy = y_noisy + rng.normal(0.0, residual_std, y_noisy.shape)

    # Signed error for std / RMS (before absolute value)
    error = y_noisy - y_clean[np.newaxis, :]                    # (N_MC, N_scenes)
    dy = np.abs(error)                                          # (N_MC, N_scenes)

    dy_mean = float(np.mean(dy))
    dy_p95  = float(np.percentile(dy, 95))
    dy_std  = float(np.std(error))
    dy_rms  = float(np.sqrt(np.mean(error ** 2)))
    mean_y_clean = float(np.mean(y_clean))
    rel_err_mean = (dy_mean / mean_y_clean * 100.0) if mean_y_clean > 0 else np.nan

    # rel_err_p95 uses the scene-average y_clean as denominator
    rel_err_p95 = (dy_p95 / mean_y_clean * 100.0) if mean_y_clean > 0 else np.nan

    # --- delta-Tb (IR only) ---------------------------------------------
    dTb_mean: float = np.nan
    dTb_p95: float  = np.nan
    dTb_std: float  = np.nan
    if ch_type == "ir":
        # Invert to T_b,1 then apply reference-channel correction
        Tb1_noisy = brightness_temperature(y_noisy, lam_um)
        Tb1_clean = brightness_temperature(y_clean, lam_um)
        Tbr_noisy = S_r * Tb1_noisy + Q_r
        Tbr_clean = S_r * Tb1_clean + Q_r
        error_Tb = Tbr_noisy - Tbr_clean[np.newaxis, :]
        dTb = np.abs(error_Tb)
        dTb_mean = float(np.nanmean(dTb))
        dTb_p95  = float(np.nanpercentile(dTb, 95))
        dTb_std  = float(np.nanstd(error_Tb))

    return {
        "dy_mean": dy_mean,
        "dy_p95": dy_p95,
        "dy_std": dy_std,
        "dy_rms": dy_rms,
        "rel_err_mean": rel_err_mean,
        "rel_err_p95": rel_err_p95,
        "dTb_mean": dTb_mean,
        "dTb_p95": dTb_p95,
        "dTb_std": dTb_std,
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
                S_t=coeff.get("S_t", 1.0),
                Q_t=coeff.get("Q_t", 0.0),
                S_r=coeff.get("S_r", 1.0),
                Q_r=coeff.get("Q_r", 0.0),
                residual_std=coeff.get("residual_std", 0.0),
            )
            results.append({
                "channel": src_ch,
                "ch_type": cht,
                "perturbation": pert,
                "perturbation_label": pert_label,
                **stats,
            })

    return results
