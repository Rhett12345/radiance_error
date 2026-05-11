# -*- coding: utf-8 -*-
"""
Radiative-transfer utility functions.

All functions operate on wavelengths expressed in micrometres (µm).
Radiance is always  W·m⁻²·sr⁻¹·µm⁻¹  unless stated otherwise.
"""

from __future__ import annotations

import numpy as np

from .config import C1_UM, C2_UM, JAC_SCALE


# ---------- Jacobian (unit conversion) --------------------------------------

def jacobian_radiance(L_nu: np.ndarray, lam_um: float) -> np.ndarray:
    """Convert radiance from wavenumber to wavelength domain.

    .. math::
        L_λ  =  L_ν  ×  1e8  /  λ²

    Parameters
    ----------
    L_nu : array_like
        Radiance in W·cm⁻²·sr⁻¹·cm  (per-wavenumber).
    lam_um : float
        Channel centre wavelength in µm.

    Returns
    -------
    ndarray
        Radiance in W·m⁻²·sr⁻¹·µm⁻¹.
    """
    return np.asarray(L_nu, dtype=np.float64) * (JAC_SCALE / (lam_um ** 2))


# ---------- Planck function -------------------------------------------------

def planck_radiance(T: np.ndarray, lam_um: float) -> np.ndarray:
    """Spectral radiance from the Planck function (wavelength form).

    .. math::
        B_λ(T) = c1 / λ⁵  ·  1 / [exp(c2 / (λ·T)) − 1]

    Parameters
    ----------
    T : array_like
        Temperature in K.
    lam_um : float
        Wavelength in µm.

    Returns
    -------
    ndarray
        Radiance in W·m⁻²·sr⁻¹·µm⁻¹.
    """
    T_arr = np.asarray(T, dtype=np.float64)
    lam5 = lam_um ** 5
    arg = C2_UM / (lam_um * T_arr)
    # Clip arg to avoid overflow in exp
    arg = np.clip(arg, None, 500.0)
    return C1_UM / lam5 / (np.exp(arg) - 1.0)


# ---------- brightness temperature ------------------------------------------

def brightness_temperature(L: np.ndarray, lam_um: float) -> np.ndarray:
    """Invert the Planck function to obtain brightness temperature.

    .. math::
        T_b = c2 / [λ · ln(1 + c1 / (λ⁵·L))]

    Parameters
    ----------
    L : array_like
        Radiance in W·m⁻²·sr⁻¹·µm⁻¹.
    lam_um : float
        Wavelength in µm.

    Returns
    -------
    ndarray
        Brightness temperature in K.
    """
    L_arr = np.asarray(L, dtype=np.float64)
    lam5 = lam_um ** 5
    ratio = C1_UM / (lam5 * L_arr)
    # Guard against log(≤0) for unphysical radiance
    Tb = np.full_like(L_arr, np.nan)
    valid = (L_arr > 0) & np.isfinite(L_arr)
    Tb[valid] = C2_UM / (lam_um * np.log1p(ratio[valid]))
    return Tb


# ---------- dL / dTb --------------------------------------------------------

def dL_dTb(T, lam_um):
    """Derivative of the Planck radiance w.r.t. temperature (vectorised).

    .. math::
        ∂L/∂T = (c1·c2 / λ⁶·T²) · exp(c2/λT) / [exp(c2/λT) − 1]²

    Parameters
    ----------
    T : float or ndarray
        Temperature in K.
    lam_um : float
        Wavelength in µm.

    Returns
    -------
    float or ndarray
        ∂L/∂T  in  W·m⁻²·sr⁻¹·µm⁻¹·K⁻¹.
    """
    T_arr = np.asarray(T, dtype=np.float64)
    scalar = T_arr.ndim == 0
    lam6 = lam_um ** 6
    x = C2_UM / (lam_um * T_arr)
    x = np.clip(x, None, 500.0)
    ex = np.exp(x)
    result = (C1_UM * C2_UM) / (lam6 * T_arr * T_arr) * ex / ((ex - 1.0) ** 2)
    return float(result) if scalar else result
