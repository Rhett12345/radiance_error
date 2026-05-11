# -*- coding: utf-8 -*-
"""
Physical constants, channel metadata, and experiment configuration.

All radiation constants use the µm-compatible formulation so that
wavelengths can be expressed directly in µm throughout the code.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (SI base)
# ---------------------------------------------------------------------------
C = 2.99792458e8          # speed of light            [m·s⁻¹]
H = 6.62607015e-34        # Planck constant           [J·s]
K = 1.380649e-23          # Boltzmann constant        [J·K⁻¹]

# Radiation constants — µm formulation
# c1  = 2·h·c² × 10²⁴       ⇒  W·µm⁴·m⁻²·sr⁻¹
# c2  = h·c/k  × 10⁶        ⇒  µm·K
C1_UM = 2.0 * H * C * C * 1e24       # = 1.191042972e8
C2_UM = (H * C / K) * 1e6          # = 1.438776877e4

# ---------------------------------------------------------------------------
# Channel centre wavelengths (µm) — AGRI nominal
# Source: FY-4A/B/C AGRI instrument documentation.
# Keys:  "sat_ch"  (e.g. "fy4a_ch01")
# ---------------------------------------------------------------------------
CHANNEL_WAVELENGTH: dict[str, float] = {}

# -- FY-4A ----------------------------------------------------------------
_fy4a = {
    "ch01": 0.47,    # blue
    "ch02": 0.65,    # red
    "ch03": 0.825,   # NIR
    "ch04": 1.375,   # cirrus
    "ch05": 1.61,    # snow / low cloud
    "ch06": 2.25,    # SWIR
    "ch07": 3.75,    # MWIR (high gain)
    "ch08": 3.75,    # MWIR (low gain)  — excluded from IR analysis
    "ch09": 6.25,    # upper water vapour
    "ch10": 7.10,    # middle water vapour
    "ch11": 8.50,    # IR
    "ch12": 10.80,   # window IR
    "ch13": 12.00,   # split-window IR
    "ch14": 13.50,   # CO₂ / cloud
}
for _ch, _lam in _fy4a.items():
    CHANNEL_WAVELENGTH[f"fy4a_{_ch}"] = _lam

# -- FY-4B ----------------------------------------------------------------
_fy4b = {
    "ch01": 0.47,
    "ch02": 0.65,
    "ch03": 0.825,
    "ch04": 1.379,   # note: slightly different from A
    "ch05": 1.61,
    "ch06": 2.25,
    "ch07": 3.75,
    "ch08": 3.75,
    "ch09": 6.25,
    "ch10": 6.95,    # note: different from A (7.10)
    "ch11": 7.42,    # note: A has no 7.42; A ch11 = 8.50
    "ch12": 8.55,    # note: A ch11 analogue
    "ch13": 10.80,   # note: A ch12 analogue
    "ch14": 12.00,   # note: A ch13 analogue
    "ch15": 13.30,   # note: A ch14 analogue
}
for _ch, _lam in _fy4b.items():
    CHANNEL_WAVELENGTH[f"fy4b_{_ch}"] = _lam

# -- FY-4C ----------------------------------------------------------------
_fy4c = {
    "ch01": 0.47,
    "ch02": 0.525,
    "ch03": 0.65,
    "ch04": 0.65,
    "ch05": 0.65,
    "ch06": 0.825,
    "ch07": 1.379,
    "ch08": 1.61,
    "ch09": 2.225,
    "ch10": 3.75,
    "ch11": 4.05,
    "ch12": 6.25,
    "ch13": 6.95,
    "ch14": 7.42,
    "ch15": 8.55,
    "ch16": 9.61,
    "ch17": 10.80,
    "ch18": 12.00,
    "ch19": 13.30,
}
for _ch, _lam in _fy4c.items():
    CHANNEL_WAVELENGTH[f"fy4c_{_ch}"] = _lam

# ---------------------------------------------------------------------------
# Channel type classification
#   reflective : solar reflective bands  (ch01 – ch06)
#   ir         : thermal infrared bands  (ch07, ch09 – ch14/15)
# ch08 is explicitly excluded — same wavelength as ch07 (low-gain duplicate).
# ---------------------------------------------------------------------------
_REFL_CH_NUMS = {1, 2, 3, 4, 5, 6}


def channel_type(channel_name: str) -> str:
    """Return 'reflective' or 'ir' for a channel full-name like 'fy4a_ch01'."""
    try:
        ch_num = int(channel_name.rsplit("_", 1)[-1].replace("ch", ""))
    except (ValueError, IndexError):
        raise ValueError(f"Cannot parse channel number from {channel_name!r}")
    if ch_num in _REFL_CH_NUMS:
        return "reflective"
    if ch_num == 8:
        raise ValueError(
            f"Channel {channel_name} (ch08) is a low-gain duplicate and "
            f"should not be used in this experiment."
        )
    return "ir"


def get_wavelength_um(channel_name: str) -> float:
    """Return the nominal centre wavelength in µm for *channel_name*."""
    try:
        return CHANNEL_WAVELENGTH[channel_name.lower()]
    except KeyError:
        raise KeyError(f"Unknown channel {channel_name!r}")


# ---------------------------------------------------------------------------
# Perturbation grids
# ---------------------------------------------------------------------------
PERTURBATION_FRAC: list[float] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
PERTURBATION_DK: list[float]   = [0.5,  1.0,  1.5,  2.0]

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
N_MC: int = 500
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Jacobian scale factor  (W·cm⁻²·sr⁻¹·cm → W·m⁻²·sr⁻¹·μm⁻¹)
#   L_λ = L_ν × JAC_SCALE / λ²
#   1e4  :  cm⁻² → m⁻²
#   1e4  :  |dν/dλ| = 1e4 / λ²   (ν in cm⁻¹, λ in µm)
# ---------------------------------------------------------------------------
JAC_SCALE: float = 1e8

# ---------------------------------------------------------------------------
# Output / plot defaults
# ---------------------------------------------------------------------------
OUTPUT_DPI: int = 300
