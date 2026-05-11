# -*- coding: utf-8 -*-
"""
Data I/O: load simulation radiance CSVs, load / save coefficient & result tables.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import get_wavelength_um, JAC_SCALE


# ---------- radiance data ---------------------------------------------------

def load_radiance_data(csv_list: List[str]) -> Dict[str, np.ndarray]:
    """Load radiance values from multiple MODTRAN-style CSV files.

    Each CSV may contain a different subset of channels (e.g. some hold only
    reflective bands ch01–ch06 while others hold IR bands ch07+).  All columns
    matching ``*_Radiance`` are collected and returned as flat float64 arrays,
    keyed by the channel full-name (e.g. ``"fy4a_ch01"``).

    Radiance is kept in **original units** (W·cm⁻²·sr⁻¹·cm).  Callers are
    responsible for the Jacobian transform.
    """
    accum: Dict[str, List[float]] = {}

    for path_str in csv_list:
        p = Path(path_str)
        if not p.is_file():
            print(f"Warning: file not found, skipping: {path_str}")
            continue

        with open(p, "r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []

            rad_cols = [c for c in fields if c.endswith("_Radiance")]
            if not rad_cols:
                print(f"Warning: no *_Radiance columns in {path_str}, skipping")
                continue

            for row in reader:
                for col in rad_cols:
                    raw = row.get(col, "").strip()
                    if not raw:
                        continue
                    try:
                        val = float(raw)
                    except (ValueError, TypeError):
                        continue
                    if not math.isfinite(val) or val <= 0:
                        continue
                    ch_name = col.replace("_Radiance", "")
                    accum.setdefault(ch_name, []).append(val)

    out: Dict[str, np.ndarray] = {}
    for ch, vals in accum.items():
        out[ch] = np.asarray(vals, dtype=np.float64)
    return out


# ---------- coefficient table ------------------------------------------------

def load_coefficients(csv_path: str) -> List[dict]:
    """Load the transfer-coefficient table.

    Expected columns: Source_Ch, Target_Ch, Direction, Model,
                      Coeff_1, Coeff_2, Intercept, R, Residual_Std.

    Optional columns: S_t, Q_t, S_r, Q_r  — Planck correction coefficients
    for IR channels (default: 1, 0, 1, 0).  Absent columns → defaults.

    Returns a list of dicts, one per row, with keys:
        source_ch, target_ch, direction, model_type,
        coeff1, coeff2, intercept, residual_std,
        S_t, Q_t, S_r, Q_r
    """
    rows: List[dict] = []
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"Coefficient file not found: {csv_path}")

    with open(p, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            coeff2_raw = row.get("Coeff_2", "").strip()
            try:
                entry = {
                    "source_ch": row["Source_Ch"].strip(),
                    "target_ch": row["Target_Ch"].strip(),
                    "direction": row["Direction"].strip(),
                    "model_type": row["Model"].strip(),
                    "coeff1": float(row["Coeff_1"].strip()),
                    "coeff2": float(coeff2_raw) if coeff2_raw else None,
                    "intercept": float(row["Intercept"].strip()),
                    "S_t": float(row["S_t"].strip()) if row.get("S_t", "").strip() else 1.0,
                    "Q_t": float(row["Q_t"].strip()) if row.get("Q_t", "").strip() else 0.0,
                    "S_r": float(row["S_r"].strip()) if row.get("S_r", "").strip() else 1.0,
                    "Q_r": float(row["Q_r"].strip()) if row.get("Q_r", "").strip() else 0.0,
                    "residual_std": float(row["Residual_Std"].strip()) if row.get("Residual_Std", "").strip() else 0.0,
                }
            except (KeyError, ValueError) as exc:
                print(f"Warning: malformed coefficient row {row}, error: {exc}")
                continue
            rows.append(entry)
    return rows


# ---------- results ----------------------------------------------------------

_RESULT_COLUMNS = [
    "channel",
    "ch_type",
    "perturbation",
    "perturbation_label",
    "dy_mean",
    "dy_p95",
    "dy_std",
    "dy_rms",
    "rel_err_mean",
    "rel_err_p95",
    "dTb_mean",
    "dTb_p95",
    "dTb_std",
]


def save_results(results: List[dict], output_path: str) -> None:
    """Write sensitivity results (one row per channel × perturbation)."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RESULT_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
