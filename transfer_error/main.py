# -*- coding: utf-8 -*-
"""
Master control program — satellite radiance transfer-model sensitivity experiment.

Usage::

    from transfer_error import main
    main(["data1.csv", "data2.csv"], "coeffs.csv", "./output/")
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .io_utils import load_coefficients, load_radiance_data, save_results
from .plotting import make_all_figures
from .sensitivity import run_experiment


def main(
    data_csv_list: List[str],
    coeff_csv: str,
    output_dir: str,
) -> None:
    """Run the complete transfer-model sensitivity experiment.

    Parameters
    ----------
    data_csv_list : list of str
        Paths to MODTRAN-simulation CSV files.
    coeff_csv : str
        Path to the transfer-coefficient table.
    output_dir : str
        Directory for output.  Writes:
        - ``sensitivity_results.csv``  — one row per channel × perturbation
        - ``fig_sensitivity_fy4a.png``  — fy4a 2×2 multi-panel figure
        - ``fig_sensitivity_fy4b.png``  — fy4b 2×2 multi-panel figure
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data -------------------------------------------------
    print("=" * 60)
    print("Step 1/4: Loading radiance data ...")
    radiance_data = load_radiance_data(data_csv_list)
    n_channels = len(radiance_data)
    n_scenes_total = sum(len(v) for v in radiance_data.values())
    print(f"  loaded {n_channels} channels, {n_scenes_total} total rows")

    # ---- 2. Load coefficients -----------------------------------------
    print("\nStep 2/4: Loading transfer coefficients ...")
    coeff_list = load_coefficients(coeff_csv)
    print(f"  loaded {len(coeff_list)} coefficient rows")

    # ---- 3. Run experiment --------------------------------------------
    print("\nStep 3/4: Running Monte Carlo sensitivity experiment ...")
    results = run_experiment(radiance_data, coeff_list)
    n_rows = len(results)
    n_refl = sum(1 for r in results if r["ch_type"] == "reflective")
    n_ir = sum(1 for r in results if r["ch_type"] == "ir")
    print(f"  done: {n_rows} result rows  (reflective: {n_refl}, IR: {n_ir})")

    # ---- 4. Save & plot -----------------------------------------------
    print("\nStep 4/4: Saving outputs ...")
    results_csv = out / "sensitivity_results.csv"
    save_results(results, str(results_csv))
    print(f"  saved {results_csv}")

    make_all_figures(results, str(out))

    print("\n" + "=" * 60)
    print("Experiment complete.")
    print(f"Output directory: {out.resolve()}")
