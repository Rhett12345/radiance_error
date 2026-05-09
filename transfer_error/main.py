# -*- coding: utf-8 -*-
"""
Master control program — satellite radiance transfer-model sensitivity experiment.

Usage::

    from main import main
    main(["data1.csv", "data2.csv"], "coeffs.csv", "./output/")
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .config import OUTPUT_DPI
from .io_utils import load_coefficients, load_radiance_data, save_results
from .plotting import plot_ir, plot_reflective
from .sensitivity import run_experiment


def main(
    data_csv_list: List[str],
    coeff_csv: str,
    output_dir: str,
) -> None:
    """Run the complete sensitivity experiment.

    Parameters
    ----------
    data_csv_list : list of str
        Paths to MODTRAN-simulation CSV files.  Each file contains columns
        like ``fy4a_ch01_Radiance`` (W·cm⁻²·sr⁻¹·cm).  Files may hold
        different channel subsets (reflective / IR).
    coeff_csv : str
        Path to the transfer-coefficient table
        (e.g. ``transfer_coeff_fy4a_fy4b_v1.csv``).
    output_dir : str
        Directory for output files.  Two files are written:
        - ``sensitivity_results.csv``
        - ``fig_reflective.png``
        - ``fig_ir.png``
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
    print("\nStep 4/4: Saving output ...")
    results_csv = out / "sensitivity_results.csv"
    save_results(results, str(results_csv))
    print(f"  saved {results_csv}")

    fig_refl = str(out / "fig_reflective.png")
    path_refl = plot_reflective(results, fig_refl)
    if path_refl:
        print(f"  saved {path_refl}")

    fig_ir = str(out / "fig_ir.png")
    path_ir = plot_ir(results, fig_ir)
    if path_ir:
        print(f"  saved {path_ir}")

    print("\n" + "=" * 60)
    print("Experiment complete.")
    print(f"Output directory: {out.resolve()}")
