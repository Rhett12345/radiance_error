#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience script — run the transfer-error sensitivity experiment.

Usage::

    cd ~/transfer_model
    python run_sensitivity.py

Or from anywhere::

    python run_sensitivity.py  \\
        --data "fy4a_fy4b_convolution/*_sat_rad.csv"  \\
        --coeff transfer_coeff_fy4a_fy4b_v1.csv         \\
        --outdir ./output
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

# Ensure the transfer_model/ parent is on sys.path so the transfer_error
# package is importable regardless of where the script is invoked.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from transfer_error.main import main


def _expand(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        if matched:
            files.extend(matched)
        elif Path(pat).is_file():
            files.append(pat)
    return sorted(set(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FY-4 radiance transfer-model sensitivity experiment"
    )
    parser.add_argument(
        "--data", nargs="+",
        default=[str(Path.home() / "transfer_model/fy4a_fy4b_convolution/*_sat_rad.csv")],
        help="Glob(s) or path(s) to radiance CSV files",
    )
    parser.add_argument(
        "--coeff",
        default=str(Path.home() / "transfer_model/transfer_coeff_fy4a_fy4b_v1.csv"),
        help="Path to the transfer-coefficient CSV",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path.cwd() / "output"),
        help="Output directory (default: ./output)",
    )
    args = parser.parse_args()

    data_files = _expand(args.data)
    if not data_files:
        print("No data files found.  Patterns used:")
        for p in args.data:
            print(f"  {p}")
        raise SystemExit(1)

    print(f"Data files:      {len(data_files)}")
    for f in data_files:
        print(f"  {f}")
    print(f"Coefficient CSV: {args.coeff}")
    print(f"Output dir:      {args.outdir}")
    print()

    main(data_files, args.coeff, args.outdir)
