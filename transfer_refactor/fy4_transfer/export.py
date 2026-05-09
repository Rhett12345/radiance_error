# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

from .model import FitResult

Results = Dict[Tuple[str, str], Dict[str, FitResult]]


def save_coefficients(results: Results, output_coeffs: str) -> None:
    Path(output_coeffs).parent.mkdir(parents=True, exist_ok=True)
    with open(output_coeffs, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Source_Ch", "Target_Ch", "Direction", "Model",
            "Coeff_1", "Coeff_2", "Intercept", "R", "Residual_Std", "N"
        ])
        for (_src_pair, _tgt_pair), directions in results.items():
            for direction, fit in directions.items():
                source_ch, target_ch = direction.split("__", 1)[1].split("__to__")
                writer.writerow([
                    source_ch,
                    target_ch,
                    direction.split("__", 1)[0],
                    fit.model_type,
                    f"{fit.coef1:.8f}",
                    "" if fit.coef2 is None else f"{fit.coef2:.8f}",
                    f"{fit.intercept:.8f}",
                    f"{fit.r:.8f}",
                    f"{fit.residual_std:.8f}",
                    fit.n,
                ])
