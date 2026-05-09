# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from .channel_catalog import normalize_sat


@dataclass
class RunConfig:
    pair: str
    source_sat: str
    target_sat: str
    input_globs: List[str]
    output_coeffs: str
    plot_dir: str
    r_threshold: float = 0.98
    polynomial_degree: int = 2
    data_field: str = "Radiance"       # 'Radiance', 'BT', or 'auto'
    fallback_field: Optional[str] = "BT"
    convert_modtran_radiance: bool = True
    make_plots: bool = True
    channel_pair_overrides: Optional[List[List[str]]] = None


def pair_to_sats(pair: str) -> tuple[str, str]:
    p = pair.strip().lower().replace("-", "").replace("_", "")
    if len(p) != 2:
        raise ValueError("pair must look like ac, bc, ab, ca, cb, or ba")
    return normalize_sat(p[0]), normalize_sat(p[1])


def default_config(pair: str) -> RunConfig:
    src, tgt = pair_to_sats(pair)
    short = f"{src[-1]}{tgt[-1]}"
    conv_dir = f"../convolution_result/{src}_{tgt}_convolution"
    return RunConfig(
        pair=short,
        source_sat=src,
        target_sat=tgt,
        input_globs=[f"{conv_dir}/*_sat_rad.csv"],
        output_coeffs=f"../transfer_result/transfer_coeff_{src}_{tgt}.csv",
        plot_dir=f"../fitting_result/line_plots_{src}_{tgt}",
    )


def load_config(path: str | Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Allow either explicit source/target or a compact pair string.
    if "source_sat" not in data or "target_sat" not in data:
        src, tgt = pair_to_sats(data.get("pair", ""))
        data["source_sat"] = src
        data["target_sat"] = tgt
    else:
        data["source_sat"] = normalize_sat(data["source_sat"])
        data["target_sat"] = normalize_sat(data["target_sat"])

    if "pair" not in data:
        data["pair"] = f"{data['source_sat'][-1]}{data['target_sat'][-1]}"

    return RunConfig(**data)


def save_config_template(cfg: RunConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)
