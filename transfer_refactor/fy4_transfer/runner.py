# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from .channel_catalog import build_channel_pairs
from .config import RunConfig
from .export import Results, save_coefficients
from .io import load_and_merge_data
from .model import fit_model
from .plotting import configure_plot_style, plot_regression


def direction_key(direction_label: str, src_ch: str, tgt_ch: str) -> str:
    # Encode source/target into the key so export does not need to infer it.
    return f"{direction_label}__{src_ch}__to__{tgt_ch}"


def run(cfg: RunConfig) -> Results:
    print("Start FY-4 transfer fitting")
    print(f"Pair: {cfg.source_sat.upper()} -> {cfg.target_sat.upper()}  (bidirectional fitting enabled)")

    channel_pairs = build_channel_pairs(
        cfg.source_sat,
        cfg.target_sat,
        overrides=cfg.channel_pair_overrides,
    )
    print(f"Using {len(channel_pairs)} channel pairs")
    for a, b in channel_pairs:
        print(f"  {a} <-> {b}")

    if cfg.make_plots:
        configure_plot_style()
        Path(cfg.plot_dir).mkdir(parents=True, exist_ok=True)

    channel_data = load_and_merge_data(cfg, channel_pairs)

    results: Results = {}
    for (src_pair_ch, tgt_pair_ch), directions in channel_data.items():
        results[(src_pair_ch, tgt_pair_ch)] = {}
        for raw_direction, xy in directions.items():
            x, y = xy["x"], xy["y"]
            if not x or not y:
                print(f"Warning: no data for {src_pair_ch} <-> {tgt_pair_ch} {raw_direction}")
                continue

            fit = fit_model(x, y, r_threshold=cfg.r_threshold, polynomial_degree=cfg.polynomial_degree)
            if fit is None:
                print(f"Warning: insufficient data for {src_pair_ch} <-> {tgt_pair_ch} {raw_direction}")
                continue

            # Determine actual x/y channel names from raw direction.
            if raw_direction == f"{cfg.source_sat.upper()}2{cfg.target_sat.upper()}":
                source_ch, target_ch = src_pair_ch, tgt_pair_ch
            else:
                source_ch, target_ch = tgt_pair_ch, src_pair_ch

            key = direction_key(raw_direction, source_ch, target_ch)
            results[(src_pair_ch, tgt_pair_ch)][key] = fit
            print(
                f"Fit {source_ch} -> {target_ch}: {fit.model_type}, "
                f"R={fit.r:.5f}, sigma={fit.residual_std:.5f}, n={fit.n}"
            )

            if cfg.make_plots:
                plot_path = plot_regression(x, y, fit, source_ch, target_ch, raw_direction, cfg.plot_dir)
                if plot_path:
                    print(f"  saved plot: {plot_path}")

    save_coefficients(results, cfg.output_coeffs)
    print(f"Saved coefficients: {cfg.output_coeffs}")
    if cfg.make_plots:
        print(f"Saved plots in: {cfg.plot_dir}")
    print("Done")
    return results
