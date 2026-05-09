# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import glob
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .channel_catalog import wavelength_um
from .config import RunConfig

DirectionData = Dict[str, Dict[str, List[float]]]
ChannelData = Dict[Tuple[str, str], DirectionData]


def expand_input_files(input_globs: Iterable[str]) -> List[str]:
    files: List[str] = []
    for pat in input_globs:
        matched = sorted(glob.glob(pat))
        if matched:
            files.extend(matched)
        else:
            # Keep literal file paths if the user gave one explicit file that exists later
            # relative to another cwd; runner will print a useful warning if it cannot open.
            if Path(pat).is_file():
                files.append(pat)
    # deduplicate while keeping order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def init_channel_data(channel_pairs: List[Tuple[str, str]], src_to_tgt: str, tgt_to_src: str) -> ChannelData:
    data: ChannelData = {}
    for src_ch, tgt_ch in channel_pairs:
        data[(src_ch, tgt_ch)] = {
            src_to_tgt: {"x": [], "y": []},
            tgt_to_src: {"x": [], "y": []},
        }
    return data


def choose_field_suffix(fields: List[str], cfg: RunConfig, file_path: str) -> str | None:
    if cfg.data_field.lower() != "auto":
        if any(name.endswith(f"_{cfg.data_field}") for name in fields):
            return cfg.data_field
        if cfg.fallback_field and any(name.endswith(f"_{cfg.fallback_field}") for name in fields):
            print(f"Warning: {file_path} has no {cfg.data_field}; using {cfg.fallback_field} instead")
            return cfg.fallback_field
        return None

    for suffix in ["Radiance", "BT"]:
        if any(name.endswith(f"_{suffix}") for name in fields):
            return suffix
    return None


def convert_value(raw: str, channel_name: str, suffix: str, cfg: RunConfig) -> float:
    val = float(raw)
    if suffix == "Radiance" and cfg.convert_modtran_radiance:
        lam = wavelength_um(channel_name)
        val = val * 1e8 / (lam ** 2)
    return val


def load_and_merge_data(cfg: RunConfig, channel_pairs: List[Tuple[str, str]]) -> ChannelData:
    src_to_tgt = f"{cfg.source_sat.upper()}2{cfg.target_sat.upper()}"
    tgt_to_src = f"{cfg.target_sat.upper()}2{cfg.source_sat.upper()}"
    channel_data = init_channel_data(channel_pairs, src_to_tgt, tgt_to_src)

    files = expand_input_files(cfg.input_globs)
    if not files:
        print("Warning: no input CSV files found. Patterns were:")
        for pat in cfg.input_globs:
            print(f"  - {pat}")
        return channel_data

    print(f"Found {len(files)} input files")
    for file_path in files:
        if not Path(file_path).exists():
            print(f"Warning: File not found, skipping {file_path}")
            continue

        try:
            with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []
                suffix = choose_field_suffix(fields, cfg, file_path)
                if suffix is None:
                    print(f"Warning: no usable data field in {file_path}, skipping")
                    continue

                for row in reader:
                    for src_ch, tgt_ch in channel_pairs:
                        src_field = f"{src_ch}_{suffix}"
                        tgt_field = f"{tgt_ch}_{suffix}"
                        if src_field not in fields or tgt_field not in fields:
                            continue
                        try:
                            src_val = convert_value(row[src_field], src_ch, suffix, cfg)
                            tgt_val = convert_value(row[tgt_field], tgt_ch, suffix, cfg)
                            if not (math.isfinite(src_val) and math.isfinite(tgt_val)):
                                continue
                        except (ValueError, TypeError, KeyError):
                            continue

                        channel_data[(src_ch, tgt_ch)][src_to_tgt]["x"].append(src_val)
                        channel_data[(src_ch, tgt_ch)][src_to_tgt]["y"].append(tgt_val)
                        channel_data[(src_ch, tgt_ch)][tgt_to_src]["x"].append(tgt_val)
                        channel_data[(src_ch, tgt_ch)][tgt_to_src]["y"].append(src_val)
        except Exception as exc:
            print(f"Error processing file {file_path}: {exc}")

    print("\nData counts:")
    for (src_ch, tgt_ch), directions in channel_data.items():
        for direction, xy in directions.items():
            n = len(xy["x"])
            print(f"  {src_ch} <-> {tgt_ch} {direction}: {n} samples")
    return channel_data
