# -*- coding: utf-8 -*-
"""FY-4 AGRI channel metadata and reusable pairing presets.

The goal of this file is to make satellite-pair switching a data problem,
not a code-editing problem. Add/modify channel metadata here once, then run
any pair with `python run_fit.py --pair ac|bc|ab`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Channel:
    sat: str              # fy4a / fy4b / fy4c
    ch: str               # ch01 / ch02 ...
    wavelength_um: float
    label: str = ""       # human-readable usage note

    @property
    def full_name(self) -> str:
        return f"{self.sat}_{self.ch}"

    @property
    def title(self) -> str:
        n = int(self.ch.replace("ch", ""))
        return f"Channel {n} ({self.wavelength_um:g} μm)"


SAT_ALIASES = {
    "a": "fy4a", "4a": "fy4a", "fy4a": "fy4a",
    "b": "fy4b", "4b": "fy4b", "fy4b": "fy4b",
    "c": "fy4c", "4c": "fy4c", "fy4c": "fy4c",
}


def normalize_sat(s: str) -> str:
    key = s.strip().lower().replace("-", "")
    if key not in SAT_ALIASES:
        raise ValueError(f"Unknown satellite {s!r}. Use one of: a, b, c, fy4a, fy4b, fy4c")
    return SAT_ALIASES[key]


# Wavelengths follow AGRI nominal channel centers. Keep both A/B 3.75 high/low
# channels in the catalog; pair presets decide whether to use one or both.
CHANNELS: Dict[str, Dict[str, Channel]] = {
    "fy4a": {
        "ch01": Channel("fy4a", "ch01", 0.47, "blue"),
        "ch02": Channel("fy4a", "ch02", 0.65, "red"),
        "ch03": Channel("fy4a", "ch03", 0.825, "nir"),
        "ch04": Channel("fy4a", "ch04", 1.375, "cirrus"),
        "ch05": Channel("fy4a", "ch05", 1.61, "snow/low cloud"),
        "ch06": Channel("fy4a", "ch06", 2.25, "swir"),
        "ch07": Channel("fy4a", "ch07", 3.75, "mwir high gain"),
        "ch08": Channel("fy4a", "ch08", 3.75, "mwir low gain"),
        "ch09": Channel("fy4a", "ch09", 6.25, "upper water vapor"),
        "ch10": Channel("fy4a", "ch10", 7.10, "middle water vapor"),
        "ch11": Channel("fy4a", "ch11", 8.50, "ir"),
        "ch12": Channel("fy4a", "ch12", 10.80, "window ir"),
        "ch13": Channel("fy4a", "ch13", 12.00, "split window ir"),
        "ch14": Channel("fy4a", "ch14", 13.50, "co2/cloud"),
    },
    "fy4b": {
        "ch01": Channel("fy4b", "ch01", 0.47, "blue"),
        "ch02": Channel("fy4b", "ch02", 0.65, "red"),
        "ch03": Channel("fy4b", "ch03", 0.825, "nir"),
        "ch04": Channel("fy4b", "ch04", 1.379, "cirrus"),
        "ch05": Channel("fy4b", "ch05", 1.61, "snow/low cloud"),
        "ch06": Channel("fy4b", "ch06", 2.25, "swir"),
        "ch07": Channel("fy4b", "ch07", 3.75, "mwir high gain"),
        "ch08": Channel("fy4b", "ch08", 3.75, "mwir low gain"),
        "ch09": Channel("fy4b", "ch09", 6.25, "upper water vapor"),
        "ch10": Channel("fy4b", "ch10", 6.95, "middle water vapor"),
        "ch11": Channel("fy4b", "ch11", 7.42, "lower water vapor"),
        "ch12": Channel("fy4b", "ch12", 8.55, "ir"),
        "ch13": Channel("fy4b", "ch13", 10.80, "window ir"),
        "ch14": Channel("fy4b", "ch14", 12.00, "split window ir"),
        "ch15": Channel("fy4b", "ch15", 13.30, "co2/cloud"),
    },
    "fy4c": {
        "ch01": Channel("fy4c", "ch01", 0.47, "blue"),
        "ch02": Channel("fy4c", "ch02", 0.525, "green / true color"),
        "ch03": Channel("fy4c", "ch03", 0.65, "pan 0.4-0.9 μm"),
        "ch04": Channel("fy4c", "ch04", 0.65, "red broad"),
        "ch05": Channel("fy4c", "ch05", 0.65, "red narrow"),
        "ch06": Channel("fy4c", "ch06", 0.825, "nir"),
        "ch07": Channel("fy4c", "ch07", 1.379, "cirrus"),
        "ch08": Channel("fy4c", "ch08", 1.61, "snow/low cloud"),
        "ch09": Channel("fy4c", "ch09", 2.225, "swir"),
        "ch10": Channel("fy4c", "ch10", 3.75, "mwir"),
        "ch11": Channel("fy4c", "ch11", 4.05, "fire"),
        "ch12": Channel("fy4c", "ch12", 6.25, "upper water vapor"),
        "ch13": Channel("fy4c", "ch13", 6.95, "middle water vapor"),
        "ch14": Channel("fy4c", "ch14", 7.42, "lower water vapor"),
        "ch15": Channel("fy4c", "ch15", 8.55, "ir"),
        "ch16": Channel("fy4c", "ch16", 9.61, "upper troposphere / ozone-like"),
        "ch17": Channel("fy4c", "ch17", 10.80, "window ir"),
        "ch18": Channel("fy4c", "ch18", 12.00, "split window ir"),
        "ch19": Channel("fy4c", "ch19", 13.30, "co2/cloud"),
    },
}

# Pair presets are intentionally explicit because some channels are not one-to-one:
# - C has extra channels (green, pan, narrow-red, fire, 9.61 μm)
# - A/B 3.75 μm high/low may map to one C 3.75 μm channel
# - A lacks the 7.42 μm low-level water-vapor channel present in B/C
# Store only the canonical forward order; build_channel_pairs() can reverse it.
PAIR_PRESETS: Dict[Tuple[str, str], List[Tuple[str, str]]] = {
    ("fy4a", "fy4c"): [
        ("ch01", "ch01"), ("ch02", "ch04"), ("ch03", "ch06"),
        ("ch04", "ch07"), ("ch05", "ch08"), ("ch06", "ch09"),
        ("ch07", "ch10"),
        # Add ("ch08", "ch10") here only if your AC CSV really contains fy4a_ch08_Radiance
        # and you want to fit the low-gain 3.75 μm channel separately.
        ("ch09", "ch12"), ("ch10", "ch13"), ("ch11", "ch15"),
        ("ch12", "ch17"), ("ch13", "ch18"), ("ch14", "ch19"),
    ],
    ("fy4b", "fy4c"): [
        ("ch01", "ch01"), ("ch02", "ch04"), ("ch03", "ch06"),
        ("ch04", "ch07"), ("ch05", "ch08"), ("ch06", "ch09"),
        ("ch07", "ch10"), ("ch08", "ch10"),
        ("ch09", "ch12"), ("ch10", "ch13"), ("ch11", "ch14"),
        ("ch12", "ch15"), ("ch13", "ch17"), ("ch14", "ch18"),
        ("ch15", "ch19"),
    ],
    ("fy4a", "fy4b"): [
        ("ch01", "ch01"), ("ch02", "ch02"), ("ch03", "ch03"),
        ("ch04", "ch04"), ("ch05", "ch05"), ("ch06", "ch06"),
        ("ch07", "ch07"),
        # Same note as above: enable ch08 only when your files contain it and you need it.
        # ("ch08", "ch08"),
        ("ch09", "ch09"), ("ch10", "ch10"), ("ch11", "ch12"),
        ("ch12", "ch13"), ("ch13", "ch14"), ("ch14", "ch15"),
    ],
}


def get_channel(full_name: str) -> Channel:
    sat, ch = full_name.rsplit("_", 1)
    sat = normalize_sat(sat)
    try:
        return CHANNELS[sat][ch.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown channel {full_name!r}") from exc


def make_full_name(sat: str, ch: str) -> str:
    return f"{normalize_sat(sat)}_{ch.lower()}"


def build_channel_pairs(
    source_sat: str,
    target_sat: str,
    overrides: Optional[Sequence[Sequence[str]]] = None,
) -> List[Tuple[str, str]]:
    """Return full-name channel pairs such as ('fy4a_ch01', 'fy4c_ch01').

    overrides may be used in config JSON for temporary experiments. Each item may be:
    - ['fy4a_ch01', 'fy4c_ch01']
    - ['ch01', 'ch01']  # interpreted under source_sat and target_sat
    """
    src = normalize_sat(source_sat)
    tgt = normalize_sat(target_sat)

    raw_pairs: List[Tuple[str, str]]
    if overrides:
        raw_pairs = []
        for item in overrides:
            if len(item) != 2:
                raise ValueError(f"Invalid channel pair override {item!r}; expected two values")
            left, right = item[0].lower(), item[1].lower()
            if "_" not in left:
                left = make_full_name(src, left)
            if "_" not in right:
                right = make_full_name(tgt, right)
            # Validate now to catch typos before a long run.
            get_channel(left)
            get_channel(right)
            raw_pairs.append((left, right))
        return raw_pairs

    key = (src, tgt)
    reverse = False
    if key not in PAIR_PRESETS:
        key = (tgt, src)
        reverse = True
    if key not in PAIR_PRESETS:
        raise ValueError(f"No channel-pair preset for {src} -> {tgt}")

    left_sat, right_sat = key
    raw_pairs = [(make_full_name(left_sat, a), make_full_name(right_sat, b)) for a, b in PAIR_PRESETS[key]]
    if reverse:
        raw_pairs = [(b, a) for a, b in raw_pairs]
    return raw_pairs


def wavelength_um(full_name: str) -> float:
    return get_channel(full_name).wavelength_um


def channel_title(full_name: str) -> str:
    return get_channel(full_name).title
