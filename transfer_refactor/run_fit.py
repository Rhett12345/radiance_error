# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

from fy4_transfer.config import default_config, load_config, save_config_template
from fy4_transfer.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit FY-4 AGRI channel transfer models")
    parser.add_argument("--pair", default="ac", help="Satellite pair, e.g. ac, ca, bc, cb, ab, ba")
    parser.add_argument("--config", default=None, help="Optional JSON config path. Overrides --pair defaults")
    parser.add_argument("--write-template", default=None, help="Write a JSON config template and exit")
    parser.add_argument("--no-plots", action="store_true", help="Run fitting without saving plot PNGs")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else default_config(args.pair)
    if args.no_plots:
        cfg.make_plots = False

    if args.write_template:
        save_config_template(cfg, args.write_template)
        print(f"Wrote template: {args.write_template}")
        return

    run(cfg)


if __name__ == "__main__":
    main()
