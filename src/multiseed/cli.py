# src/multiseed/cli.py

import argparse
import logging
from pathlib import Path

import numpy as np

from .config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multiseed",
        description="Run group-level rs-fMRI multi-seed analyses from a config file.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="Path to a config file describing the analysis to run.",
    )
    parser.add_argument(
        "--write-default-config",
        metavar="PATH",
        type=Path,
        help="Write a default/empty config file to PATH and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(levelname)s %(asctime)s] %(message)s",
        level=level,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    if args.write_default_config is not None and args.config is not None:
        parser.error(
            "provide either a config file to run or --write-default-config PATH, not both"
        )

    if args.write_default_config is not None:
        Config.write_default_config(str(args.write_default_config))
        logging.info("Wrote default config to %s", args.write_default_config)
        return 0

    if args.config is None:
        parser.error(
            "a config file is required unless --write-default-config PATH is used"
        )

    # ensure reproducibility of random numbers
    rng = np.random.default_rng(seed=42)

    config = Config.from_file(str(args.config))
    config.run(rng)

    return 0
