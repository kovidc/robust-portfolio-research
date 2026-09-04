#!/usr/bin/env python3
"""Run the core quantitative experiment from its JSON configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from robust_portfolio.research.core_experiment import run_core_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "core_experiment.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    result = run_core_experiment(
        arguments.config,
        repository_root=REPOSITORY_ROOT,
        output_dir=arguments.output,
    )
    print(f"Core experiment artifacts: {result['output_directory']}")
    print(f"Outer decisions: {len(result['outer_dates'])}")
    print(f"Completed strategy-cost rows: {len(result['metrics'])}")
    print(f"Explicitly failed variants: {len(result['failures'])}")


if __name__ == "__main__":
    main()
