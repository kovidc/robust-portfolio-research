#!/usr/bin/env python3
"""Run the final research analysis from its JSON configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from robust_portfolio.research.final_analysis import run_final_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "final_analysis.json")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    result = run_final_analysis(arguments.config, repository_root=REPOSITORY_ROOT, output_dir=arguments.output)
    print(f"Final analysis artifacts: {result['output_directory']}")
    print(f"Explicit failed solves retained: {len(result['failures'])}")
    print("Final research tables: 4")


if __name__ == "__main__":
    main()
