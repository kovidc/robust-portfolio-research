"""Reproducible run manifests with code, config, data, and environment provenance."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Mapping

from robust_portfolio.config import ResearchConfig
from robust_portfolio.data.providers import sha256_file


def git_state(repository_root: Path) -> dict:
    def run(*args):
        completed = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": None}
    return {"commit": commit, "dirty": bool(status), "status": status.splitlines()}


def dependency_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "numpy", "pandas", "scipy", "cvxpy", "matplotlib", "yfinance",
        "clarabel", "osqp", "scs",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def build_run_manifest(
    *,
    repository_root: Path,
    config: ResearchConfig,
    input_paths: Mapping[str, Path | str],
    artifact_paths: Mapping[str, str],
    execution_convention: dict[str, str],
    universe_mode: str,
    survivor_conditioned: bool,
    survivorship_bias_free: bool,
    strategy_name: str,
    cost_model: dict,
    result_label: str,
) -> dict:
    input_hashes = {
        name: {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(Path(path).resolve()),
        }
        for name, path in sorted(input_paths.items())
    }
    return {
        "schema_version": 1,
        "result_label": result_label,
        "strategy": strategy_name,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(repository_root),
        "configuration": {
            "path": str(config.path),
            "canonical_sha256": config.sha256,
        },
        "inputs": input_hashes,
        "environment": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dependency_versions": dependency_versions(),
        },
        "execution_convention": execution_convention,
        "universe": {
            "mode": universe_mode,
            "survivor_conditioned": survivor_conditioned,
            "survivorship_bias_free": survivorship_bias_free,
        },
        "cost_model": cost_model,
        "artifact_locations": dict(sorted(artifact_paths.items())),
        "limitations": list(config.payload["limitations"]),
    }


def write_manifest(path: Path | str, manifest: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
