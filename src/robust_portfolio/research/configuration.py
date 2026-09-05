"""Core experiment configuration with deterministic validation and hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CoreExperimentConfig:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: Path | str) -> CoreExperimentConfig:
        resolved = Path(path).resolve()
        with resolved.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        cls._validate(payload)
        return cls(path=resolved, payload=payload)

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        required = {
            "schema_version", "experiment", "data", "clock", "walkforward",
            "annualization", "universe", "constraints", "means", "covariances",
            "uncertainty", "risk_matching", "costs", "optimization", "outputs",
            "limitations",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Core experiment config is missing sections: {sorted(missing)}")
        if payload["schema_version"] != 2:
            raise ValueError("Core experiment requires schema_version=2.")
        if payload["data"]["universe_mode"] != "SURVIVOR_PANEL":
            raise ValueError("The public-data core experiment config must declare SURVIVOR_PANEL.")
        if payload["data"]["survivorship_bias_free"]:
            raise ValueError("The stored ETF panel is survivor-conditioned.")
        if payload["clock"]["execution_convention"] != "CLOSE_T_AFTER_RETURN":
            raise ValueError("The core experiment must retain the close-after-return convention.")
        if int(payload["walkforward"]["minimum_prior_inner_folds"]) < 4:
            raise ValueError("The core experiment requires at least four prior inner folds.")
        cap = float(payload["constraints"]["maximum_weight"])
        if not 0.0 < cap <= 1.0:
            raise ValueError("maximum_weight must lie in (0, 1].")
        risks = [float(value) for value in payload["risk_matching"]["target_annual_volatility"]]
        if not risks or risks != sorted(set(risks)) or risks[0] <= 0.0:
            raise ValueError("Risk targets must be unique, positive, and increasing.")
        bps = [float(value) for value in payload["costs"]["basis_points_per_dollar_traded"]]
        if not bps or any(value < 0.0 for value in bps):
            raise ValueError("Cost scenarios must be nonnegative.")
        if int(payload["uncertainty"]["bootstrap_replications"]) < 2:
            raise ValueError("At least two bootstrap replications are required.")
        coverage = float(payload["uncertainty"]["coverage_probability"])
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage_probability must lie strictly between zero and one.")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def section(self, name: str) -> dict[str, Any]:
        return self.payload[name]

    def repository_path(self, repository_root: Path, key: str) -> Path:
        return (repository_root / self.payload["data"][key]).resolve()
