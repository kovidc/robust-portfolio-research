"""Validated, hashable configuration for research-foundation runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchConfig:
    """A JSON configuration with deterministic canonical serialization."""

    path: Path
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: Path | str) -> ResearchConfig:
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        cls._validate(payload)
        return cls(path=config_path, payload=payload)

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        required_sections = {
            "schema_version",
            "experiment",
            "data",
            "universe",
            "backtest",
            "turnover",
            "costs",
            "artifacts",
            "limitations",
        }
        missing = required_sections.difference(payload)
        if missing:
            raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

        if payload["schema_version"] != 1:
            raise ValueError("Unsupported research configuration schema version.")
        if payload["data"]["universe_mode"] not in {"POINT_IN_TIME", "SURVIVOR_PANEL"}:
            raise ValueError("data.universe_mode must be POINT_IN_TIME or SURVIVOR_PANEL.")
        if payload["backtest"]["execution_convention"] != "CLOSE_T_AFTER_RETURN":
            raise ValueError("The research foundation supports only CLOSE_T_AFTER_RETURN execution.")
        if int(payload["universe"]["required_history_observations"]) < 1:
            raise ValueError("required_history_observations must be positive.")
        if int(payload["backtest"]["estimation_window_observations"]) < 1:
            raise ValueError("estimation_window_observations must be positive.")
        if float(payload["backtest"]["initial_nav"]) <= 0:
            raise ValueError("initial_nav must be positive.")
        maximum_weight = payload["backtest"]["maximum_weight"]
        if maximum_weight is not None and not 0 < float(maximum_weight) <= 1:
            raise ValueError("maximum_weight must be in (0, 1] or null.")
        if payload["costs"]["model"] not in {"ZERO", "LINEAR"}:
            raise ValueError("costs.model must be ZERO or LINEAR.")
        if float(payload["costs"]["linear_rate_per_dollar_traded"]) < 0:
            raise ValueError("Linear cost rates cannot be negative.")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def section(self, name: str) -> dict[str, Any]:
        return self.payload[name]

    def resolve_repository_path(self, repository_root: Path, key: str) -> Path:
        return (repository_root / self.payload["data"][key]).resolve()
