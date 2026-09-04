"""Final analysis configuration validation and hashing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FinalAnalysisConfig:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: Path | str) -> "FinalAnalysisConfig":
        resolved = Path(path).resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        cls._validate(payload)
        return cls(resolved, payload)

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        required = {
            "schema_version", "experiment", "inputs", "comparison",
            "risk_attainment", "direct_robustness", "clone_experiment",
            "clustering", "regimes", "inference", "sensitivity", "outputs",
            "limitations",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Final analysis config is missing sections: {sorted(missing)}")
        if payload["schema_version"] != 3:
            raise ValueError("Final analysis requires schema_version=3.")
        if payload["inference"]["risk_free_treatment"] != "ZERO_RF_PROVISIONAL":
            raise ValueError("The final analysis has no validated risk-free input.")
        thresholds = [float(x) for x in payload["clustering"]["correlation_thresholds"]]
        if thresholds != [0.80, 0.90, 0.95, 0.975]:
            raise ValueError("Configured clustering thresholds changed.")
        dates = payload["direct_robustness"]["selected_outer_dates"]
        if dates != payload["clone_experiment"]["selected_outer_dates"]:
            raise ValueError("Direct and clone experiments must share selected dates.")
        if int(payload["inference"]["replications"]) != 2000:
            raise ValueError("Final inference requires 2,000 replications.")

    @property
    def sha256(self) -> str:
        value = json.dumps(
            self.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def section(self, name: str) -> dict[str, Any]:
        return self.payload[name]

    def repository_path(self, root: Path, key: str) -> Path:
        return (root / self.payload["inputs"][key]).resolve()
