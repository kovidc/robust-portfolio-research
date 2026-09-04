"""Point-in-time-capable and explicitly survivor-conditioned universe builders."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import pandas as pd

from .schemas import ReturnPanel, UniverseSnapshot


class UniverseMode(str, Enum):
    POINT_IN_TIME = "POINT_IN_TIME"
    SURVIVOR_PANEL = "SURVIVOR_PANEL"


class PointInTimeDataUnavailable(RuntimeError):
    """Raised when a true historical universe is requested without its metadata."""


@dataclass(frozen=True)
class UniverseRules:
    required_history_observations: int
    require_complete_required_window: bool = True

    def __post_init__(self):
        if self.required_history_observations < 1:
            raise ValueError("required_history_observations must be positive.")


def _history_decision(
    series: pd.Series,
    rules: UniverseRules,
) -> tuple[bool, str | None, int]:
    observed = int(series.notna().sum())
    if observed < rules.required_history_observations:
        return False, f"INSUFFICIENT_HISTORY:{observed}/{rules.required_history_observations}", observed
    if rules.require_complete_required_window:
        required_window = series.tail(rules.required_history_observations)
        missing = int(required_window.isna().sum())
        if missing:
            return False, f"MISSING_IN_REQUIRED_WINDOW:{missing}", observed
    return True, None, observed


class SurvivorPanelUniverseBuilder:
    """As-of eligibility within a predeclared, survivor-selected public panel."""

    mode = UniverseMode.SURVIVOR_PANEL
    limitation = (
        "Declared assets come from the current public-data survivor panel; inactive funds "
        "and historical membership are unavailable. This is not survivorship-bias-free."
    )

    def __init__(self, declared_assets: Iterable[str], rules: UniverseRules):
        assets = tuple(dict.fromkeys(str(asset) for asset in declared_assets))
        if not assets:
            raise ValueError("The declared survivor panel cannot be empty.")
        self.declared_assets = assets
        self.rules = rules

    def snapshot(self, panel: ReturnPanel) -> UniverseSnapshot:
        values = panel.values
        eligible = []
        exclusions = {}
        observations = {}
        for asset in self.declared_assets:
            if asset not in values.columns:
                exclusions[asset] = "ABSENT_FROM_FROZEN_RETURN_PANEL"
                observations[asset] = 0
                continue
            is_eligible, reason, observed = _history_decision(values[asset], self.rules)
            observations[asset] = observed
            if is_eligible:
                eligible.append(asset)
            else:
                exclusions[asset] = reason
        return UniverseSnapshot(
            as_of=panel.as_of,
            mode=self.mode.value,
            eligible_assets=tuple(eligible),
            exclusion_reasons=exclusions,
            history_observations=observations,
            survivor_conditioned=True,
            survivorship_bias_free=False,
            limitation=self.limitation,
        )


class PointInTimeUniverseBuilder:
    """Universe builder for dated metadata, usable now with synthetic fixtures."""

    mode = UniverseMode.POINT_IN_TIME

    def __init__(
        self,
        metadata: pd.DataFrame | None,
        rules: UniverseRules,
        *,
        survivorship_bias_free_claim_supported: bool = False,
    ):
        if metadata is None:
            raise PointInTimeDataUnavailable(
                "POINT_IN_TIME mode requires historical listing/inactivation metadata and "
                "inactive-fund return histories; those data are not present in this repository."
            )
        required = {"asset", "listing_date"}
        missing = required.difference(metadata.columns)
        if missing:
            raise ValueError(f"Point-in-time metadata is missing columns: {sorted(missing)}")
        dated = metadata.copy()
        if dated["asset"].duplicated().any():
            raise ValueError("Point-in-time metadata must have one row per stable asset identifier.")
        dated["listing_date"] = pd.to_datetime(dated["listing_date"])
        if "inactive_date" not in dated:
            dated["inactive_date"] = pd.NaT
        else:
            dated["inactive_date"] = pd.to_datetime(dated["inactive_date"])
        self.metadata = dated.set_index("asset").sort_index()
        self.rules = rules
        self.survivorship_bias_free_claim_supported = bool(
            survivorship_bias_free_claim_supported
        )

    def snapshot(self, panel: ReturnPanel) -> UniverseSnapshot:
        values = panel.values
        eligible = []
        exclusions = {}
        observations = {}
        for asset, row in self.metadata.iterrows():
            asset = str(asset)
            if pd.Timestamp(row["listing_date"]) >= panel.as_of:
                exclusions[asset] = "NOT_LISTED_AS_OF"
                observations[asset] = 0
                continue
            inactive_date = row["inactive_date"]
            if pd.notna(inactive_date) and pd.Timestamp(inactive_date) <= panel.as_of:
                exclusions[asset] = "INACTIVE_AS_OF"
                observations[asset] = int(values[asset].notna().sum()) if asset in values else 0
                continue
            if asset not in values.columns:
                exclusions[asset] = "NO_RETURN_HISTORY_AS_OF"
                observations[asset] = 0
                continue
            is_eligible, reason, observed = _history_decision(values[asset], self.rules)
            observations[asset] = observed
            if is_eligible:
                eligible.append(asset)
            else:
                exclusions[asset] = reason
        return UniverseSnapshot(
            as_of=panel.as_of,
            mode=self.mode.value,
            eligible_assets=tuple(eligible),
            exclusion_reasons=exclusions,
            history_observations=observations,
            survivor_conditioned=False,
            survivorship_bias_free=self.survivorship_bias_free_claim_supported,
            limitation=(
                None
                if self.survivorship_bias_free_claim_supported
                else "Dated metadata were supplied, but complete inactive-fund coverage has not "
                "been certified; no survivorship-bias-free claim is made."
            ),
        )
