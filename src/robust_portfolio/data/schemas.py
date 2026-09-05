"""Immutable schemas for as-of return panels and universe snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd


def _timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


@dataclass(frozen=True)
class ReturnPanel:
    """Return observations strictly earlier than an exclusive as-of boundary."""

    as_of: pd.Timestamp
    _values: pd.DataFrame
    source_sha256: str

    def __post_init__(self):
        as_of = _timestamp(self.as_of)
        values = self._values.copy(deep=True)
        if not isinstance(values.index, pd.DatetimeIndex):
            raise TypeError("ReturnPanel index must be a DatetimeIndex.")
        values.index = pd.DatetimeIndex([_timestamp(value) for value in values.index])
        if not values.index.is_monotonic_increasing or not values.index.is_unique:
            raise ValueError("ReturnPanel dates must be sorted and unique.")
        if len(values) and not bool((values.index < as_of).all()):
            raise ValueError("ReturnPanel may contain only observations strictly before as_of.")
        object.__setattr__(self, "as_of", as_of)
        object.__setattr__(self, "_values", values)

    @property
    def values(self) -> pd.DataFrame:
        """Return a defensive copy; the bounded information set cannot be mutated."""
        return self._values.copy(deep=True)

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self._values.columns)

    @property
    def last_observation(self) -> pd.Timestamp | None:
        return None if self._values.empty else self._values.index[-1]

    def trailing(self, observations: int) -> ReturnPanel:
        if observations < 1:
            raise ValueError("observations must be positive.")
        return ReturnPanel(
            as_of=self.as_of,
            _values=self._values.tail(observations),
            source_sha256=self.source_sha256,
        )


@dataclass(frozen=True)
class UniverseSnapshot:
    """Eligibility decision made using one bounded ReturnPanel."""

    as_of: pd.Timestamp
    mode: str
    eligible_assets: tuple[str, ...]
    exclusion_reasons: Mapping[str, str]
    history_observations: Mapping[str, int]
    survivor_conditioned: bool
    survivorship_bias_free: bool
    limitation: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "as_of", _timestamp(self.as_of))
        object.__setattr__(self, "eligible_assets", tuple(self.eligible_assets))
        object.__setattr__(
            self, "exclusion_reasons", MappingProxyType(dict(self.exclusion_reasons))
        )
        object.__setattr__(
            self, "history_observations", MappingProxyType(dict(self.history_observations))
        )
        if self.survivorship_bias_free and self.survivor_conditioned:
            raise ValueError("A survivor-conditioned snapshot cannot be survivorship-bias-free.")

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "mode": self.mode,
            "eligible_assets": list(self.eligible_assets),
            "exclusion_reasons": dict(self.exclusion_reasons),
            "history_observations": dict(self.history_observations),
            "survivor_conditioned": self.survivor_conditioned,
            "survivorship_bias_free": self.survivorship_bias_free,
            "limitation": self.limitation,
        }
