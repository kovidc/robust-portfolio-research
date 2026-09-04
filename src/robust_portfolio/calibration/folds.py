"""Mechanical outer-start derivation and strictly historical inner folds."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HistoricalFold:
    decision_date: pd.Timestamp
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    fit_observations: int
    validation_observations: int

    def to_dict(self) -> dict:
        return {
            "decision_date": self.decision_date.isoformat(),
            "fit_start": self.fit_start.isoformat(),
            "fit_end": self.fit_end.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "fit_observations": self.fit_observations,
            "validation_observations": self.validation_observations,
        }


def _fold_before(
    decision_date: pd.Timestamp,
    next_boundary: pd.Timestamp,
    return_dates: pd.DatetimeIndex,
    estimation_window: int,
) -> HistoricalFold | None:
    fit = return_dates[return_dates < decision_date][-estimation_window:]
    validation = return_dates[(return_dates > decision_date) & (return_dates < next_boundary)]
    if len(fit) < estimation_window or len(validation) == 0:
        return None
    return HistoricalFold(
        decision_date=decision_date,
        fit_start=fit[0],
        fit_end=fit[-1],
        validation_start=validation[0],
        validation_end=validation[-1],
        fit_observations=len(fit),
        validation_observations=len(validation),
    )


def derive_outer_schedule(
    return_dates,
    rebalance_dates,
    *,
    estimation_window: int,
    minimum_prior_inner_folds: int,
) -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, tuple[HistoricalFold, ...]]]:
    """Return outer dates and the latest completed folds strictly before each outer date."""
    returns = pd.DatetimeIndex(pd.to_datetime(return_dates)).sort_values()
    schedule = pd.DatetimeIndex(pd.to_datetime(rebalance_dates)).sort_values()
    folds_by_outer: dict[pd.Timestamp, tuple[HistoricalFold, ...]] = {}
    valid_outer = []
    for outer_position, outer_date in enumerate(schedule):
        candidates = []
        for position in range(outer_position):
            fold = _fold_before(
                schedule[position], schedule[position + 1], returns, estimation_window
            )
            if fold is not None and fold.validation_end < outer_date:
                candidates.append(fold)
        if len(candidates) >= minimum_prior_inner_folds:
            valid_outer.append(outer_date)
            folds_by_outer[outer_date] = tuple(candidates[-minimum_prior_inner_folds:])
    if not valid_outer:
        raise ValueError("No rebalance date has the required estimation history and inner folds.")
    return pd.DatetimeIndex(valid_outer), folds_by_outer
