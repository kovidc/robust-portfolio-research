"""Execution-convention interface and the supported close-at-t policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class ExecutionConvention(ABC):
    """Defines information and return boundaries around an execution event."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def information_as_of(self, execution_date) -> pd.Timestamp:
        """Return the exclusive data boundary supplied to the strategy."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict[str, str]:
        raise NotImplementedError


@dataclass(frozen=True)
class CloseAfterReturnExecutionConvention(ExecutionConvention):
    """Old holdings earn row t; target executes at close t; new holdings start after t."""

    @property
    def name(self) -> str:
        return "CLOSE_T_AFTER_RETURN"

    def information_as_of(self, execution_date) -> pd.Timestamp:
        # ReturnPanel treats this timestamp as an exclusive boundary.
        return pd.Timestamp(execution_date)

    def describe(self) -> dict[str, str]:
        return {
            "return_index_semantics": "row t is the close t-1 to close t return",
            "forecast_information": "return observations strictly before t",
            "pre_execution_return": "old holdings earn row t",
            "execution": "target executes at close t after row t is earned",
            "first_new_weight_return": "first return row strictly after t",
        }
