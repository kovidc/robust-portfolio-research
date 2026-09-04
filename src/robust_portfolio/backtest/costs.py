"""Transaction-cost interfaces operating on actual risky-asset dollar trades."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Mapping

import numpy as np
import pandas as pd


class CostModel(ABC):
    name: str

    @abstractmethod
    def cost(self, dollar_trades: pd.Series) -> float:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError


@dataclass(frozen=True)
class ZeroCostModel(CostModel):
    name: str = "ZERO"

    def cost(self, dollar_trades: pd.Series) -> float:
        return 0.0

    def describe(self) -> dict:
        return {"model": self.name, "cost_per_dollar_traded": 0.0}


class LinearCostModel(CostModel):
    """Cost equals sum_i rate_i times absolute risky-asset dollar trade_i."""

    name = "LINEAR"

    def __init__(self, rates: Real | Mapping[str, float] | pd.Series):
        if isinstance(rates, Real):
            rate = float(rates)
            self._scalar_rate = rate
            self._rates = None
            values = np.array([rate])
        else:
            self._scalar_rate = None
            self._rates = pd.Series(dict(rates), dtype=float)
            values = self._rates.to_numpy()
        if not np.isfinite(values).all() or bool((values < 0).any()):
            raise ValueError("Linear cost rates must be finite and nonnegative.")
        if bool((values >= 1.0).any()):
            raise ValueError("The research foundation requires each linear cost rate to be below 100%.")

    def rates_for(self, assets) -> pd.Series:
        assets = pd.Index(assets)
        if self._scalar_rate is not None:
            return pd.Series(self._scalar_rate, index=assets, dtype=float)
        missing = assets.difference(self._rates.index)
        if len(missing):
            raise KeyError(f"No linear cost rate supplied for assets: {missing.tolist()}")
        return self._rates.reindex(assets)

    def cost(self, dollar_trades: pd.Series) -> float:
        rates = self.rates_for(dollar_trades.index)
        return float((rates * dollar_trades.abs()).sum())

    def describe(self) -> dict:
        if self._scalar_rate is not None:
            return {
                "model": self.name,
                "scalar_cost_per_dollar_traded": self._scalar_rate,
            }
        return {
            "model": self.name,
            "asset_cost_per_dollar_traded": self._rates.sort_index().to_dict(),
        }


def cost_model_from_config(cost_config: Mapping) -> CostModel:
    """Build the supported cost model without changing engine code."""
    model = cost_config["model"]
    rate = float(cost_config.get("linear_rate_per_dollar_traded", 0.0))
    if model == "ZERO":
        if rate != 0.0:
            raise ValueError("A ZERO cost configuration must have a zero linear rate.")
        return ZeroCostModel()
    if model == "LINEAR":
        return LinearCostModel(rate)
    raise ValueError(f"Unsupported cost model: {model}")
