"""Portfolio state with explicit risky holdings, cash, and NAV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ACCOUNTING_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PortfolioState:
    """Self-financing state measured at one market close."""

    timestamp: pd.Timestamp
    nav: float
    holdings: pd.Series
    cash: float

    def __post_init__(self):
        timestamp = pd.Timestamp(self.timestamp)
        holdings = self.holdings.astype(float).copy()
        if holdings.index.duplicated().any():
            raise ValueError("Portfolio holdings must have unique asset identifiers.")
        if not np.isfinite(holdings.to_numpy()).all():
            raise ValueError("Portfolio holdings must be finite.")
        if float(holdings.min()) < -ACCOUNTING_TOLERANCE:
            raise ValueError("Research foundation portfolio holdings cannot be short.")
        if not np.isfinite(self.nav) or self.nav < -ACCOUNTING_TOLERANCE:
            raise ValueError("NAV must be finite and nonnegative.")
        if not np.isfinite(self.cash) or self.cash < -ACCOUNTING_TOLERANCE:
            raise ValueError("Cash must be finite and nonnegative.")
        accounted_nav = float(holdings.sum() + self.cash)
        if not np.isclose(accounted_nav, self.nav, atol=ACCOUNTING_TOLERANCE, rtol=0.0):
            raise ValueError(
                f"Holdings plus cash ({accounted_nav}) do not equal NAV ({self.nav})."
            )
        holdings = holdings.clip(lower=0.0)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "nav", float(self.nav))
        object.__setattr__(self, "holdings", holdings)
        object.__setattr__(self, "cash", max(float(self.cash), 0.0))

    @classmethod
    def all_cash(cls, timestamp, nav: float, assets=()) -> "PortfolioState":
        return cls(
            timestamp=pd.Timestamp(timestamp),
            nav=float(nav),
            holdings=pd.Series(0.0, index=list(assets), dtype=float),
            cash=float(nav),
        )

    @property
    def weights(self) -> pd.Series:
        if self.nav <= ACCOUNTING_TOLERANCE:
            return pd.Series(0.0, index=self.holdings.index, dtype=float)
        return self.holdings / self.nav

    @property
    def cash_weight(self) -> float:
        return 0.0 if self.nav <= ACCOUNTING_TOLERANCE else self.cash / self.nav

    def reindex(self, assets) -> "PortfolioState":
        return PortfolioState(
            timestamp=self.timestamp,
            nav=self.nav,
            holdings=self.holdings.reindex(list(assets), fill_value=0.0),
            cash=self.cash,
        )
