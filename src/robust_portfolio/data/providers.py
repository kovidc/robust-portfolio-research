"""Frozen return providers with exclusive as-of access."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .schemas import ReturnPanel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenCsvReturnProvider:
    """Read-only provider for a content-addressed return CSV."""

    def __init__(self, path: Path | str):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        returns = pd.read_csv(self.path, index_col=0, parse_dates=True)
        if returns.empty:
            raise ValueError("The return CSV is empty.")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("Return CSV index must contain dates.")
        returns = returns.astype(float).sort_index()
        if not returns.index.is_unique:
            raise ValueError("Return CSV dates must be unique.")
        self._returns = returns
        self.sha256 = sha256_file(self.path)

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self._returns.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._returns.index.copy()

    def panel(
        self,
        *,
        as_of,
        assets: Iterable[str] | None = None,
        trailing_observations: int | None = None,
    ) -> ReturnPanel:
        """Return only rows strictly before as_of."""
        boundary = pd.Timestamp(as_of)
        bounded = self._returns.loc[self._returns.index < boundary]
        if assets is not None:
            requested = list(assets)
            missing = sorted(set(requested).difference(bounded.columns))
            if missing:
                raise KeyError(f"Unknown assets requested: {missing}")
            bounded = bounded.loc[:, requested]
        if trailing_observations is not None:
            if trailing_observations < 1:
                raise ValueError("trailing_observations must be positive.")
            bounded = bounded.tail(trailing_observations)
        return ReturnPanel(
            as_of=boundary,
            _values=bounded,
            source_sha256=self.sha256,
        )

    def return_ending_at(self, date) -> pd.Series:
        """Market return used by the engine, never exposed to the strategy context."""
        timestamp = pd.Timestamp(date)
        if timestamp not in self._returns.index:
            raise KeyError(f"No return row ends at {timestamp}.")
        return self._returns.loc[timestamp].copy()
