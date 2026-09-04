"""Dependence-aware inference and model-selection diagnostics."""

from .model_selection import deflated_sharpe_probability
from .stationary_bootstrap import (
    bootstrap_headline_statistics,
    stationary_bootstrap_indices,
)

__all__ = [
    "bootstrap_headline_statistics",
    "deflated_sharpe_probability",
    "stationary_bootstrap_indices",
]
