"""Self-financing accounting, execution, and backtest engine."""

from .clock import CloseAfterReturnExecutionConvention
from .costs import LinearCostModel, ZeroCostModel, cost_model_from_config
from .engine import BacktestEngine, BacktestResult, DecisionContext
from .execution import ExecutionResult, execute_target
from .state import PortfolioState

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CloseAfterReturnExecutionConvention",
    "DecisionContext",
    "ExecutionResult",
    "LinearCostModel",
    "PortfolioState",
    "ZeroCostModel",
    "cost_model_from_config",
    "execute_target",
]
