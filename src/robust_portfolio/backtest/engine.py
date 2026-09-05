"""Self-financing backtest engine with strict information boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from robust_portfolio.config import ResearchConfig
from robust_portfolio.data.providers import FrozenCsvReturnProvider
from robust_portfolio.data.schemas import ReturnPanel, UniverseSnapshot

from .accounting import apply_close_to_close_return
from .clock import CloseAfterReturnExecutionConvention, ExecutionConvention
from .costs import CostModel, ZeroCostModel
from .execution import ExecutionResult, execute_target
from .state import ACCOUNTING_TOLERANCE, PortfolioState


@dataclass(frozen=True)
class DecisionContext:
    """The only information object supplied to a target-producing strategy."""

    execution_date: pd.Timestamp
    returns: ReturnPanel
    universe: UniverseSnapshot
    decision_state: PortfolioState
    execution_convention: str


TargetPolicy = Callable[[DecisionContext], pd.Series]


@dataclass
class BacktestResult:
    strategy_name: str
    result_label: str
    execution_convention: dict[str, str]
    daily_states: list[PortfolioState]
    daily_ledger: pd.DataFrame
    executions: list[ExecutionResult]
    universe_snapshots: list[UniverseSnapshot]
    artifact_paths: dict[str, str] = field(default_factory=dict)
    manifest: dict | None = None

    @property
    def initial_execution(self) -> ExecutionResult:
        initial = [execution for execution in self.executions if execution.initial_formation]
        if len(initial) != 1:
            raise ValueError(f"Expected one initial execution, found {len(initial)}.")
        return initial[0]

    @property
    def recurring_executions(self) -> list[ExecutionResult]:
        return [execution for execution in self.executions if not execution.initial_formation]


class BacktestEngine:
    """Apply market returns, then execute targets at the same close."""

    def __init__(
        self,
        *,
        returns: FrozenCsvReturnProvider,
        universe_builder,
        config: ResearchConfig,
        cost_model: CostModel | None = None,
        execution_convention: ExecutionConvention | None = None,
    ):
        self.returns = returns
        self.universe_builder = universe_builder
        self.config = config
        self.cost_model = cost_model or ZeroCostModel()
        self.execution_convention = execution_convention or CloseAfterReturnExecutionConvention()

        configured = config.section("backtest")["execution_convention"]
        if configured != self.execution_convention.name:
            raise ValueError(
                f"Config execution convention {configured} does not match "
                f"engine convention {self.execution_convention.name}."
            )
        configured_cost = config.section("costs")["model"]
        if configured_cost != self.cost_model.name:
            raise ValueError(
                f"Config cost model {configured_cost} does not match "
                f"engine cost model {self.cost_model.name}."
            )
        if configured_cost == "LINEAR":
            description = self.cost_model.describe()
            configured_rate = float(
                config.section("costs")["linear_rate_per_dollar_traded"]
            )
            actual_rate = description.get("scalar_cost_per_dollar_traded")
            if actual_rate is None or not np.isclose(
                float(actual_rate), configured_rate, atol=0.0, rtol=0.0
            ):
                raise ValueError(
                    "The research foundation LINEAR engine cost rate must exactly match the scalar config rate."
                )

    def run(
        self,
        *,
        strategy_name: str,
        target_policy: TargetPolicy,
        rebalance_dates,
        artifact_dir: Path | str,
        input_paths: Mapping[str, Path | str],
        repository_root: Path | str,
    ) -> BacktestResult:
        """Run and persist one configured experiment, including its manifest."""
        schedule = pd.DatetimeIndex(pd.to_datetime(list(rebalance_dates))).sort_values()
        if schedule.empty or not schedule.is_unique:
            raise ValueError("Rebalance dates must be nonempty, sorted, and unique.")
        missing_dates = schedule.difference(self.returns.dates)
        if len(missing_dates):
            raise ValueError(f"Rebalance dates lack return rows: {missing_dates.tolist()}")

        first_rebalance = schedule[0]
        simulation_dates = self.returns.dates[self.returns.dates >= first_rebalance]
        if simulation_dates.empty:
            raise ValueError("There are no returns on or after the first rebalance.")

        initial_timestamp = first_rebalance - pd.Timedelta(nanoseconds=1)
        initial_nav = float(self.config.section("backtest")["initial_nav"])
        state = PortfolioState.all_cash(initial_timestamp, initial_nav, self.returns.assets)
        schedule_set = set(schedule)
        estimation_window = int(
            self.config.section("backtest")["estimation_window_observations"]
        )
        cash_return = float(self.config.section("backtest")["cash_daily_return"])
        maximum_weight = self.config.section("backtest")["maximum_weight"]
        maximum_weight = None if maximum_weight is None else float(maximum_weight)

        daily_states = []
        daily_records = []
        executions = []
        snapshots = []

        for return_date in simulation_dates:
            start_nav = state.nav
            target = None
            snapshot = None

            # Construct the target before row t is applied. The policy sees
            # only returns strictly before t and the state at close t-1.
            if return_date in schedule_set:
                information_boundary = self.execution_convention.information_as_of(return_date)
                full_panel = self.returns.panel(as_of=information_boundary)
                snapshot = self.universe_builder.snapshot(full_panel)
                if not snapshot.eligible_assets:
                    raise ValueError(f"No assets are eligible as of {return_date}.")
                context = DecisionContext(
                    execution_date=pd.Timestamp(return_date),
                    returns=full_panel.trailing(estimation_window),
                    universe=snapshot,
                    decision_state=state,
                    execution_convention=self.execution_convention.name,
                )
                target = target_policy(context).astype(float)
                noneligible = [
                    asset
                    for asset, weight in target.items()
                    if abs(float(weight)) > ACCOUNTING_TOLERANCE
                    and asset not in snapshot.eligible_assets
                ]
                if noneligible:
                    raise ValueError(
                        f"Target contains assets not eligible at {return_date}: {noneligible}"
                    )

            # Existing holdings then earn the close t-1 to close t return.
            old_state_after_return = apply_close_to_close_return(
                state,
                self.returns.return_ending_at(return_date),
                return_date=return_date,
                cash_return=cash_return,
            )
            transaction_cost = 0.0
            executed = False

            if target is not None:
                execution = execute_target(
                    old_state_after_return,
                    target,
                    execution_date=return_date,
                    cost_model=self.cost_model,
                    maximum_weight=maximum_weight,
                    initial_formation=not executions,
                )
                state = execution.post_trade_state
                transaction_cost = execution.transaction_cost
                executed = True
                executions.append(execution)
                snapshots.append(snapshot)
            else:
                state = old_state_after_return

            if start_nav <= ACCOUNTING_TOLERANCE:
                daily_return = np.nan
            else:
                daily_return = state.nav / start_nav - 1.0
            daily_states.append(state)
            daily_records.append(
                {
                    "date": pd.Timestamp(return_date),
                    "start_nav": start_nav,
                    "pre_trade_nav": old_state_after_return.nav,
                    "transaction_cost": transaction_cost,
                    "end_nav": state.nav,
                    "daily_net_return": daily_return,
                    "executed_at_close": executed,
                    "cash": state.cash,
                    "cash_weight": state.cash_weight,
                }
            )

        result = BacktestResult(
            strategy_name=strategy_name,
            result_label=self.config.section("experiment")["result_label"],
            execution_convention=self.execution_convention.describe(),
            daily_states=daily_states,
            daily_ledger=pd.DataFrame(daily_records).set_index("date"),
            executions=executions,
            universe_snapshots=snapshots,
        )

        from robust_portfolio.reporting.artifacts import write_backtest_artifacts
        from robust_portfolio.reporting.manifests import (
            build_run_manifest,
            write_manifest,
        )

        artifact_paths = write_backtest_artifacts(
            result,
            artifact_dir=artifact_dir,
            repository_root=Path(repository_root),
        )
        manifest_path = Path(artifact_dir).resolve() / self.config.section("artifacts")[
            "manifest_filename"
        ]
        artifact_paths["run_manifest"] = str(manifest_path)
        manifest = build_run_manifest(
            repository_root=Path(repository_root),
            config=self.config,
            input_paths=input_paths,
            artifact_paths=artifact_paths,
            execution_convention=self.execution_convention.describe(),
            universe_mode=self.universe_builder.mode.value,
            survivor_conditioned=bool(result.universe_snapshots[0].survivor_conditioned),
            survivorship_bias_free=bool(result.universe_snapshots[0].survivorship_bias_free),
            strategy_name=strategy_name,
            cost_model=self.cost_model.describe(),
            result_label=result.result_label,
        )
        write_manifest(manifest_path, manifest)
        result.artifact_paths = artifact_paths
        result.manifest = manifest
        return result
