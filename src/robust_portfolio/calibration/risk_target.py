"""Deterministic ex-ante risk-attainment calibration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from robust_portfolio.optimizers.problems import (
    OptimizationResult,
    global_minimum_variance,
    solve_risk_aversion,
)


@dataclass(frozen=True)
class RiskAttainmentResult:
    status: str
    requested_volatility: float
    minimum_feasible_volatility: float
    zero_risk_aversion_volatility: float
    solution: OptimizationResult | None
    iterations: int
    lower_risk_aversion: float | None
    upper_risk_aversion: float | None
    reason: str | None = None


def calibrate_risk_aversion(
    mean: pd.Series,
    decision_covariance: pd.DataFrame,
    *,
    target_volatility: float,
    maximum_weight: float,
    solver_order: list[str],
    common_base_covariance: pd.DataFrame | None = None,
    standard_errors: pd.Series | None = None,
    box_rho: float = 0.0,
    mean_error_covariance: pd.DataFrame | None = None,
    ellipsoid_rho: float = 0.0,
    volatility_tolerance: float = 2e-4,
    initial_upper_risk_aversion: float = 1.0,
    maximum_risk_aversion: float = 1e8,
    maximum_bisection_iterations: int = 40,
    feasibility_tolerance: float = 5e-7,
) -> RiskAttainmentResult:
    """Tune nonnegative gamma using current forecasts only.

    The attainable interval is bounded by constrained GMV risk and the risk of
    the optimizer's deterministic zero-gamma solution. A request outside that
    interval is data, not an invitation to use negative risk aversion.
    """
    if target_volatility <= 0.0 or volatility_tolerance <= 0.0:
        raise ValueError("Target and tolerance must be positive.")
    if initial_upper_risk_aversion <= 0.0:
        raise ValueError("initial_upper_risk_aversion must be positive.")
    if maximum_risk_aversion < initial_upper_risk_aversion:
        raise ValueError("maximum_risk_aversion is below the initial bracket.")
    if maximum_bisection_iterations < 1:
        raise ValueError("maximum_bisection_iterations must be positive.")

    common = decision_covariance if common_base_covariance is None else common_base_covariance
    robust_args = {
        "common_base_covariance": common,
        "standard_errors": standard_errors,
        "box_rho": box_rho,
        "mean_error_covariance": mean_error_covariance,
        "ellipsoid_rho": ellipsoid_rho,
        "feasibility_tolerance": feasibility_tolerance,
    }
    minimum = global_minimum_variance(
        decision_covariance,
        maximum_weight=maximum_weight,
        solver_order=solver_order,
        feasibility_tolerance=feasibility_tolerance,
    )
    zero = solve_risk_aversion(
        mean,
        decision_covariance,
        risk_aversion=0.0,
        maximum_weight=maximum_weight,
        solver_order=solver_order,
        **robust_args,
    )
    minimum_vol = minimum.predicted_volatility
    zero_vol = zero.predicted_volatility
    if target_volatility < minimum_vol - volatility_tolerance:
        return RiskAttainmentResult(
            "TARGET_NOT_ATTAINABLE", target_volatility, minimum_vol, zero_vol,
            None, 0, None, None, "requested target is below constrained GMV risk",
        )
    if target_volatility > zero_vol + volatility_tolerance:
        return RiskAttainmentResult(
            "TARGET_NOT_ATTAINABLE", target_volatility, minimum_vol, zero_vol,
            None, 0, None, None,
            "requested target exceeds the zero-risk-aversion portfolio risk",
        )
    if abs(zero_vol - target_volatility) <= volatility_tolerance:
        return RiskAttainmentResult(
            "ATTAINED", target_volatility, minimum_vol, zero_vol,
            zero, 0, 0.0, 0.0, None,
        )

    low_gamma = 0.0
    low_solution = zero
    high_gamma = float(initial_upper_risk_aversion)
    high_solution = solve_risk_aversion(
        mean,
        decision_covariance,
        risk_aversion=high_gamma,
        maximum_weight=maximum_weight,
        solver_order=solver_order,
        **robust_args,
    )
    while high_solution.predicted_volatility > target_volatility + volatility_tolerance:
        low_gamma, low_solution = high_gamma, high_solution
        high_gamma *= 2.0
        if high_gamma > maximum_risk_aversion:
            return RiskAttainmentResult(
                "TARGET_NOT_ATTAINABLE", target_volatility, minimum_vol, zero_vol,
                None, 0, low_gamma, maximum_risk_aversion,
                "nonnegative risk-aversion search did not reach the target",
            )
        high_solution = solve_risk_aversion(
            mean,
            decision_covariance,
            risk_aversion=high_gamma,
            maximum_weight=maximum_weight,
            solver_order=solver_order,
            **robust_args,
        )

    best = min(
        (low_solution, high_solution),
        key=lambda result: abs(result.predicted_volatility - target_volatility),
    )
    iterations = 0
    for iterations in range(1, maximum_bisection_iterations + 1):
        gamma = 0.5 * (low_gamma + high_gamma)
        candidate = solve_risk_aversion(
            mean,
            decision_covariance,
            risk_aversion=gamma,
            maximum_weight=maximum_weight,
            solver_order=solver_order,
            **robust_args,
        )
        if abs(candidate.predicted_volatility - target_volatility) < abs(
            best.predicted_volatility - target_volatility
        ):
            best = candidate
        if abs(candidate.predicted_volatility - target_volatility) <= volatility_tolerance:
            best = candidate
            break
        if candidate.predicted_volatility > target_volatility:
            low_gamma = gamma
        else:
            high_gamma = gamma
    status = (
        "ATTAINED"
        if abs(best.predicted_volatility - target_volatility) <= volatility_tolerance
        else "TARGET_NOT_ATTAINABLE"
    )
    return RiskAttainmentResult(
        status=status,
        requested_volatility=target_volatility,
        minimum_feasible_volatility=minimum_vol,
        zero_risk_aversion_volatility=zero_vol,
        solution=best if status == "ATTAINED" else None,
        iterations=iterations,
        lower_risk_aversion=low_gamma,
        upper_risk_aversion=high_gamma,
        reason=None if status == "ATTAINED" else "bisection tolerance was not reached",
    )
