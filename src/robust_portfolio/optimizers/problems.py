"""Feasibility-checked convex GMV and risk-targeted MVO problems."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


class OptimizationFailure(RuntimeError):
    """Raised instead of silently substituting another portfolio."""


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    status: str
    solver: str
    objective_value: float
    predicted_volatility: float
    common_base_volatility: float
    target_volatility: float | None
    target_binding: bool | None
    sum_residual: float
    lower_violation: float
    cap_violation: float
    risk_aversion: float | None = None


def _matrix(frame: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
    if not frame.index.equals(frame.columns):
        raise ValueError("Covariance rows and columns must be identically labeled.")
    matrix = frame.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Covariance must be finite.")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0.0):
        raise ValueError("Covariance must be symmetric.")
    if float(np.linalg.eigvalsh(matrix).min()) < -1e-9:
        raise ValueError("Covariance must be positive semidefinite.")
    return frame.index, (matrix + matrix.T) / 2.0


def _solve(problem: cp.Problem, solver_order: list[str]) -> tuple[str, str]:
    failures = []
    for solver in solver_order:
        if solver not in cp.installed_solvers():
            failures.append(f"{solver}:NOT_INSTALLED")
            continue
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception as error:  # noqa: BLE001 - Try the next configured solver.
            failures.append(f"{solver}:{type(error).__name__}:{error}")
            continue
        if problem.status == cp.OPTIMAL:
            return problem.status, solver
        failures.append(f"{solver}:{problem.status}")
    raise OptimizationFailure("All configured solvers failed: " + " | ".join(failures))


def _clean_and_diagnose(raw, assets, maximum_weight, tolerance) -> tuple[pd.Series, float, float, float]:
    values = np.asarray(raw, dtype=float).ravel()
    if not np.isfinite(values).all():
        raise OptimizationFailure("Solver returned non-finite weights.")
    # Preserve small positive weights: zeroing hundreds of them can create a
    # material full-investment residual even when the solver solution is valid.
    values[(values < 0.0) & (values >= -tolerance)] = 0.0
    if values.min() < -tolerance or values.max() > maximum_weight + tolerance:
        raise OptimizationFailure("Solver weights violate box constraints beyond tolerance.")
    values = np.clip(values, 0.0, maximum_weight)
    difference = 1.0 - values.sum()
    if abs(difference) > tolerance:
        raise OptimizationFailure("Solver weights violate full investment beyond tolerance.")
    if difference > 0.0:
        room = maximum_weight - values
        position = int(np.argmax(room))
        values[position] += difference
    elif difference < 0.0:
        position = int(np.argmax(values))
        values[position] += difference
    weights = pd.Series(values, index=assets, dtype=float)
    return (
        weights,
        abs(float(weights.sum()) - 1.0),
        max(0.0, -float(weights.min())),
        max(0.0, float(weights.max()) - maximum_weight),
    )


def _capacity(assets: int, maximum_weight: float) -> None:
    if assets * maximum_weight < 1.0 - 1e-12:
        raise OptimizationFailure("Maximum-weight constraint makes full investment infeasible.")


def global_minimum_variance(
    covariance: pd.DataFrame,
    *,
    maximum_weight: float,
    solver_order: list[str],
    feasibility_tolerance: float = 5e-7,
) -> OptimizationResult:
    assets, sigma = _matrix(covariance)
    _capacity(len(assets), maximum_weight)
    weights = cp.Variable(len(assets))
    variance = cp.quad_form(weights, cp.psd_wrap(sigma))
    problem = cp.Problem(
        cp.Minimize(variance),
        [cp.sum(weights) == 1.0, weights >= 0.0, weights <= maximum_weight],
    )
    status, solver = _solve(problem, solver_order)
    clean, sum_residual, lower, cap = _clean_and_diagnose(
        weights.value, assets, maximum_weight, feasibility_tolerance
    )
    volatility = float(np.sqrt(max(float(clean @ sigma @ clean), 0.0)))
    return OptimizationResult(
        weights=clean,
        status=status,
        solver=solver,
        objective_value=float(problem.value),
        predicted_volatility=volatility,
        common_base_volatility=volatility,
        target_volatility=None,
        target_binding=None,
        sum_residual=sum_residual,
        lower_violation=lower,
        cap_violation=cap,
    )


def solve_target_risk(
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
    feasibility_tolerance: float = 5e-7,
    target_binding_tolerance: float = 2e-4,
) -> OptimizationResult:
    assets, risk_matrix = _matrix(decision_covariance)
    _capacity(len(assets), maximum_weight)
    mu = mean.reindex(assets).to_numpy(dtype=float)
    if not np.isfinite(mu).all():
        raise ValueError("Expected returns must be finite and aligned.")

    minimum = global_minimum_variance(
        decision_covariance,
        maximum_weight=maximum_weight,
        solver_order=solver_order,
        feasibility_tolerance=feasibility_tolerance,
    )
    if minimum.predicted_volatility > target_volatility + feasibility_tolerance:
        raise OptimizationFailure(
            f"Target volatility {target_volatility:.8f} is below feasible minimum "
            f"{minimum.predicted_volatility:.8f}."
        )

    weights = cp.Variable(len(assets))
    if standard_errors is not None:
        s = standard_errors.reindex(assets).to_numpy(dtype=float)
        robust_return = mu @ weights - float(box_rho) * cp.norm1(cp.multiply(s, weights))
    else:
        robust_return = mu @ weights

    if mean_error_covariance is not None:
        c_assets, c_matrix = _matrix(mean_error_covariance.reindex(index=assets, columns=assets))
        if not c_assets.equals(assets):
            raise ValueError("Mean-error covariance labels do not align.")
        eigenvalues, eigenvectors = np.linalg.eigh(c_matrix)
        root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        robust_return -= float(ellipsoid_rho) * cp.norm(root.T @ weights, 2)

    risk = cp.quad_form(weights, cp.psd_wrap(risk_matrix))
    problem = cp.Problem(
        cp.Maximize(robust_return),
        [
            cp.sum(weights) == 1.0,
            weights >= 0.0,
            weights <= maximum_weight,
            risk <= float(target_volatility) ** 2,
        ],
    )
    status, solver = _solve(problem, solver_order)
    clean, sum_residual, lower, cap = _clean_and_diagnose(
        weights.value, assets, maximum_weight, feasibility_tolerance
    )
    predicted = float(np.sqrt(max(float(clean @ risk_matrix @ clean), 0.0)))
    if predicted > target_volatility + 5.0 * feasibility_tolerance:
        raise OptimizationFailure("Risk-target constraint is violated beyond tolerance.")
    if common_base_covariance is None:
        common_matrix = risk_matrix
    else:
        _, common_matrix = _matrix(common_base_covariance.reindex(index=assets, columns=assets))
    common_volatility = float(np.sqrt(max(float(clean @ common_matrix @ clean), 0.0)))
    return OptimizationResult(
        weights=clean,
        status=status,
        solver=solver,
        objective_value=float(problem.value),
        predicted_volatility=predicted,
        common_base_volatility=common_volatility,
        target_volatility=float(target_volatility),
        target_binding=abs(predicted - target_volatility) <= target_binding_tolerance,
        sum_residual=sum_residual,
        lower_violation=lower,
        cap_violation=cap,
    )


def solve_risk_aversion(
    mean: pd.Series,
    decision_covariance: pd.DataFrame,
    *,
    risk_aversion: float,
    maximum_weight: float,
    solver_order: list[str],
    common_base_covariance: pd.DataFrame | None = None,
    standard_errors: pd.Series | None = None,
    box_rho: float = 0.0,
    mean_error_covariance: pd.DataFrame | None = None,
    ellipsoid_rho: float = 0.0,
    feasibility_tolerance: float = 5e-7,
) -> OptimizationResult:
    """Solve concave robust mean-variance utility for nonnegative gamma."""
    if risk_aversion < 0.0:
        raise ValueError("risk_aversion must be nonnegative.")
    assets, risk_matrix = _matrix(decision_covariance)
    _capacity(len(assets), maximum_weight)
    mu = mean.reindex(assets).to_numpy(dtype=float)
    if not np.isfinite(mu).all():
        raise ValueError("Expected returns must be finite and aligned.")

    weights = cp.Variable(len(assets))
    robust_return = mu @ weights
    if standard_errors is not None:
        s = standard_errors.reindex(assets).to_numpy(dtype=float)
        robust_return -= float(box_rho) * cp.norm1(cp.multiply(s, weights))
    if mean_error_covariance is not None:
        _, error_matrix = _matrix(
            mean_error_covariance.reindex(index=assets, columns=assets)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(error_matrix)
        root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        robust_return -= float(ellipsoid_rho) * cp.norm(root.T @ weights, 2)

    variance = cp.quad_form(weights, cp.psd_wrap(risk_matrix))
    problem = cp.Problem(
        cp.Maximize(robust_return - 0.5 * float(risk_aversion) * variance),
        [cp.sum(weights) == 1.0, weights >= 0.0, weights <= maximum_weight],
    )
    status, solver = _solve(problem, solver_order)
    clean, sum_residual, lower, cap = _clean_and_diagnose(
        weights.value, assets, maximum_weight, feasibility_tolerance
    )
    predicted = float(np.sqrt(max(float(clean @ risk_matrix @ clean), 0.0)))
    if common_base_covariance is None:
        common_matrix = risk_matrix
    else:
        _, common_matrix = _matrix(
            common_base_covariance.reindex(index=assets, columns=assets)
        )
    common_volatility = float(np.sqrt(max(float(clean @ common_matrix @ clean), 0.0)))
    return OptimizationResult(
        weights=clean,
        status=status,
        solver=solver,
        objective_value=float(problem.value),
        predicted_volatility=predicted,
        common_base_volatility=common_volatility,
        target_volatility=None,
        target_binding=None,
        sum_residual=sum_residual,
        lower_violation=lower,
        cap_violation=cap,
        risk_aversion=float(risk_aversion),
    )
