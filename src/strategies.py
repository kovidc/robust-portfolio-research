import numpy as np
import pandas as pd

from covariance import estimate_cmiewma_covariance, estimate_iewma_covariance

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - depends on local environment
    cp = None

_MISSING_CVXPY_WARNING_SHOWN = False


def equal_weight_strategy(returns_window):
    """Assign the same weight to every ETF."""
    asset_count = len(returns_window.columns)
    if asset_count == 0:
        raise ValueError("The return window has no assets.")

    weights = np.repeat(1.0 / asset_count, asset_count)
    return pd.Series(weights, index=returns_window.columns, name="equal_weight")


def _build_statistics(returns_window, covariance_method="sample"):
    """
    Create annualized estimates used by the optimization strategies.

    Supported covariance methods:
    - sample: simple historical sample covariance
    - iewma: single IEWMA covariance forecast
    - cmiewma: combined-multiple IEWMA covariance forecast
    """
    clean_window = returns_window.dropna(axis=1, how="any")

    if clean_window.empty:
        raise ValueError("The return window is empty after dropping missing columns.")

    observation_count = len(clean_window)
    if observation_count < 2:
        raise ValueError("At least two return observations are needed for optimization.")

    mean_returns = clean_window.mean() * 252

    if covariance_method == "sample":
        covariance = clean_window.cov() * 252
    elif covariance_method == "iewma":
        covariance = pd.DataFrame(
            estimate_iewma_covariance(clean_window) * 252,
            index=clean_window.columns,
            columns=clean_window.columns,
        )
    elif covariance_method == "cmiewma":
        covariance = pd.DataFrame(
            estimate_cmiewma_covariance(clean_window) * 252,
            index=clean_window.columns,
            columns=clean_window.columns,
        )
    else:
        raise ValueError(f"Unsupported covariance method: {covariance_method}")

    covariance = (covariance + covariance.T) / 2.0
    covariance = covariance + np.eye(len(covariance)) * 1e-6

    standard_error = clean_window.std(ddof=1) / np.sqrt(observation_count)
    standard_error = (standard_error * 252).clip(lower=1e-8)

    return clean_window.columns, mean_returns, covariance, standard_error


def _nearest_psd(matrix, minimum_eigenvalue=1e-8):
    """
    Project a symmetric matrix to the nearest positive semidefinite matrix.

    This prevents cvxpy from failing when the sample covariance matrix is
    numerically close to PSD but not cleanly recognized as such.
    """
    symmetric_matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
    clipped_eigenvalues = np.clip(eigenvalues, minimum_eigenvalue, None)
    psd_matrix = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T
    psd_matrix = (psd_matrix + psd_matrix.T) / 2.0
    return psd_matrix


def _normalize_with_cap(raw_weights, asset_names, max_weight):
    """
    Clean up solver output so the weights are long-only, sum to one,
    and remain near the requested cap even after rounding noise.
    """
    weights = pd.Series(np.asarray(raw_weights).ravel(), index=asset_names, dtype=float)
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = weights.clip(lower=0.0, upper=max_weight)

    total_weight = weights.sum()
    if total_weight <= 0:
        asset_count = len(asset_names)
        return pd.Series(np.repeat(1.0 / asset_count, asset_count), index=asset_names, dtype=float)

    weights = weights / total_weight
    tolerance = 1e-10

    for _ in range(20):
        overweight_mask = weights > (max_weight + tolerance)
        if not overweight_mask.any():
            break

        excess_weight = (weights[overweight_mask] - max_weight).sum()
        weights[overweight_mask] = max_weight

        underweight_mask = weights < (max_weight - tolerance)
        if not underweight_mask.any():
            break

        room = max_weight - weights[underweight_mask]
        room_total = room.sum()
        if room_total <= 0:
            break

        weights.loc[underweight_mask] += excess_weight * (room / room_total)

    weights = weights.clip(lower=0.0, upper=max_weight)
    weights = weights / weights.sum()
    return weights


def _solve_markowitz_problem(mean_returns, covariance, max_weight, gamma):
    """Solve the optimization problem with cvxpy."""
    if cp is None:
        raise ImportError("cvxpy is not installed.")

    asset_count = len(mean_returns)
    if asset_count * max_weight < 1.0:
        raise ValueError("The max weight is too small to build a fully invested portfolio.")

    weights = cp.Variable(asset_count)
    covariance_matrix = _nearest_psd(covariance.to_numpy())
    covariance_psd = cp.psd_wrap(covariance_matrix)
    objective = mean_returns.to_numpy() @ weights - gamma * cp.quad_form(weights, covariance_psd)

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight,
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)
    solver_candidates = [cp.OSQP, cp.CLARABEL, cp.SCS]

    last_error = None
    for solver_name in solver_candidates:
        try:
            problem.solve(solver=solver_name, verbose=False)
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return weights.value
        except Exception as error:
            last_error = error

    if last_error is not None:
        raise ValueError(f"Optimization failed after trying multiple solvers: {last_error}")

    raise ValueError(f"Optimization failed with status: {problem.status}")


def _handle_strategy_failure(strategy_name, error, fallback_weights):
    """Print a readable warning and return the fallback portfolio."""
    global _MISSING_CVXPY_WARNING_SHOWN

    if cp is None:
        if not _MISSING_CVXPY_WARNING_SHOWN:
            print(
                "cvxpy is not installed, so the classical and robust Markowitz strategies are "
                "falling back to equal-weight portfolios. Install the project requirements to "
                "enable optimization."
            )
            _MISSING_CVXPY_WARNING_SHOWN = True
    else:
        print(f"{strategy_name} failed, using equal weight fallback. Reason: {error}")

    return fallback_weights


def classical_markowitz_strategy(returns_window, gamma=10, max_weight=0.10):
    """Classical mean-variance optimization using a single IEWMA covariance forecast."""
    fallback_weights = equal_weight_strategy(returns_window)

    try:
        asset_names, mean_returns, covariance, _ = _build_statistics(
            returns_window,
            covariance_method="iewma",
        )
        raw_weights = _solve_markowitz_problem(
            mean_returns=mean_returns,
            covariance=covariance,
            max_weight=max_weight,
            gamma=gamma,
        )
        return _normalize_with_cap(raw_weights, asset_names, max_weight)
    except Exception as error:
        return _handle_strategy_failure("Classical Markowitz", error, fallback_weights)


def robust_markowitz_strategy(
    returns_window,
    gamma=10,
    rho=1.0,
    max_weight=0.10,
    cov_uncertainty=0.10,
):
    """
    Robust mean-variance optimization with:
    - a CM-IEWMA covariance forecast,
    - a worst-case expected-return haircut, and
    - a diagonal covariance uncertainty bump.

    This follows the spirit of the `cvx_options` reference repo more closely
    than the earlier custom L2 penalty. Because the portfolio is long-only,
    the robust return term simplifies from mu.T @ w - rho.T @ |w| to
    (mu - rho).T @ w.
    """
    fallback_weights = equal_weight_strategy(returns_window)

    try:
        asset_names, mean_returns, covariance, standard_error = _build_statistics(
            returns_window,
            covariance_method="cmiewma",
        )
        robust_mean_returns = mean_returns - rho * standard_error
        covariance_bump = np.diag(np.diag(covariance.to_numpy())) * cov_uncertainty
        robust_covariance = covariance + covariance_bump
        raw_weights = _solve_markowitz_problem(
            mean_returns=robust_mean_returns,
            covariance=robust_covariance,
            max_weight=max_weight,
            gamma=gamma,
        )
        return _normalize_with_cap(raw_weights, asset_names, max_weight)
    except Exception as error:
        return _handle_strategy_failure("Robust Markowitz", error, fallback_weights)
