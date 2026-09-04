"""Closed-form robust counterparts and explicit inner-problem diagnostics."""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def box_worst_case_mean(mu, standard_errors, rho: float, weights) -> float:
    mu = np.asarray(mu, dtype=float)
    s = np.asarray(standard_errors, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(mu @ w - float(rho) * np.sum(np.abs(s * w)))


def explicit_box_worst_case_mean(mu, standard_errors, rho: float, weights) -> float:
    mu = np.asarray(mu, dtype=float)
    s = np.asarray(standard_errors, dtype=float)
    w = np.asarray(weights, dtype=float)
    candidate = cp.Variable(len(mu))
    problem = cp.Problem(
        cp.Minimize(candidate @ w),
        [candidate - mu <= rho * s, mu - candidate <= rho * s],
    )
    problem.solve(solver="CLARABEL")
    if problem.status != cp.OPTIMAL:
        raise ValueError(f"Explicit box inner problem failed: {problem.status}")
    return float(problem.value)


def diagonal_robust_covariance(covariance, kappa: float) -> np.ndarray:
    sigma = np.asarray(covariance, dtype=float)
    return sigma + float(kappa) * np.diag(np.diag(sigma))


def explicit_diagonal_worst_case_variance(covariance, kappa: float, weights) -> float:
    sigma = np.asarray(covariance, dtype=float)
    w = np.asarray(weights, dtype=float)
    diagonal_shock = cp.Variable(len(w))
    diagonal = np.diag(sigma)
    problem = cp.Problem(
        cp.Maximize(cp.sum(cp.multiply(diagonal_shock, w**2)) + float(w @ sigma @ w)),
        [diagonal_shock <= kappa * diagonal, diagonal_shock >= -kappa * diagonal],
    )
    problem.solve(solver="CLARABEL")
    if problem.status != cp.OPTIMAL:
        raise ValueError(f"Explicit diagonal inner problem failed: {problem.status}")
    return float(problem.value)


def ellipsoid_worst_case_mean(mu, mean_error_covariance, rho: float, weights) -> float:
    mu = np.asarray(mu, dtype=float)
    covariance = np.asarray(mean_error_covariance, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(mu @ w - float(rho) * np.sqrt(max(float(w @ covariance @ w), 0.0)))


def explicit_ellipsoid_worst_case_mean(mu, mean_error_covariance, rho: float, weights) -> float:
    mu = np.asarray(mu, dtype=float)
    covariance = np.asarray(mean_error_covariance, dtype=float)
    w = np.asarray(weights, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    u = cp.Variable(len(mu))
    candidate = mu + root @ u
    problem = cp.Problem(cp.Minimize(candidate @ w), [cp.norm(u, 2) <= rho])
    problem.solve(solver="CLARABEL")
    if problem.status != cp.OPTIMAL:
        raise ValueError(f"Explicit ellipsoid inner problem failed: {problem.status}")
    return float(problem.value)
