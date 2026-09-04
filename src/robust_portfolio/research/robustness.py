"""Direct allocation robustness, PSD shocks, and clone exposure helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from robust_portfolio.estimators.covariance import nearest_psd


def allocation_diagnostics(candidate: pd.Series, baseline: pd.Series) -> dict[str, float]:
    assets = baseline.index.union(candidate.index)
    left = baseline.reindex(assets, fill_value=0.0).to_numpy(dtype=float)
    right = candidate.reindex(assets, fill_value=0.0).to_numpy(dtype=float)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return {
        "l1_weight_change": float(np.abs(right - left).sum()),
        "cosine_similarity": float(right @ left / denominator) if denominator > 0.0 else np.nan,
        "hhi": float(np.square(right).sum()),
        "effective_holdings": float(1.0 / np.square(right).sum()),
    }


def asset_class_l1_change(
    candidate: pd.Series,
    baseline: pd.Series,
    asset_classes: pd.Series,
) -> float:
    assets = baseline.index.union(candidate.index)
    classes = asset_classes.reindex(assets)
    if classes.isna().any():
        raise ValueError("Every asset requires an asset-class label.")
    base = baseline.reindex(assets, fill_value=0.0).groupby(classes).sum()
    perturbed = candidate.reindex(assets, fill_value=0.0).groupby(classes).sum()
    return float((perturbed - base).abs().sum())


def psd_covariance_perturbations(
    covariance: pd.DataFrame,
    *,
    variance_scale: float,
    correlation_to_identity_weight: float,
    leading_eigenvalue_scale: float,
) -> dict[str, pd.DataFrame]:
    if variance_scale < 1.0 or leading_eigenvalue_scale < 1.0:
        raise ValueError("Variance and eigenvalue shocks must not reduce risk.")
    if not 0.0 <= correlation_to_identity_weight <= 1.0:
        raise ValueError("Correlation blend weight must lie in [0, 1].")
    assets = covariance.index
    matrix = covariance.to_numpy(dtype=float)
    standard_deviation = np.sqrt(np.maximum(np.diag(matrix), 1e-18))
    correlation = matrix / np.outer(standard_deviation, standard_deviation)
    correlation_shock = (
        (1.0 - correlation_to_identity_weight) * correlation
        + correlation_to_identity_weight * np.eye(len(assets))
    )
    correlation_matrix = np.outer(standard_deviation, standard_deviation) * correlation_shock
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues[-1] *= leading_eigenvalue_scale
    eigen_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    shocks = {
        "variance_scale": matrix * variance_scale,
        "correlation_to_identity": correlation_matrix,
        "leading_eigenvalue": eigen_matrix,
    }
    output = {}
    for name, value in shocks.items():
        psd, _, _ = nearest_psd(value, absolute_floor=1e-10, relative_floor=1e-8)
        output[name] = pd.DataFrame(psd, index=assets, columns=assets)
    return output


def fold_clone_weight(weights: pd.Series, source_asset: str, clone_asset: str) -> pd.Series:
    folded = weights.drop(labels=[clone_asset], errors="ignore").copy()
    folded.loc[source_asset] = float(folded.get(source_asset, 0.0)) + float(weights.get(clone_asset, 0.0))
    return folded


def clone_distortions(
    augmented: pd.Series,
    baseline: pd.Series,
    *,
    source_asset: str,
    clone_asset: str,
    asset_classes: pd.Series,
) -> dict[str, float]:
    folded = fold_clone_weight(augmented, source_asset, clone_asset)
    diagnostics = allocation_diagnostics(augmented, baseline)
    diagnostics.update(
        {
            "economic_exposure_l1_change": allocation_diagnostics(folded, baseline)["l1_weight_change"],
            "source_plus_clone_weight": float(augmented.get(source_asset, 0.0) + augmented.get(clone_asset, 0.0)),
            "source_baseline_weight": float(baseline.get(source_asset, 0.0)),
            "asset_class_exposure_l1_change": asset_class_l1_change(
                folded, baseline, asset_classes
            ),
        }
    )
    return diagnostics
