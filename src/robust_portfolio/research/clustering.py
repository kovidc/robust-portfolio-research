"""Training-only ETF redundancy diagnostics and deterministic medoids."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def correlation_distance(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty or returns.isna().any().any():
        raise ValueError("Clustering requires a complete training return panel.")
    ordered = returns.reindex(columns=sorted(returns.columns))
    correlation = ordered.corr().clip(-1.0, 1.0)
    distance = np.sqrt(np.maximum((1.0 - correlation.to_numpy()) / 2.0, 0.0))
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return pd.DataFrame(distance, index=ordered.columns, columns=ordered.columns)


def hierarchical_clusters(
    distance: pd.DataFrame,
    *,
    correlation_threshold: float,
    method: str = "average",
) -> pd.Series:
    if not 0.0 <= correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must lie in [0, 1].")
    if not distance.index.equals(distance.columns):
        raise ValueError("Distance rows and columns must have identical labels.")
    ordered = sorted(distance.index)
    matrix = distance.reindex(index=ordered, columns=ordered).to_numpy(dtype=float)
    if len(ordered) == 1:
        return pd.Series([1], index=ordered, dtype=int)
    cutoff = np.sqrt((1.0 - float(correlation_threshold)) / 2.0)
    raw = fcluster(linkage(squareform(matrix, checks=True), method=method), cutoff, criterion="distance")
    members = {}
    for asset, label in zip(ordered, raw):
        members.setdefault(int(label), []).append(asset)
    stable_order = sorted(members, key=lambda label: tuple(sorted(members[label])))
    remap = {label: position + 1 for position, label in enumerate(stable_order)}
    return pd.Series([remap[int(label)] for label in raw], index=ordered, dtype=int)


def cluster_medoids(distance: pd.DataFrame, clusters: pd.Series) -> tuple[str, ...]:
    medoids = []
    for _, members in clusters.groupby(clusters, sort=True):
        assets = sorted(members.index)
        within = distance.loc[assets, assets].mean(axis=1)
        minimum = float(within.min())
        tied = sorted(within.index[np.isclose(within.to_numpy(), minimum, atol=1e-14, rtol=0.0)])
        medoids.append(tied[0])
    return tuple(medoids)


def covariance_spectrum(covariance: pd.DataFrame) -> dict[str, float]:
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance.to_numpy(dtype=float)), 0.0)
    positive = eigenvalues[eigenvalues > 1e-14]
    condition = float(eigenvalues.max() / max(float(eigenvalues.min()), 1e-18))
    probabilities = eigenvalues / eigenvalues.sum()
    nonzero = probabilities > 0.0
    effective_rank = float(np.exp(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))))
    return {
        "condition_number": condition,
        "effective_rank": effective_rank,
        "minimum_eigenvalue": float(eigenvalues.min()),
        "second_smallest_eigenvalue": float(np.sort(eigenvalues)[1]) if len(eigenvalues) > 1 else float(eigenvalues[0]),
        "positive_eigenvalue_count": len(positive),
    }
