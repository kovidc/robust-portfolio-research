import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from robust_portfolio.research.clustering import (
    cluster_medoids,
    correlation_distance,
    covariance_spectrum,
    hierarchical_clusters,
)
from robust_portfolio.research.robustness import (
    allocation_diagnostics,
    clone_distortions,
    psd_covariance_perturbations,
)


class TestRobustnessAndClustering(unittest.TestCase):
    def test_allocation_metrics_and_clone_economic_exposure(self):
        baseline = pd.Series({"A": 0.6, "B": 0.4})
        augmented = pd.Series({"A": 0.2, "A_CLONE": 0.4, "B": 0.4})
        classes = pd.Series({"A": "x", "A_CLONE": "x", "B": "y"})
        raw = allocation_diagnostics(augmented, baseline)
        clone = clone_distortions(
            augmented, baseline, source_asset="A", clone_asset="A_CLONE",
            asset_classes=classes,
        )
        self.assertAlmostEqual(raw["l1_weight_change"], 0.8)
        self.assertAlmostEqual(clone["economic_exposure_l1_change"], 0.0)
        self.assertAlmostEqual(clone["asset_class_exposure_l1_change"], 0.0)

    def test_covariance_perturbations_remain_psd(self):
        covariance = pd.DataFrame(
            [[0.04, 0.018], [0.018, 0.01]], index=["A", "B"], columns=["A", "B"]
        )
        shocks = psd_covariance_perturbations(
            covariance, variance_scale=1.1,
            correlation_to_identity_weight=0.1, leading_eigenvalue_scale=1.1,
        )
        self.assertEqual(set(shocks), {"variance_scale", "correlation_to_identity", "leading_eigenvalue"})
        for shocked in shocks.values():
            self.assertGreaterEqual(np.linalg.eigvalsh(shocked).min(), -1e-12)

    def test_clustering_is_deterministic_and_medoid_rule_is_exact(self):
        returns = pd.DataFrame(
            {"C": [1, 2, 3, 4, 5], "A": [1, 2, 3, 4, 5.1], "B": [-1, 0, 1, 0, -1]},
            dtype=float,
        )
        distance = correlation_distance(returns)
        first = hierarchical_clusters(distance, correlation_threshold=0.95)
        second = hierarchical_clusters(distance, correlation_threshold=0.95)
        pd.testing.assert_series_equal(first, second)
        self.assertEqual(cluster_medoids(distance, first), cluster_medoids(distance, second))

    def test_effective_rank(self):
        covariance = pd.DataFrame(np.eye(4))
        self.assertAlmostEqual(covariance_spectrum(covariance)["effective_rank"], 4.0)

    def test_future_rows_do_not_change_training_clustering(self):
        training = pd.DataFrame(np.arange(24).reshape(8, 3), columns=list("ABC"), dtype=float)
        future = pd.DataFrame([[100.0, -100.0, 5.0]], columns=list("ABC"))
        base = hierarchical_clusters(correlation_distance(training), correlation_threshold=0.9)
        asof = hierarchical_clusters(correlation_distance(pd.concat([training, future]).iloc[:8]), correlation_threshold=0.9)
        pd.testing.assert_series_equal(base, asof)
