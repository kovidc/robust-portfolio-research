"""Validate regenerated core outputs without pinning an obsolete source commit."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from robust_portfolio.data.providers import sha256_file
from robust_portfolio.research.configuration import CoreExperimentConfig
from robust_portfolio.research.final_analysis import _validate_core
from robust_portfolio.research.final_analysis_configuration import (
    FinalAnalysisConfig,
)


class TestCoreInputs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        core_path = self.root / "core.json"
        core_path.write_bytes((ROOT / "configs/core_experiment.json").read_bytes())
        self.core_config = CoreExperimentConfig.load(core_path)
        self.core_dir = self.root / "core"
        self.core_dir.mkdir()
        self.data = self.root / "returns.csv"
        self.data.write_text("date,A\n2020-01-02,0.01\n")
        self.artifact = self.core_dir / "targets.csv"
        self.artifact.write_text("asset,weight\nA,1.0\n")
        self.manifest = {
            "configuration": {"canonical_sha256": self.core_config.sha256},
            "git": {"commit": "a-new-producing-commit"},
            "inputs": {"returns": {"path": str(self.data), "sha256": sha256_file(self.data)}},
            "artifact_locations": {"targets": str(self.artifact)},
        }
        self.write_manifest()
        payload = json.loads((ROOT / "configs/final_analysis.json").read_text())
        payload["inputs"]["core_config"] = str(core_path)
        payload["inputs"]["core_artifact_directory"] = str(self.core_dir)
        payload["experiment"]["core_config_sha256"] = self.core_config.sha256
        final_path = self.root / "final.json"
        final_path.write_text(json.dumps(payload))
        self.config = FinalAnalysisConfig.load(final_path)

    def write_manifest(self):
        (self.core_dir / "run_manifest.json").write_text(json.dumps(self.manifest))

    def test_accepts_regeneration_from_a_new_commit(self):
        directory, manifest = _validate_core(self.config, self.root)
        self.assertEqual(directory, self.core_dir)
        self.assertEqual(manifest["git"]["commit"], "a-new-producing-commit")

    def test_rejects_stale_artifact_configuration(self):
        self.manifest["configuration"]["canonical_sha256"] = "stale"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "configuration hash"):
            _validate_core(self.config, self.root)

    def test_rejects_changed_current_configuration(self):
        payload = self.core_config.payload.copy()
        payload["means"] = {**payload["means"], "ewma_half_life": 42}
        self.core_config.path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "current core configuration"):
            _validate_core(self.config, self.root)

    def test_rejects_changed_input_data(self):
        self.data.write_text("date,A\n2020-01-02,0.02\n")
        with self.assertRaisesRegex(ValueError, "input hash changed"):
            _validate_core(self.config, self.root)

    def test_rejects_missing_artifact(self):
        self.artifact.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
            _validate_core(self.config, self.root)
