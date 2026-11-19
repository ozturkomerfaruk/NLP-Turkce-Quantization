"""Verify the revision dataset without modifying the submitted experiment."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RevisionDatasetTests(unittest.TestCase):
    def test_build_preserves_source_records_and_baseline_membership(self):
        baseline_path = ROOT / "data/processed/turblimp_1000.jsonl"
        baseline_bytes = baseline_path.read_bytes()
        baseline = {
            row["pair_id"]: row
            for row in map(json.loads, baseline_bytes.splitlines())
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "revision_v2"
            command = [
                sys.executable, str(ROOT / "scripts/prepare_revision_dataset.py"),
                "--output-dir", str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            dataset = output / "turblimp_3662.jsonl"
            manifest_path = output / "manifest.json"
            dataset_bytes = dataset.read_bytes()
            manifest_bytes = manifest_path.read_bytes()
            rows = list(map(json.loads, dataset_bytes.splitlines()))

            self.assertEqual(len(rows), 3662)
            self.assertEqual(len({row["pair_id"] for row in rows}), 3662)
            self.assertEqual(
                len({(row["sentence_good"], row["sentence_bad"]) for row in rows}),
                3662,
            )
            self.assertEqual(Counter(row["category"] for row in rows), {
                "subject_agreement": 934,
                "argument_structure_transitive": 835,
                "nominalization": 945,
                "irregular_forms": 948,
            })
            submitted = {
                row["pair_id"]: {key: value for key, value in row.items() if key != "in_baseline"}
                for row in rows if row["in_baseline"]
            }
            self.assertEqual(submitted, baseline)
            self.assertEqual(sum(row["in_baseline"] is False for row in rows), 2662)
            manifest = json.loads(manifest_bytes)
            self.assertEqual(manifest["dataset"]["sha256"], hashlib.sha256(dataset_bytes).hexdigest())
            self.assertEqual(manifest["baseline"]["sha256"], hashlib.sha256(baseline_bytes).hexdigest())

            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(dataset.read_bytes(), dataset_bytes)
            self.assertEqual(manifest_path.read_bytes(), manifest_bytes)

        self.assertEqual(baseline_path.read_bytes(), baseline_bytes)
