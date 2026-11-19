"""Check the Stage 9 analysis inputs and the frozen cluster rule."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_revision_analysis import cluster_key, build_analysis_rows


class RevisionAnalysisInputsTests(unittest.TestCase):
    def test_common_prefix_and_turkish_casefold_are_preserved(self):
        pair = {"category": "agreement", "sentence_good": "Ama İyi çocuklar geldi.",
                "sentence_bad": "AMA İyi çocuklar geldin."}
        self.assertEqual(cluster_key(pair), ("agreement", "iyi çocuklar geldi.", "iyi çocuklar geldin."))
        pair["sentence_bad"] = "Ve İyi çocuklar geldin."
        self.assertEqual(cluster_key(pair)[1], "ama iyi çocuklar geldi.")

    def test_all_verified_configurations_align_on_unique_pairs(self):
        rows = build_analysis_rows()
        self.assertEqual(len(rows), 3662)
        self.assertEqual(len({row["pair_id"] for row in rows}), 3662)
        self.assertEqual(sum(row["in_baseline"] for row in rows), 1000)
        for row in rows:
            for model in ("eurollm", "qwen"):
                for precision in ("fp16", "q8", "q4"):
                    for method in ("mean", "total"):
                        self.assertIn(row[f"{model}_{precision}_{method}"], (-1, 0, 1))
