import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_revision_scores import compare_run


def record(good, bad, token_hash="tokens"):
    return {
        "pair_id": "category:0001",
        "category": "category",
        "in_baseline": False,
        "good": {"n_tokens": 2, "token_ids_sha256": token_hash, "mean_logp": good, "total_logp": 2 * good},
        "bad": {"n_tokens": 2, "token_ids_sha256": token_hash, "mean_logp": bad, "total_logp": 2 * bad},
    }


class ScoreComparisonTests(unittest.TestCase):
    def test_reports_score_drift_and_preference_changes(self):
        rows = compare_run([record(-1, -2)], [record(-2, -1)], "eurollm/fp16")
        self.assertEqual([row["preference_changes"] for row in rows], [1, 1])
        self.assertEqual([row["reference_correct"] for row in rows], [1, 1])
        self.assertEqual([row["reproduced_correct"] for row in rows], [0, 0])
        self.assertEqual([row["max_abs_sentence_score_delta"] for row in rows], [1.0, 2.0])

    def test_rejects_different_tokenization(self):
        with self.assertRaisesRegex(ValueError, "token"):
            compare_run([record(-1, -2)], [record(-1, -2, "different")], "eurollm/fp16")


if __name__ == "__main__":
    unittest.main()
