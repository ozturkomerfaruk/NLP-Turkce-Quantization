"""Verify the revision pilot selection and reject incompatible checkpoints."""

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from score_revision import load_dataset, pair_scores, select_rows, validate_chunk
from score_main import token_hash


class RevisionScoringTests(unittest.TestCase):
    def test_smoke_uses_two_baseline_pairs_per_category(self):
        rows, _ = load_dataset()
        smoke = select_rows(rows, "smoke")
        self.assertEqual(len(rows), 3662)
        self.assertEqual(len(smoke), 8)
        self.assertTrue(all(row["in_baseline"] for row in smoke))
        self.assertEqual(set(Counter(row["category"] for row in smoke).values()), {2})

    def test_resume_checks_identity_tokenization_and_both_score_formulas(self):
        row = {"pair_id": "category:0001", "category": "category", "in_baseline": True}
        identity = {"model": "eurollm", "precision": "fp16", "mode": "smoke"}
        good = {"total_logp": -6.0, "mean_logp": -2.0, "n_tokens": 3,
                "token_ids_sha256": token_hash([1, 2, 3])}
        bad = {"total_logp": -4.0, "mean_logp": -4.0, "n_tokens": 1,
               "token_ids_sha256": token_hash([2])}
        expected = ((3, good["token_ids_sha256"]), (1, bad["token_ids_sha256"]))
        record = pair_scores(row, good, bad)
        self.assertTrue(record["mean_correct"])
        self.assertFalse(record["total_correct"])
        checkpoint = {"identity": identity, "chunk_index": 0, "records": [record]}
        validate_chunk(checkpoint, identity, 0, [row], [expected])
        with self.assertRaises(ValueError):
            validate_chunk(checkpoint, {**identity, "mode": "main"}, 0, [row], [expected])
        with self.assertRaises(ValueError):
            validate_chunk(checkpoint, identity, 0, [row], [expected[::-1]])
        good["total_logp"] = float("nan")
        with self.assertRaises(ValueError):
            validate_chunk(checkpoint, identity, 0, [row], [expected])
