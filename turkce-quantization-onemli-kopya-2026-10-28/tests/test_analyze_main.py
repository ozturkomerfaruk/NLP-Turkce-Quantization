import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import analyze_main


class SavedExperimentTests(unittest.TestCase):
    def test_loads_project_manifest_and_saved_results(self):
        _, ids, _, scores, _, _ = analyze_main.load_data()
        self.assertEqual(len(ids), 1000)
        self.assertEqual(
            {name: sum(row['correct'] for row in rows.values())
             for name, rows in scores.items()},
            {'fp16': 912, 'q8': 915, 'q4': 884},
        )

    def test_accepts_small_positive_margin_under_scoring_protocol(self):
        read_jsonl = analyze_main.read_jsonl

        def read_with_small_margin(path):
            rows = read_jsonl(path)
            if path == analyze_main.RESULTS / 'fp16' / 'chunk_00.jsonl':
                row = next(row for row in rows if row['correct'])
                row['good']['mean_logp'] = row['bad']['mean_logp'] + 5e-7
                row['mean_logp_diff'] = row['good']['mean_logp'] - row['bad']['mean_logp']
            return rows

        with patch.object(analyze_main, 'read_jsonl', side_effect=read_with_small_margin):
            analyze_main.load_data()


if __name__ == '__main__':
    unittest.main()
