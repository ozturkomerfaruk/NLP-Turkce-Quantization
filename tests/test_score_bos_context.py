"""Check explicit BOS context for tokenizers without a configured BOS."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from score_main import score_sentence


class StopAfterInputs(Exception):
    pass


class BosContextTests(unittest.TestCase):
    def test_explicit_bos_predicts_first_token_and_is_not_a_target(self):
        tokenizer = SimpleNamespace(bos_token_id=None, encode=Mock(return_value=[41, 42]))
        mx = SimpleNamespace(array=Mock(side_effect=lambda data, dtype: data), int32="int32")
        model = Mock(side_effect=StopAfterInputs)

        with self.assertRaises(StopAfterInputs):
            score_sentence("örnek", model, tokenizer, mx, None, 3, bos_token_id=151643)

        tokenizer.encode.assert_called_once_with("örnek", add_special_tokens=False)
        self.assertEqual(mx.array.call_args_list, [
            call([[151643, 41]], dtype="int32"),
            call([[41, 42]], dtype="int32"),
        ])
        model.assert_called_once_with([[151643, 41]])

    def test_missing_bos_still_requires_an_explicit_choice(self):
        tokenizer = SimpleNamespace(bos_token_id=None)
        with self.assertRaisesRegex(ValueError, "BOS token ID is missing"):
            score_sentence("örnek", None, tokenizer, None, None, 3)
