"""Check that model comparison distinguishes tensor bytes from file layout."""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_eurollm_revision import compare_weights


class WeightComparisonTests(unittest.TestCase):
    def test_detects_payload_changes_but_ignores_header_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.safetensors"
            new = root / "new.safetensors"
            header = json.dumps({
                "example": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}
            }).encode()

            def write(path, metadata, payload):
                path.write_bytes(struct.pack("<Q", len(metadata)) + metadata + payload)

            write(old, header, b"ab")
            write(new, header + b" ", b"ab")
            comparison = compare_weights(old, new)
            self.assertFalse(comparison["file_identical"])
            self.assertTrue(comparison["tensors_identical"])

            write(new, header, b"ac")
            comparison = compare_weights(old, new)
            self.assertFalse(comparison["tensors_identical"])
            self.assertEqual(comparison["different_tensors"], ["example"])
