#!/usr/bin/env python3
"""Compare recomputed revision CSVs with the archived statistical tables."""

import argparse
import csv
import math
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent.parent / "results/revision_v2/statistics"


def equivalent(expected, actual):
    if expected == actual:
        return True
    try:
        return math.isclose(float(expected), float(actual), rel_tol=2e-10, abs_tol=1e-12)
    except ValueError:
        return False


def compare(reference, recomputed):
    with reference.open(encoding="utf-8", newline="") as stream:
        expected = list(csv.reader(stream))
    with recomputed.open(encoding="utf-8", newline="") as stream:
        actual = list(csv.reader(stream))
    if not expected or not actual or expected[0] != actual[0] or len(expected) != len(actual):
        raise ValueError(f"{reference.name}: different header or row count")

    for line, (reference_row, recomputed_row) in enumerate(zip(expected[1:], actual[1:]), 2):
        if len(reference_row) != len(recomputed_row):
            raise ValueError(f"{reference.name}:{line}: different column count")
        for column, expected_value, actual_value in zip(expected[0], reference_row, recomputed_row):
            if not equivalent(expected_value, actual_value):
                raise ValueError(
                    f"{reference.name}:{line} ({column}): expected {expected_value!r}, got {actual_value!r}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE, help="Archived statistics directory")
    args = parser.parse_args()
    references = sorted(args.reference.glob("*.csv"))
    try:
        if not references or {p.name for p in args.tables.glob("*.csv")} != {p.name for p in references}:
            raise ValueError("The recomputed and archived CSV filenames differ")
        for reference in references:
            compare(reference, args.tables / reference.name)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Verified {len(references)} statistical tables against the archived CSVs.")


if __name__ == "__main__":
    main()
