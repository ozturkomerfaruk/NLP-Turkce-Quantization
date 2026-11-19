#!/usr/bin/env python3
"""Compare two audited revision runs by pair, preference and sentence score."""

import argparse
import csv
import math
import sys
from pathlib import Path

from prepare_revision_analysis import METHODS, verified_records
from score_revision import FAMILIES, PRECISIONS, SIDES, read_json


def compare_run(reference, reproduced, run):
    pairs = 0
    summaries = {
        method: dict(run=run, score=method, reference_correct=0, reproduced_correct=0,
                     preference_changes=0, max_abs_sentence_score_delta=0.0)
        for method in METHODS
    }
    for before, after in zip(reference, reproduced, strict=True):
        pairs += 1
        pair_id = before["pair_id"]
        if any(before[field] != after[field] for field in ("pair_id", "category", "in_baseline")):
            raise ValueError(f"{run}: mismatched pair at {pair_id}")
        for side in SIDES:
            if any(before[side][field] != after[side][field] for field in ("token_ids_sha256", "n_tokens")):
                raise ValueError(f"{run}: tokenization differs for {pair_id}/{side}")
        for method, summary in summaries.items():
            field = f"{method}_logp"
            old_good, old_bad = (before[side][field] for side in SIDES)
            new_good, new_bad = (after[side][field] for side in SIDES)
            if not all(math.isfinite(value) for value in (old_good, old_bad, new_good, new_bad)):
                raise ValueError(f"{run}: non-finite score for {pair_id}")
            old_decision = int(old_good > old_bad) - int(old_good < old_bad)
            new_decision = int(new_good > new_bad) - int(new_good < new_bad)
            summary["reference_correct"] += old_decision == 1
            summary["reproduced_correct"] += new_decision == 1
            summary["preference_changes"] += old_decision != new_decision
            summary["max_abs_sentence_score_delta"] = max(
                summary["max_abs_sentence_score_delta"], abs(old_good - new_good), abs(old_bad - new_bad)
            )
    return [{**summary, "pairs": pairs} for summary in summaries.values()]


def compare_results(reference_dir, reproduced_dir):
    reference = read_json(reference_dir / "verification.json")
    reproduced = read_json(reproduced_dir / "verification.json")
    for key in ("dataset_sha256", "method", "scoring_code_sha256"):
        if reference[key] != reproduced[key]:
            raise ValueError(f"Incompatible audit field: {key}")

    summaries = []
    for model in FAMILIES:
        for precision in PRECISIONS:
            run = f"{model}/{precision}"
            old, new = reference["runs"][run], reproduced["runs"][run]
            for key in ("source_revision", "weights_sha256", "bos_context_id", "pairs"):
                if old[key] != new[key]:
                    raise ValueError(f"{run}: incompatible audit field {key}")
            rows = compare_run(
                verified_records(reference, model, precision, reference_dir),
                verified_records(reproduced, model, precision, reproduced_dir),
                run,
            )
            if rows[0]["pairs"] != old["pairs"]:
                raise ValueError(f"{run}: compared pair count differs from audit")
            summaries.extend(rows)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Archived results directory with verification.json")
    parser.add_argument("reproduced", type=Path, help="Recomputed results directory with verification.json")
    args = parser.parse_args()
    try:
        rows = compare_results(args.reference, args.reproduced)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
