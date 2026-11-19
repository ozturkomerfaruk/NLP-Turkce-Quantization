"""Build the full, deduplicated TurBLiMP dataset for the manuscript revision."""

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from prepare_dataset import (
    CATEGORIES,
    OUT_DIR,
    RAW_DIR,
    ROOT,
    SOURCE_BASE,
    SOURCE_REVISION,
    read_unique_pairs,
    sha256,
    write_jsonl,
)

EXPECTED_COUNTS = {
    "subject_agreement": 934,
    "argument_structure_transitive": 835,
    "nominalization": 945,
    "irregular_forms": 948,
}
DATASET_NAME = "turblimp_3662.jsonl"
MANIFEST_NAME = "manifest.json"


def require_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def build_records() -> tuple[list, dict]:
    reference = json.loads((OUT_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    if reference["source_revision"] != SOURCE_REVISION:
        raise ValueError("The baseline manifest does not match the pinned TurBLiMP revision")

    baseline_path = OUT_DIR / "turblimp_1000.jsonl"
    baseline_hash = require_hash(baseline_path, reference["main_dataset"]["sha256"])
    with baseline_path.open(encoding="utf-8") as stream:
        baseline = [json.loads(line) for line in stream]

    baseline_by_id = {row["pair_id"]: row for row in baseline}
    baseline_counts = Counter(row["category"] for row in baseline)
    if len(baseline) != 1000 or len(baseline_by_id) != 1000 or baseline_counts != dict.fromkeys(CATEGORIES, 250):
        raise ValueError("Expected 1,000 distinct baseline IDs, 250 in each category")

    seen_pairs = set()
    matched_ids = set()
    records = []
    source_statistics = {}
    raw_hashes = {}
    for category, filename in CATEGORIES.items():
        path = RAW_DIR / filename
        raw_hashes[filename] = require_hash(path, reference["raw_file_hashes"][filename])
        pairs, statistics = read_unique_pairs(path, category, seen_pairs)
        if len(pairs) != EXPECTED_COUNTS[category]:
            raise ValueError(f"{category}: expected {EXPECTED_COUNTS[category]} pairs, got {len(pairs)}")
        source_statistics[category] = statistics

        for row in pairs:
            pair_id = row["pair_id"]
            in_baseline = pair_id in baseline_by_id
            if in_baseline:
                if row != baseline_by_id[pair_id]:
                    raise ValueError(f"Baseline record differs from the source: {pair_id}")
                matched_ids.add(pair_id)
            records.append({**row, "in_baseline": in_baseline})

    if len(records) != 3662 or len({row["pair_id"] for row in records}) != 3662:
        raise ValueError("Expected 3,662 distinct revision IDs")
    if matched_ids != baseline_by_id.keys():
        missing = sorted(baseline_by_id.keys() - matched_ids)
        raise ValueError(f"Missing baseline pairs: {', '.join(missing)}")

    manifest = {
        "source_revision": SOURCE_REVISION,
        "source_base_url": SOURCE_BASE,
        "raw_file_hashes": raw_hashes,
        "category_order": list(CATEGORIES),
        "category_counts": EXPECTED_COUNTS,
        "source_statistics": source_statistics,
        "baseline": {
            "file": baseline_path.relative_to(ROOT).as_posix(),
            "sha256": baseline_hash,
            "count": len(matched_ids),
            "category_counts": dict(baseline_counts),
        },
        "previously_unscored": {
            "count": len(records) - len(matched_ids),
            "category_counts": {
                category: EXPECTED_COUNTS[category] - baseline_counts[category]
                for category in CATEGORIES
            },
        },
        "membership_field": "in_baseline",
    }
    return records, manifest


def write_dataset(output_dir: Path, records: list, manifest: dict) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage both files together so an interrupted build cannot publish half a dataset.
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=".turblimp-build-") as temp:
        staged = Path(temp) / output_dir.name
        staged.mkdir()
        dataset = staged / DATASET_NAME
        write_jsonl(dataset, records)
        manifest["dataset"] = {
            "file": DATASET_NAME,
            "count": len(records),
            "sha256": sha256(dataset),
        }
        (staged / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        if not output_dir.exists():
            try:
                staged.rename(output_dir)
                return
            except OSError:
                if not output_dir.is_dir():
                    raise

        for name in (DATASET_NAME, MANIFEST_NAME):
            existing = output_dir / name
            if not existing.is_file() or sha256(existing) != sha256(staged / name):
                raise ValueError(f"Refusing to overwrite existing output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "revision_v2")
    args = parser.parse_args()

    try:
        records, manifest = build_records()
        write_dataset(args.output_dir.resolve(), records, manifest)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    print(f"Verified {len(records)} pairs in {args.output_dir}")
    print(f"SHA-256: {manifest['dataset']['sha256']}")


if __name__ == "__main__":
    main()
