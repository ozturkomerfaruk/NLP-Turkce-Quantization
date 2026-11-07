"""Verify the frozen main benchmark and derive its reproducible 20-pair pilot.

Select five pairs per category and write a manifest tying both datasets to their
source revision, sampling seeds and SHA-256 identities. Never alter the main set."""


import json
import random
from collections import Counter

from prepare_dataset import (
    CATEGORIES,
    OUT_DIR,
    ROOT,
    SEED,
    SOURCE_REVISION,
    sha256,
    write_jsonl,
)

# Keep pilot sampling reproducible and balanced across the four categories.
PILOT_SEED = 42
PILOT_PER_CATEGORY = 5

EXPECTED_HASH = (
    "bcb2c7f25475f8290d1511d742e1e9f8f515fd29df11dd3e3f64f2f53bcb4835"
)

# Derive the pilot and its provenance manifest from the frozen main dataset.
INPUT_PATH = OUT_DIR / "turblimp_1000.jsonl"
PILOT_PATH = OUT_DIR / "turblimp_pilot_20.jsonl"
MANIFEST_PATH = OUT_DIR / "manifest.json"


# Ignore blank lines when reading one JSON object per sentence pair.
# Load the frozen records without reordering or modifying their sentence content.
def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    # Refuse to sample from modified data: all models must see the same benchmark.
    actual_hash = sha256(INPUT_PATH)

    if actual_hash != EXPECTED_HASH:
        raise ValueError("The main dataset SHA-256 has changed!")

    # Confirm the benchmark size before selecting the pilot subset.
    records = load_jsonl(INPUT_PATH)

    if len(records) != 1000:
        raise ValueError("The main dataset must contain 1000 pairs.")

    # Unique IDs keep later model comparisons aligned to the same pairs.
    pair_ids = [record["pair_id"] for record in records]

    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Duplicate pair_id found.")

    # Use a private seeded generator so unrelated randomness cannot affect selection.
    rng = random.Random(PILOT_SEED)
    pilot_records = []

    # Verify category membership before applying balanced sampling.
    counts = Counter(record["category"] for record in records)

    if set(counts) != set(CATEGORIES):
        raise ValueError("Unexpected category found.")

    # Select five pairs per category from the main set; this is not a held-out test set.
    for category in CATEGORIES:
        category_records = [
            record for record in records
            if record["category"] == category
        ]

        if len(category_records) != 250:
            raise ValueError(f"{category}: Expected 250 pairs.")

        selected = rng.sample(
            category_records,
            PILOT_PER_CATEGORY
        )

        pilot_records.extend(selected)

        print(f"{category}: {len(selected)} pairs selected.")

    write_jsonl(PILOT_PATH, pilot_records)

    raw_hashes = {}

    for filename in CATEGORIES.values():
        raw_path = ROOT / "data" / "raw" / filename
        raw_hashes[filename] = sha256(raw_path)

    # Record seeds, source revision and file hashes so the sample can be audited.
    manifest = {
        "source_revision": SOURCE_REVISION,
        "dataset_seed": SEED,
        "pilot_seed": PILOT_SEED,
        "pilot_per_category": PILOT_PER_CATEGORY,
        "category_order": list(CATEGORIES),
        "category_counts": dict(counts),
        "main_dataset": {
            "file": str(INPUT_PATH.relative_to(ROOT)),
            "count": len(records),
            "sha256": actual_hash,
        },
        "pilot_dataset": {
            "file": str(PILOT_PATH.relative_to(ROOT)),
            "count": len(pilot_records),
            "sha256": sha256(PILOT_PATH),
        },
        "raw_file_hashes": raw_hashes,
    }
    
    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        # Report the pilot identity that subsequent scoring runs must verify.
        print("\nPilot dataset created.")
        print("Total pairs:", len(pilot_records))
        print("Pilot SHA-256:", sha256(PILOT_PATH))
        print("Manifest:", MANIFEST_PATH)


# Run dataset finalization only when invoked as a script.
if __name__ == "__main__":
    main()
