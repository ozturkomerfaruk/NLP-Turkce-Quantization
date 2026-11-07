"""Build the fixed, balanced TurBLiMP benchmark from four pinned source CSVs.

Reuse local raw files when available, remove invalid and duplicate pairs, then
sample 250 pairs per category with seed 42. Preserve the Turkish sentences and
source metadata: these are experimental inputs, not interface text."""

import csv
import hashlib
import json
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


SEED = 42
SAMPLE_PER_CATEGORY = 250

SOURCE_REVISION = "297de13fb7a0ce524fe32e8b175c6b5255d66960"

SOURCE_BASE = (
    "https://raw.githubusercontent.com/ezgibasar/TurBLiMP/"
    f"{SOURCE_REVISION}/data/base/"
)

CATEGORIES = {
    "subject_agreement": "augmented_subject_verb_agreement.csv",
    "argument_structure_transitive": "augmented_argument_structure_transitive.csv",
    "nominalization": "augmented_nominalization.csv",
    "irregular_forms": "augmented_irregular_forms.csv",
}


# Hash file bytes in bounded blocks to identify the exact data used in an experiment.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


# Reuse cached sources; download via a temporary file to avoid accepting partial CSVs.
def fetch_csv(filename: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    path = RAW_DIR / filename

    if path.exists():
        print(f"File already exists: {filename}")
        return path

    url = SOURCE_BASE + filename

    print(f"Downloading: {filename}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "turblimp-research/1.0"}
    )

    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as output:
            while chunk := response.read(128 * 1024):
                output.write(chunk)

        tmp.replace(path)

    finally:
        tmp.unlink(missing_ok=True)

    return path


# Inspect source columns and a few labelled pairs without changing the dataset.
def preview_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        print("CSV columns:", reader.fieldnames)

        for index, row in enumerate(reader, start=1):
            print(f"\nSentence pair {index}")
            print("Good:", row["good_sentence"])
            print("Bad:", row["bad_sentence"])

            if index == 3:
                break



# Clean each source and remove exact duplicates within and across categories.
# Preserve source row IDs and linguistic metadata for later error analysis.
def read_unique_pairs(
    path: Path,
    category: str,
    already_seen: set
) -> tuple[list, dict]:

    result = []
    seen_here = set()

    stats = {
        "original_rows": 0,
        "invalid_rows": 0,
        "duplicates_within_category": 0,
        "duplicates_across_categories": 0,
    }

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")

        required_columns = {"good_sentence", "bad_sentence"}

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected CSV columns: {path}")

        for source_index, record in enumerate(reader, start=1):
            stats["original_rows"] += 1

            good = (record["good_sentence"] or "").strip()
            bad = (record["bad_sentence"] or "").strip()

            if not good or not bad or good == bad:
                stats["invalid_rows"] += 1
                continue

            key = (good, bad)

            if key in seen_here:
                stats["duplicates_within_category"] += 1
                continue

            seen_here.add(key)

            if key in already_seen:
                stats["duplicates_across_categories"] += 1
                continue

            already_seen.add(key)

            result.append({
                "pair_id": f"{category}:{source_index:04d}",
                "category": category,
                "sentence_good": good,
                "sentence_bad": bad,
                "source_record_index": source_index,
                "source_file": str(path.relative_to(ROOT)),
                "good_cue": (record.get("good_cue") or "").strip(),
                "bad_cue": (record.get("bad_cue") or "").strip(),
                "critical_region": (record.get("critical_region") or "").strip(),
                "phenomenon": (record.get("phenomenon") or "").strip(),
                "subtype": (record.get("subtype") or "").strip(),
            })

    stats["unique_available"] = len(result)

    return result, stats


# Use deterministic serialization and replace the target only after a complete write.
def write_jsonl(path: Path, rows: list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for record in rows:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True
                ) + "\n"
            )

    tmp.replace(path)


# Sample in the fixed category order with one seeded generator, then save the benchmark.
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    global_seen = set()

    selected_by_category = {}
    source_statistics = {}
    raw_hashes = {}

    for category, filename in CATEGORIES.items():
        print(f"\nProcessing category: {category}")

        path = fetch_csv(filename)

        raw_hashes[filename] = sha256(path)

        pairs, stats = read_unique_pairs(
            path,
            category,
            global_seen
        )

        if len(pairs) < SAMPLE_PER_CATEGORY:
            raise ValueError(
                f"{category} does not contain enough pairs!"
            )

        selected = rng.sample(
            pairs,
            SAMPLE_PER_CATEGORY
        )

        selected_by_category[category] = selected
        source_statistics[category] = stats

        print("Original pairs:", stats["original_rows"])
        print("Unique pairs:", stats["unique_available"])
        print("Selected pairs:", len(selected))

    total = sum(
        len(pairs)
        for pairs in selected_by_category.values()
    )

    print("\nTotal selected pairs:", total)

    for category, pairs in selected_by_category.items():
        print(f"\nCategory: {category}")
        print("Example good sentence:", pairs[0]["sentence_good"])
        print("Example bad sentence:", pairs[0]["sentence_bad"])
        
    all_selected = []

    for category in CATEGORIES:
        all_selected.extend(
            selected_by_category[category]
        )

    output_path = OUT_DIR / "turblimp_1000.jsonl"

    write_jsonl(output_path, all_selected)

    print("\n--- DATASET SAVED ---")
    print("File:", output_path)
    print("Total records:", len(all_selected))
    print("SHA-256:", sha256(output_path))



if __name__ == "__main__":
    main()
