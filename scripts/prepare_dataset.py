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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream: # Dosyayı ikili modda aç ve oku rb → read binary.
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def fetch_csv(filename: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    path = RAW_DIR / filename

    if path.exists():
        print(f"Dosya zaten mevcut: {filename}")
        return path

    url = SOURCE_BASE + filename

    print(f"İndiriliyor: {filename}")

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


def preview_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        print("CSV sütunları:", reader.fieldnames)

        for index, row in enumerate(reader, start=1):
            print(f"\nCümle çifti {index}")
            print("Doğru:", row["good_sentence"])
            print("Hatalı:", row["bad_sentence"])

            if index == 3:
                break



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
            raise ValueError(f"Beklenmeyen CSV sütunları: {path}")

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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    global_seen = set()

    selected_by_category = {}
    source_statistics = {}
    raw_hashes = {}

    for category, filename in CATEGORIES.items():
        print(f"\nİşlenen kategori: {category}")

        path = fetch_csv(filename)

        raw_hashes[filename] = sha256(path)

        pairs, stats = read_unique_pairs(
            path,
            category,
            global_seen
        )

        if len(pairs) < SAMPLE_PER_CATEGORY:
            raise ValueError(
                f"{category} kategorisinde yeterli çift yok!"
            )

        selected = rng.sample(
            pairs,
            SAMPLE_PER_CATEGORY
        )

        selected_by_category[category] = selected
        source_statistics[category] = stats

        print("Orijinal çift:", stats["original_rows"])
        print("Benzersiz çift:", stats["unique_available"])
        print("Seçilen çift:", len(selected))

    total = sum(
        len(pairs)
        for pairs in selected_by_category.values()
    )

    print("\nToplam seçilen çift:", total)

    for category, pairs in selected_by_category.items():
        print(f"\nKategori: {category}")
        print("Örnek doğru:", pairs[0]["sentence_good"])
        print("Örnek hatalı:", pairs[0]["sentence_bad"])
        
    all_selected = []

    for category in CATEGORIES:
        all_selected.extend(
            selected_by_category[category]
        )

    output_path = OUT_DIR / "turblimp_1000.jsonl"

    write_jsonl(output_path, all_selected)

    print("\n--- Veri Seti Kaydedildi ---")
    print("Dosya:", output_path)
    print("Toplam kayıt:", len(all_selected))
    print("SHA-256:", sha256(output_path))



if __name__ == "__main__":
    main()
