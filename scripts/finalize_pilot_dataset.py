
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

# Sabitler
PILOT_SEED = 42
PILOT_PER_CATEGORY = 5

EXPECTED_HASH = (
    "bcb2c7f25475f8290d1511d742e1e9f8f515fd29df11dd3e3f64f2f53bcb4835"
)

# Dosya yolları
INPUT_PATH = OUT_DIR / "turblimp_1000.jsonl"
PILOT_PATH = OUT_DIR / "turblimp_pilot_20.jsonl"
MANIFEST_PATH = OUT_DIR / "manifest.json"


# JSONL dosyasını okuma
def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    # Ana veri setinin SHA-256 doğrulaması
    actual_hash = sha256(INPUT_PATH)

    if actual_hash != EXPECTED_HASH:
        raise ValueError("Ana veri setinin SHA-256 değeri değişmiş!")

    # Ana veri setini yükleme ve kayıt sayısını doğrulama
    records = load_jsonl(INPUT_PATH)

    if len(records) != 1000:
        raise ValueError("Ana veri setinde 1000 çift bulunmalı.")

    # Tekrarlanan pair_id kontrolü
    pair_ids = [record["pair_id"] for record in records]

    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Tekrarlanan pair_id bulundu.")

    # Rastgele örnekleme hazırlığı
    rng = random.Random(PILOT_SEED)
    pilot_records = []

    # Kategori sayılarını hesaplama ve doğrulama
    counts = Counter(record["category"] for record in records)

    if set(counts) != set(CATEGORIES):
        raise ValueError("Beklenmeyen kategori bulundu.")

    # Her kategoriden rastgele 5 çift seçme
    for category in CATEGORIES:
        category_records = [
            record for record in records
            if record["category"] == category
        ]

        if len(category_records) != 250:
            raise ValueError(f"{category}: 250 çift bekleniyor.")

        selected = rng.sample(
            category_records,
            PILOT_PER_CATEGORY
        )

        pilot_records.extend(selected)

        print(f"{category}: {len(selected)} çift seçildi.")

    write_jsonl(PILOT_PATH, pilot_records)

    raw_hashes = {}

    for filename in CATEGORIES.values():
        raw_path = ROOT / "data" / "raw" / filename
        raw_hashes[filename] = sha256(raw_path)

    # Deneyin metadata bilgilerini oluşturma
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
        # Pilot veri setinin oluşturulma sonuçları
        print("\nPilot veri seti oluşturuldu.")
        print("Toplam çift:", len(pilot_records))
        print("Pilot SHA-256:", sha256(PILOT_PATH))
        print("Manifest:", MANIFEST_PATH)


# Programın giriş noktası
if __name__ == "__main__":
    main()
