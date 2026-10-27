
"""FP16, Q8 ve Q4 TurBLiMP pilot sonuçlarını doğrular ve karşılaştırır.

Modelleri yüklemez; yalnızca yerel tokenizer'ları ve kayıtlı JSONL/JSON
sonuçlarını okur. Ana veri setine veya mevcut pilot sonuçlarına dokunmaz.
"""

import hashlib
import json
import math
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
PILOT_PATH = ROOT / "data" / "processed" / "turblimp_pilot_20.jsonl"
RESULTS_DIR = ROOT / "results" / "pilot"

MODELS = ("fp16", "q8", "q4")

EXPECTED_SHA256 = (
    "420407b129d0e997680a50ab11ff49319e933c1b82b07068042b5d1249239f27"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_results(
    model_key: str,
    expected_pairs: list[dict]
) -> dict[str, dict]:
    """20 satırı, kayıtların kimliklerini, puanları ve özet dosyasını doğrular."""

    results_path = RESULTS_DIR / f"{model_key}_pilot_20.jsonl"
    summary_path = RESULTS_DIR / f"{model_key}_summary.json"

    rows = read_jsonl(results_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    expected_ids = [pair["pair_id"] for pair in expected_pairs]
    observed_ids = [row["pair_id"] for row in rows]

    assert_condition(
        len(rows) == 20,
        f"{model_key}: 20 sonuç bekleniyor."
    )

    assert_condition(
        observed_ids == expected_ids,
        f"{model_key}: Çift ID'leri veya sıraları farklı."
    )

    assert_condition(
        len(set(observed_ids)) == 20,
        f"{model_key}: Tekrar eden çift ID'si var."
    )

    successes = 0
    ties = 0

    for pair, row in zip(expected_pairs, rows):
        assert_condition(
            row["category"] == pair["category"],
            f"{model_key}: kategori hatası."
        )

        assert_condition(
            row["model"].lower() == model_key,
            f"{model_key}: model etiketi yanlış."
        )

        assert_condition(
            row["dataset_sha256"] == EXPECTED_SHA256,
            f"{model_key}: veri hash'i farklı."
        )

        assert_condition(
            row["scoring_method"] == "mean_token_log_probability",
            f"{model_key}: yöntem farklı."
        )

        good = float(row["good_score"])
        bad = float(row["bad_score"])
        delta = float(row["delta"])

        assert_condition(
            all(map(math.isfinite, (good, bad, delta))),
            f"{model_key}: geçersiz sayı."
        )

        assert_condition(
            math.isclose(delta, good - bad, abs_tol=1e-5),
            f"{model_key}: puan farkı yanlış."
        )

        assert_condition(
            type(row["correct"]) is bool
            and row["correct"] == (delta > 0),
            f"{model_key}: doğru/yanlış kararı hatalı."
        )

        for prefix, average in (("good", good), ("bad", bad)):
            count = row[f"{prefix}_token_count"]

            assert_condition(
                type(count) is int and count > 0,
                f"{model_key}: token sayısı hatalı."
            )

            total_field = f"{prefix}_total_score"

            if total_field in row:
                assert_condition(
                    math.isclose(
                        float(row[total_field]),
                        average * count,
                        abs_tol=0.001
                    ),
                    f"{model_key}: toplam/ortalama puan tutarsız."
                )

        successes += int(row["correct"])
        ties += int(delta == 0)

    assert_condition(
        summary["precision"].lower() == model_key,
        f"{model_key}: özet precision yanlış."
    )

    assert_condition(
        summary["dataset_sha256"] == EXPECTED_SHA256,
        f"{model_key}: özet hash'i farklı."
    )

    assert_condition(
        summary["total_pairs"] == 20,
        f"{model_key}: özet çift sayısı yanlış."
    )

    assert_condition(
        summary["successful_pairs"] == successes,
        f"{model_key}: özet doğru sayısı yanlış."
    )

    assert_condition(
        summary["unsuccessful_pairs"] == 20 - successes,
        f"{model_key}: özet yanlış sayısı yanlış."
    )

    assert_condition(
        summary["tie_count"] == ties,
        f"{model_key}: özet eşitlik sayısı yanlış."
    )

    assert_condition(
        math.isclose(
            float(summary["accuracy"]),
            successes / 20,
            abs_tol=1e-12
        ),
        f"{model_key}: özet doğruluk yanlış."
    )

    print(
        f"{model_key.upper():4} | "
        f"{successes:2}/20 doğru | "
        f"%{100 * successes / 20:.1f} | doğrulandı"
    )

    return {row["pair_id"]: row for row in rows}


def verify_tokenizers(
    pairs: list[dict],
    results: dict[str, dict[str, dict]]
) -> None:
    """Model ağırlıklarını yüklemeden 40 cümlenin token ID'lerini karşılaştırır."""

    tokenizers = {
        key: AutoTokenizer.from_pretrained(
            str(ROOT / "models" / f"eurollm-{key}"),
            local_files_only=True
        )
        for key in MODELS
    }

    for pair in pairs:
        for field, prefix in (
            ("sentence_good", "good"),
            ("sentence_bad", "bad")
        ):
            sentence = pair[field]

            baseline = tokenizers["fp16"].encode(
                sentence,
                add_special_tokens=False
            )

            for key in MODELS:
                tok = tokenizers[key]

                ids = tok.encode(
                    sentence,
                    add_special_tokens=False
                )

                assert_condition(
                    tok.bos_token_id == tokenizers["fp16"].bos_token_id,
                    f"{key}: BOS token farklı."
                )

                assert_condition(
                    ids == baseline,
                    f"{key}: {pair['pair_id']} için token ID'leri farklı."
                )

                assert_condition(
                    len(ids)
                    == results[key][pair["pair_id"]][f"{prefix}_token_count"],
                    f"{key}: {pair['pair_id']} token sayısı sonuçla farklı."
                )

    print("Tokenizer: BOS ve 40 cümlenin token ID'leri üç modelde aynı.")


def main() -> None:
    actual_hash = hashlib.sha256(
        PILOT_PATH.read_bytes()
    ).hexdigest()

    assert_condition(
        actual_hash == EXPECTED_SHA256,
        "Pilot veri setinin SHA-256 değeri farklı!"
    )

    pairs = read_jsonl(PILOT_PATH)

    assert_condition(
        len(pairs) == 20,
        "Pilot veri setinde 20 çift bulunmalı."
    )

    assert_condition(
        len({p["pair_id"] for p in pairs}) == 20,
        "Pilot veri setinde tekrar var."
    )

    print("\n--- DOSYA VE SONUÇ DOĞRULAMASI ---")
    print("Pilot SHA-256 doğrulandı. 20 çift mevcut.")

    results = {
        key: validate_results(key, pairs)
        for key in MODELS
    }

    verify_tokenizers(pairs, results)

    print("\n--- KATEGORİ BAZLI DOĞRULUK ---")

    categories = list(
        dict.fromkeys(pair["category"] for pair in pairs)
    )

    by_category = {}

    for category in categories:
        ids = [
            p["pair_id"]
            for p in pairs
            if p["category"] == category
        ]

        by_category[category] = {
            key: {
                "correct": sum(
                    results[key][pid]["correct"]
                    for pid in ids
                ),
                "total": len(ids),
            }
            for key in MODELS
        }

        summary = " | ".join(
            f"{key.upper()}: "
            f"{by_category[category][key]['correct']}/{len(ids)}"
            for key in MODELS
        )

        print(f"{category}: {summary}")

    print("\n--- FP16'YA GÖRE KARAR DEĞİŞİMLERİ ---")

    changed = {}

    for key in ("q8", "q4"):
        transitions = []

        for pair in pairs:
            pid = pair["pair_id"]

            before = results["fp16"][pid]
            after = results[key][pid]

            if before["correct"] != after["correct"]:
                transitions.append({
                    "pair_id": pid,
                    "category": pair["category"],
                    "fp16_correct": before["correct"],
                    "quantized_correct": after["correct"],
                    "fp16_delta": before["delta"],
                    "quantized_delta": after["delta"],
                })

                print(
                    f"{key.upper()} | {pid} | "
                    f"FP16: {before['delta']:+.6f} "
                    f"-> {key.upper()}: {after['delta']:+.6f}"
                )

        changed[key] = transitions

        if not transitions:
            print(f"{key.upper()}: Karar değişimi yok.")

    output = {
        "dataset_sha256": actual_hash,
        "total_pairs": 20,
        "accuracy": {
            key: sum(
                row["correct"]
                for row in results[key].values()
            ) / 20
            for key in MODELS
        },
        "categories": by_category,
        "decision_changes_from_fp16": changed,
        "token_ids_equal_across_models": True,
    }

    output_path = RESULTS_DIR / "pilot_comparison.json"

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )

    print(f"\nKarşılaştırma kaydedildi: {output_path}")
    print("PİLOT KARŞILAŞTIRMA KONTROLLERİ BAŞARILI.")


if __name__ == "__main__":
    main()
