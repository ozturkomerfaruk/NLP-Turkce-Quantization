import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

# Model seçimini komut satırından al: --model fp16 / q8 / q4
parser = argparse.ArgumentParser(description="TurBLiMP 20 çiftlik pilot değerlendirmesi")
parser.add_argument("--model", required=True, choices=["fp16", "q8", "q4"])
args = parser.parse_args()
MODEL_KEY = args.model
PRECISION = MODEL_KEY.upper()

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / f"eurollm-{MODEL_KEY}"
PILOT_PATH = ROOT / "data" / "processed" / "turblimp_pilot_20.jsonl"

EXPECTED_SHA256 = (
    "420407b129d0e997680a50ab11ff49319e933c1b82b07068042b5d1249239f27"
)

print(f"{PRECISION} model mevcut:", (MODEL_PATH / "config.json").is_file())
print("Pilot veri seti mevcut:", PILOT_PATH.is_file())

if not (MODEL_PATH / "config.json").is_file():
    raise FileNotFoundError(f"Model yapılandırması bulunamadı: {MODEL_PATH / 'config.json'}")
if not PILOT_PATH.is_file():
    raise FileNotFoundError(f"Pilot veri seti bulunamadı: {PILOT_PATH}")


def load_first_pair():
    with PILOT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                return json.loads(line)

    raise ValueError("Pilot veri dosyası boş.")


# ADIM 10 — Yeni puanlama fonksiyonu
def score_sentence(model, tokenizer, sentence):
    token_ids = tokenizer.encode(
        sentence,
        add_special_tokens=False
    )

    if not token_ids:
        raise ValueError("Cümle boş veya tokenize edilemedi.")

    bos_id = tokenizer.bos_token_id

    if bos_id is None:
        raise ValueError("BOS token bulunamadı.")

    sequence = [bos_id] + token_ids

    inputs = mx.array(
        [sequence[:-1]],
        dtype=mx.int32
    )

    targets = mx.array(
        [sequence[1:]],
        dtype=mx.int32
    )

    logits = model(inputs)

    log_probs = nn.log_softmax(
        logits.astype(mx.float32),
        axis=-1
    )

    selected = mx.take_along_axis(
        log_probs,
        targets[..., None],
        axis=-1
    )

    total_score = mx.sum(selected)
    average_score = total_score / len(token_ids)

    mx.eval(total_score, average_score)

    return {
        "token_count": len(token_ids),
        "total": total_score.item(),
        "average": average_score.item()
    }


# ADIM 11 — Pilot dosyasının bütünlüğünü doğrula
print("\n--- VERİ SETİ BÜTÜNLÜK KONTROLÜ ---")

dataset_sha256 = hashlib.sha256(
    PILOT_PATH.read_bytes()
).hexdigest()

print("Beklenen SHA-256:", EXPECTED_SHA256)
print("Hesaplanan SHA-256:", dataset_sha256)

if dataset_sha256 != EXPECTED_SHA256:
    raise ValueError("Pilot veri setinin SHA-256 değeri uyuşmuyor!")

print("SHA-256 doğrulaması başarılı.")


pair = load_first_pair()

print("\nİlk cümle çifti:")
print(pair)
print("Veri tipi:", type(pair))

good = pair["sentence_good"]
bad = pair["sentence_bad"]
pair_id = pair["pair_id"]
category = pair["category"]
critical_region = pair["critical_region"]

print("\n--- İLK DENEY ÇİFTİ ---")
print("Çift ID:", pair_id)
print("Kategori:", category)
print("Kritik bölge:", critical_region)

print("\nDoğru cümle:", good)
print("Hatalı cümle:", bad)

print(f"\n{PRECISION} modeli yükleniyor...")

model, tokenizer = load(str(MODEL_PATH))

model.eval()  # Modeli değerlendirme moduna alır.

print(f"{PRECISION} modeli başarıyla yüklendi.")

token_ids = tokenizer.encode(
    good,
    add_special_tokens=False
)

print("\nDoğru cümle:", good)
print("Token ID'leri:", token_ids)
print("Token sayısı:", len(token_ids))

tokens = tokenizer.convert_ids_to_tokens(token_ids)

print("\n--- TOKEN ANALİZİ ---")

for index, (token_id, token) in enumerate(
    zip(token_ids, tokens), start=1
):
    print(index, token_id, repr(token))

bos_id = tokenizer.bos_token_id

if bos_id is None:
    raise ValueError("Tokenizer BOS token içermiyor.")

print("\nBOS Token:", tokenizer.bos_token)
print("BOS ID:", bos_id)

sequence = [bos_id] + token_ids

print("\nSequence:", sequence)
print("Sequence uzunluğu:", len(sequence))

inputs = sequence[:-1]
targets = sequence[1:]

print("\n--- INPUT / TARGET EŞLEŞMESİ ---")

for index, (input_id, target_id) in enumerate(
    zip(inputs, targets), start=1
):
    input_token = tokenizer.convert_ids_to_tokens(input_id)
    target_token = tokenizer.convert_ids_to_tokens(target_id)

    print(
        f"{index}. {input_token} -> {target_token}"
    )


input_tensor = mx.array(
    [inputs],
    dtype=mx.int32
)

target_tensor = mx.array(
    [targets],
    dtype=mx.int32
)

print("\n--- TENSOR KONTROLÜ ---")

print("Input Shape:", input_tensor.shape)
print("Target Shape:", target_tensor.shape)
print("Input Dtype:", input_tensor.dtype)

print(f"\n{PRECISION} Forward Pass başlıyor...")
logits = model(input_tensor)
mx.eval(logits)

print("\n--- FORWARD PASS SONUCU ---")

print("Logits Shape:", logits.shape)
print("Logits Dtype:", logits.dtype)

print("Batch Size:", logits.shape[0])
print("Sequence Length:", logits.shape[1])
print("Vocabulary Size:", logits.shape[2])

log_probs = nn.log_softmax(
    logits.astype(mx.float32),
    axis=-1
)

print("\nLog Probabilities Shape:", log_probs.shape)
print("Log Probabilities Dtype:", log_probs.dtype)

selected = mx.take_along_axis(
    log_probs,
    target_tensor[..., None],
    axis=-1
)

token_log_probs = mx.squeeze(selected, axis=-1)

total_score = mx.sum(token_log_probs)
average_score = total_score / len(targets)
mx.eval(total_score, average_score)

print(f"\n--- {PRECISION} TOKEN PUANLARI ---")

scores = token_log_probs[0].tolist()

for token, score in zip(tokens, scores):
    print(f"{repr(token):15} {score:.4f}")

print(f"\n--- {PRECISION} CÜMLE PUANI ---")

print("Token sayısı:", len(targets))
print("Toplam log-olasılık:", total_score.item())
print("Ortalama log-olasılık:", average_score.item())


bad_token_ids = tokenizer.encode(
    bad,
    add_special_tokens=False
)

print("\n--- HATALI CÜMLE ---")
print("Cümle:", bad)
print("Token ID'leri:", bad_token_ids)
print("Token sayısı:", len(bad_token_ids))

bad_sequence = [bos_id] + bad_token_ids

bad_inputs = mx.array(
    [bad_sequence[:-1]],
    dtype=mx.int32
)

bad_targets = mx.array(
    [bad_sequence[1:]],
    dtype=mx.int32
)

bad_logits = model(bad_inputs)

bad_log_probs = nn.log_softmax(
    bad_logits.astype(mx.float32),
    axis=-1
)

bad_selected = mx.take_along_axis(
    bad_log_probs,
    bad_targets[..., None],
    axis=-1
)

bad_token_scores = mx.squeeze(
    bad_selected,
    axis=-1
)

bad_total_score = mx.sum(bad_token_scores)

bad_average_score = (
    bad_total_score / len(bad_token_ids)
)

mx.eval(bad_total_score, bad_average_score)


good_score = average_score.item()
bad_score = bad_average_score.item()

difference = good_score - bad_score

print(f"\n--- {PRECISION} KARŞILAŞTIRMA ---")

print(f"Doğru cümle: {good_score:.6f}")
print(f"Hatalı cümle: {bad_score:.6f}")
print(f"Puan farkı: {difference:.6f}")

if difference > 0:
    print("Karar: DOĞRU")
elif difference < 0:
    print("Karar: HATALI")
else:
    print("Karar: EŞİTLİK")


bad_tokens = tokenizer.convert_ids_to_tokens(
    bad_token_ids
)

good_scores = token_log_probs[0].tolist()
bad_scores = bad_token_scores[0].tolist()

print("\n--- TOKEN BAZLI KARŞILAŞTIRMA ---")

# Önceki konumsal tabloyu koruyoruz. Token sayıları farklıysa
# zip ile yanlış hizalama yapmıyoruz. Sayılar eşit olsa bile
# bu tablo dilbilimsel eşleştirme değil, yalnızca pozisyon analizidir.
if len(tokens) == len(bad_tokens):
    print("Not: Karşılaştırma yalnızca aynı token konumları arasındadır;")
    print("farklı token sınırları dilbilimsel olarak hizalanmış sayılmaz.")
    print(
        f"{'No':<4}"
        f"{'Doğru Token':<15}"
        f"{'Hatalı Token':<15}"
        f"{'Good':>10}"
        f"{'Bad':>10}"
        f"{'Fark':>10}"
    )

    for i, (good_token, bad_token, good_log, bad_log) in enumerate(
        zip(tokens, bad_tokens, good_scores, bad_scores),
        start=1
    ):
        token_difference = good_log - bad_log

        print(
            f"{i:<4}"
            f"{good_token:<15}"
            f"{bad_token:<15}"
            f"{good_log:>10.4f}"
            f"{bad_log:>10.4f}"
            f"{token_difference:>+10.4f}"
        )
else:
    print("Token sayıları farklı: konumsal karşılaştırma atlandı.")

print("\n--- TOKEN BAZLI BAĞIMSIZ ANALİZ ---")

print("\nDOĞRU CÜMLE TOKENLARI")
print(f"{'No':<5}{'Token':<20}{'Log P':>12}")

for i, (token, score) in enumerate(
    zip(tokens, good_scores), start=1
):
    print(f"{i:<5}{token:<20}{score:>12.4f}")

print("\nHATALI CÜMLE TOKENLARI")
print(f"{'No':<5}{'Token':<20}{'Log P':>12}")

for i, (token, score) in enumerate(
    zip(bad_tokens, bad_scores), start=1
):
    print(f"{i:<5}{token:<20}{score:>12.4f}")

print("\n--- TOKEN SAYISI KONTROLÜ ---")
print("Doğru cümle token sayısı:", len(tokens))
print("Hatalı cümle token sayısı:", len(bad_tokens))

if len(tokens) != len(bad_tokens):
    print(
        "Token sayıları farklı. "
        "Konumsal token karşılaştırması yapılmıyor."
    )


# ADIM 10 — Fonksiyon doğrulama testi

print("\n--- FONKSİYON TESTİ ---")

good_result = score_sentence(model, tokenizer, good)
bad_result = score_sentence(model, tokenizer, bad)

print("Doğru cümle:", good_result)
print("Hatalı cümle:", bad_result)

assert abs(good_result["average"] - good_score) < 1e-4
assert abs(bad_result["average"] - bad_score) < 1e-4

print("Fonksiyon doğrulaması başarılı.")

print(f"\n--- {PRECISION}: 20 ÇİFTLİK PİLOT DEĞERLENDİRME ---")

with PILOT_PATH.open("r", encoding="utf-8") as file:
    pilot_pairs = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

if len(pilot_pairs) != 20:
    raise ValueError(
        f"20 çift bekleniyordu, {len(pilot_pairs)} bulundu."
    )

pilot_results = []

for index, current_pair in enumerate(pilot_pairs, start=1):

    good_result = score_sentence(
        model,
        tokenizer,
        current_pair["sentence_good"]
    )

    bad_result = score_sentence(
        model,
        tokenizer,
        current_pair["sentence_bad"]
    )

    delta = good_result["average"] - bad_result["average"]

    correct = delta > 0

    result = {
        "pair_id": current_pair["pair_id"],
        "category": current_pair["category"],
        "model": PRECISION,
        "dataset_sha256": dataset_sha256,
        "scoring_method": "mean_token_log_probability",
        "good_score": good_result["average"],
        "good_total_score": good_result["total"],
        "bad_score": bad_result["average"],
        "bad_total_score": bad_result["total"],
        "good_token_count": good_result["token_count"],
        "bad_token_count": bad_result["token_count"],
        "delta": delta,
        "correct": correct
    }

    pilot_results.append(result)

    print(
        f"{index:02d}/20 | "
        f"{current_pair['pair_id']} | "
        f"Fark: {delta:+.6f} | "
        f"{'DOĞRU' if correct else 'HATALI'}"
    )

success_count = sum(
    result["correct"]
    for result in pilot_results
)

tie_count = sum(
    result["delta"] == 0
    for result in pilot_results
)

accuracy = success_count / len(pilot_results)

print(f"\n--- {PRECISION} PİLOT ÖZETİ ---")
print("Toplam çift:", len(pilot_results))
print("Başarılı çift:", success_count)
print("Başarısız çift:", len(pilot_results) - success_count)
print("Eşit puan:", tie_count)
print(f"Başarı oranı: %{accuracy * 100:.2f}")

category_results = {}
for result in pilot_results:
    key = result["category"]
    if key not in category_results:
        category_results[key] = {"total": 0, "correct": 0}
    category_results[key]["total"] += 1
    category_results[key]["correct"] += int(result["correct"])

print("\n--- KATEGORİ BAZLI SONUÇLAR ---")
for category_name, values in category_results.items():
    category_accuracy = 100 * values["correct"] / values["total"]
    print(f"{category_name}: {values['correct']}/{values['total']} (%{category_accuracy:.2f})")

RESULTS_DIR = ROOT / "results" / "pilot"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH = RESULTS_DIR / f"{MODEL_KEY}_pilot_20.jsonl"

with RESULTS_PATH.open("w", encoding="utf-8") as file:
    for result in pilot_results:
        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

print("\n--- SONUÇLAR KAYDEDİLDİ ---")
print("Dosya:", RESULTS_PATH)
print("Kaydedilen kayıt sayısı:", len(pilot_results))

SUMMARY_PATH = RESULTS_DIR / f"{MODEL_KEY}_summary.json"

summary = {
    "model": "EuroLLM-1.7B Base",
    "precision": PRECISION,
    "model_path": str(MODEL_PATH),
    "dataset_path": str(PILOT_PATH),
    "dataset_sha256": dataset_sha256,
    "scoring_method": "mean_token_log_probability",
    "scoring_dtype": "float32",
    "tokenization": {
        "add_special_tokens": False,
        "prepend_bos": True,
        "score_bos": False,
        "append_eos": False
    },
    "environment": {
        "python_version": platform.python_version(),
        "mlx_version": version("mlx"),
        "mlx_lm_version": version("mlx-lm"),
        "platform": platform.platform()
    },
    "total_pairs": len(pilot_results),
    "successful_pairs": success_count,
    "unsuccessful_pairs": len(pilot_results) - success_count,
    "tie_count": tie_count,
    "accuracy": accuracy,
    "category_results": category_results
}

with SUMMARY_PATH.open("w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        ensure_ascii=False,
        indent=4
    )

print("Özet dosyası:", SUMMARY_PATH)
