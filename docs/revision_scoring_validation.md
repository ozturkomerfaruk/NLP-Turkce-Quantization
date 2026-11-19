# Revizyon puanlamasının doğrulanması

[Revizyon puanlama betiği](../scripts/score_revision.py), [mevcut cümle puanlama işlevini](../scripts/score_main.py) kullanır. İlk deneyin komut arayüzü ve kayıt yolları korunur.

## Küçük testi yeniden çalıştırma

Depo kökünde:

```bash
for family in eurollm qwen; do
    for precision in fp16 q8 q4; do
        .venv/bin/python scripts/score_revision.py \
            --model "$family" --precision "$precision" --mode smoke --max-chunks 2 || exit 1
    done
done
```

Küçük testte dört kategorinin her birinden, ilk deneyde puanlanmış ilk iki çift alınır. Altı yapılandırmada aynı sekiz çift kullanılır. Teknik testin skorları `outputs/revision_v2/scoring_smoke/` altında saklanır. Her yapılandırma ayrı bir süreçte çalışır. Ana deney `--mode main` ile açıkça seçilir ve skorları `results/revision_v2/<model>/<precision>/` altında 50 çiftlik parçalar halinde saklar.

## Veri, tokenizer ve kayıt kontrolleri

- Revizyon verisinin sabit SHA-256 değeri ve model dosyalarının ilgili manifestteki SHA-256 değerleri doğrulanır.
- Her model ailesinin FP16 tokenizer’ı referans alınır. Her iki cümlenin token sayısı ve token dizisinin SHA-256 değeri hassasiyet düzeyleri arasında eşleştirilir. EuroLLM ve Qwen kendi tokenizer’larıyla değerlendirilir.
- EuroLLM tokenizer BOS kimliğini kullanır. Qwen model yapılandırmasındaki `151643` kimliği Qwen’in başlangıç bağlamıdır. Cümlenin kendi token’ları hedef alınır. Sona EOS eklenmez.
- Mean ve Total aynı ileri geçişten kaydedilir. Sonlu ve geçerli skorlar, token başına normalizasyon, her ölçüt için fark ve doğru karar ile beraberlik işaretleri, kayıt yazılmadan ve devam edilirken kontrol edilir.
- Her JSON kontrol noktası skorları ve parça ölçümlerini tek işlemle kaydeder. Veri, model, FP16 referansı, çalışma modu, ortam ve kod kimlikleri eşleştiğinde tamamlanan parçalar atlanır. Aynı yapılandırma üzerinde eşzamanlı iki yazıcı dosya kilidiyle engellenir.
- Ana mod, aynı kod ve çalışma ortamıyla oluşturulmuş küçük test kayıtlarını doğrular. Puanlama kodu değişirse yeni küçük test gerekir.

## Doğrulanan puanlama kodu

| Dosya | SHA-256 |
| --- | --- |
| `scripts/score_main.py` | `7cd3a0dc547acfe16da1092556df46909a8caa2dc4302aae6e6894112e0bfc59` |
| `scripts/score_revision.py` | `57528813bfe592d0d281ba27fb6db5d0e20a810c5858fdac7d4841fa40e5cedd` |

18 Kasım 2026’da sekiz proje testi geçti. Altı yapılandırmada sekizer çift ve ikişer kontrol noktası kaydedildi. EuroLLM FP16 ilk parçanın ardından durdurulup yeni bir süreçte ikinci parçadan devam etti. Üçüncü çalıştırma iki parçayı doğrulayıp model yüklemeden tamamlandı.
