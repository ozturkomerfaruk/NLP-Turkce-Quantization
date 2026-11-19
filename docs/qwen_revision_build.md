# Qwen3-1.7B-Base model hazırlığı

İkinci model [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base/tree/ea980cb0a6c2ae4b936e82123acc929f1cec04c1) ve kaynak commit’i `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. Kaynak ağırlık indirilince Hub’da yayımlanan SHA-256 değeriyle doğrulanır.

## Dönüşüm ve kayıt

Depo kökünde, `requirements.txt` sürümlerini kullanan MLX ortamında:

```bash
mkdir -p outputs/revision_v2/qwen
(set -o pipefail; bash scripts/build_qwen_revision.sh 2>&1 | tee -a outputs/revision_v2/qwen/build.log)
```

[Hazırlama betiği](../scripts/build_qwen_revision.sh) model kaynağını indirir, FP16 referansını `--dtype float16` ile oluşturur ve o referanstan ayrı ayrı Q8 ve Q4 üretir. İki kuantize sürüm EuroLLM ile aynı `--q-group-size 64 --q-mode affine` ayarlarını kullanır. Kaynak ve dönüştürülmüş modeller `models/revision_v2/qwen/` altında tutulur. [İlk dönüşüm kaydı](qwen_revision_conversion_trace.txt), çalıştırılan komutları ve o sıradaki model yollarını arşivler. [Doğrulama manifesti](qwen_revision_manifest.json) güncel model yollarını ve model ile tokenizer dosyalarının SHA-256 değerlerini içerir.

## Başlangıç bağlamı ve skor

Sabit kaynak tokenizer’ında `bos_token` değeri `null`. Model yapılandırmasındaki `bos_token_id=151643`, tokenizer sözlüğünde `<|endoftext|>` kimliğine karşılık gelir. Qwen için bu kimlik yalnızca başlangıç bağlamı olarak kullanılır. Cümle kendi tokenizer’ıyla, özel token eklenmeden kodlanır. Model girişindeki ilk konum cümlenin ilk token’ını tahmin eder. Mean ve Total hesabına yalnız cümlenin kendi token’ları girer ve sona EOS eklenmez. EuroLLM mevcut tokenizer BOS kuralıyla devam eder.

[Doğrulama betiği](../scripts/verify_qwen_revision.py), ana veri kümesindeki cümlelerle örtüşmeyen üç yapay minimal çift kullanır. FP16, Q8 ve Q4 token dizilerini sabit kaynak tokenizer’ıyla karşılaştırır. Aynı modelden alınan konum başına logitlerle Mean ve Total skorunu ikinci bir hesap yoluyla kontrol eder. Bu örnekler yalnızca teknik doğrulama içindir.

## Yerel doğrulama sonucu

Kaynak ağırlığın SHA-256 değeri Hugging Face kaydıyla eşleşti. Üç Qwen yapılandırması Apple Silicon ve MLX-LM 0.32.2 ortamında yüklendi. FP16, Q8 ve Q4, kaynak tokenizer’ıyla aynı token dizilerini üretti. Her yapılandırmada üç yapay minimal çiftin Mean ve Total skorları ikinci hesapla `0,00005` mutlak tolerans içinde eşleşti.

Ana revizyon deneyinde model aileleri, kendi FP16 referanslarına göre Q8 ve Q4 doğruluk değişimleri üzerinden karşılaştırıldı.
