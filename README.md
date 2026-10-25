# NLP-Turkce-Quantization

EuroLLM-1.7B Base modelinde FP16, Q8 ve Q4 sürümlerinin Türkçe morfosentaktik minimal çiftler üzerindeki karşılaştırılması.

## Mevcut durum

Model ve TurBLiMP veri hazırlığı yerel ortamda tamamlandı. Bu depo veri hazırlama betiğini, proje notlarını, ortam sürümlerini ve veri hash kayıtlarını içerir. Pilot, cümle puanlama, ana deney ve istatistiksel analiz kodları ile araştırma sonuçları henüz yoktur. **Depo şu anda veri hazırlama aşamasını yeniden üretmeye yöneliktir; tüm deneyin yeniden üretimi henüz tamamlanmamıştır.**

## Veri setini yeniden oluşturma

Veri hazırlama betiği yalnızca Python standart kütüphanesini kullanır. Referans ortam Python 3.12.15'tir. İnternet erişimi gerekir.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python scripts/prepare_dataset.py
shasum -a 256 data/processed/turblimp_1000.jsonl
```

Beklenen çıktı: dört kategoriden 250'şer, toplam 1.000 çift. Beklenen SHA-256:

```text
bcb2c7f25475f8290d1511d742e1e9f8f515fd29df11dd3e3f64f2f53bcb4835
```

Kaynak TurBLiMP commit'i `297de13fb7a0ce524fe32e8b175c6b5255d66960`, örnekleme seed'i `42`'dir. Betik mevcut ham dosyaları tekrar kullanır; temiz bir klonda ilgili dosyaları indirir. [Manifest](data/processed/manifest.json), mevcut yerel ham ve işlenmiş dosyaların hash kayıtlarını içerir; betik manifesti otomatik güncellemez. Kaynak veri ve modellerin kullanım/lisans koşulları kendi kaynaklarından kontrol edilmelidir.

## Model ortamı ve kalan işler

Yerel ortam: macOS, Apple Silicon M2, 16 GB RAM; Python 3.12.15, MLX-LM 0.32.2, MLX 0.32.4. `requirements.txt`, mevcut ortamda kurulu paketlerin sabitlenmiş sürüm kaydıdır; temiz ortamda kurulum bu aktarım sırasında test edilmemiştir. Veri hazırlamak için bu paketlerin kurulması gerekmez. Model çalışmaları için uyumlu Apple Silicon ortamında:

```bash
python -m pip install -r requirements.txt
```

Kaynak model `utter-project/EuroLLM-1.7B`; kurulum notlarındaki indirme revizyonu `refs/pr/5`'tir. Bu ad değişebilir: tam model yeniden üretimi için ağırlık/tokenizer dosyalarının değişmez commit ve hash kayıtları ile indirme/dönüştürme komutları ayrıca tamamlanmalıdır. FP16'dan ayrı ayrı üretilmiş Q8 ve Q4 koşullarında `group_size=64`, `affine` kullanılmıştır. Model ağırlıkları depoya dahil değildir.

Tam deneyin tekrarlanabilmesi için model hazırlama adımları, puanlama kodu ve kuralları, deney komutları, sonuçlar ve analiz kodları da yayımlanmalıdır. Süre ve bellek ölçümleri donanım ve sistem yüküne göre değişebilir.

## Dosyalar ve çalışma düzeni

- `scripts/prepare_dataset.py`: veri hazırlama kodu.
- `ProjeMarkdown/`: tamamlanan ve planlanan aşamaların notları.
- `data/processed/manifest.json`: yerel veri hash kayıtları.
- `requirements.txt`: yerel Python paket sürümleri.

`.venv/`, `models/`, ham veriler ve işlenmiş JSONL dosyaları `.gitignore` ile dışlanır. Bu depo mevcut deney klasöründen alınmış bir kopyadır; deney klasöründeki sonraki değişiklikler buraya otomatik aktarılmaz.
