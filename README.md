# NLP-Turkce-Quantization

EuroLLM-1.7B Base modelinde FP16, Q8 ve Q4 sürümlerinin Türkçe morfosentaktik minimal çiftler üzerindeki karşılaştırılması.

## Mevcut durum

27 Ekim 2026 itibarıyla veri hazırlığı ve FP16/Q8/Q4 için 20 çiftlik pilot puanlama ve karşılaştırma tamamlandı. Depo güncel betikleri, manifesti, pilot sonuçlarını ve tarihli proje belgelerini içerir. Pilot doğruluğu FP16 %75, Q8 %75, Q4 %70’tir; bunlar ana deney sonuçları değildir. Süre/bellek ölçümleri, 1.000 çiftlik ana deney ve istatistiksel analiz bekliyor. Deney devam ediyor. Güncel durum için [durum arşivinin son kaydına](ProjeMarkdown/MEVCUT_DURUM.md) bakın.

## Veri setini yeniden oluşturma

Veri hazırlama betiği yalnızca Python standart kütüphanesini kullanır. Referans ortam Python 3.12.15'tir. İnternet erişimi gerekir.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python scripts/prepare_dataset.py
python scripts/finalize_pilot_dataset.py
shasum -a 256 data/processed/turblimp_1000.jsonl
```

Beklenen çıktı: dört kategoriden 250'şer, toplam 1.000 çift. Beklenen SHA-256:

```text
bcb2c7f25475f8290d1511d742e1e9f8f515fd29df11dd3e3f64f2f53bcb4835
```

Kaynak TurBLiMP commit'i `297de13fb7a0ce524fe32e8b175c6b5255d66960`, örnekleme seed'i `42`'dir. Betik mevcut ham dosyaları tekrar kullanır; temiz bir klonda ilgili dosyaları indirir. [Manifest](data/processed/manifest.json), ham veriler ile ana ve pilot örneklemlerin hash kayıtlarını içerir; `finalize_pilot_dataset.py` manifesti ve 20 çiftlik pilotu oluşturur. Kaynak veri ve modellerin kullanım/lisans koşulları kendi kaynaklarından kontrol edilmelidir.

## Model ortamı ve kalan işler

Yerel ortam: macOS, Apple Silicon M2, 16 GB RAM; Python 3.12.15, MLX-LM 0.32.2, MLX 0.32.4. `requirements.txt`, mevcut ortamda kurulu paketlerin sabitlenmiş sürüm kaydıdır; temiz ortamda kurulum bu aktarım sırasında test edilmemiştir. Veri hazırlamak için bu paketlerin kurulması gerekmez. Model çalışmaları için uyumlu Apple Silicon ortamında:

```bash
python -m pip install -r requirements.txt
```

Kaynak model `utter-project/EuroLLM-1.7B`; kurulum notlarındaki indirme revizyonu `refs/pr/5`'tir. Bu ad değişebilir: tam model yeniden üretimi için ağırlık/tokenizer dosyalarının değişmez commit ve hash kayıtları ile indirme/dönüştürme komutları ayrıca tamamlanmalıdır. FP16'dan ayrı ayrı üretilmiş Q8 ve Q4 koşullarında `group_size=64`, `affine` kullanılmıştır. Model ağırlıkları depoya dahil değildir.

Pilot puanlama için hazırlanmış modellerin `models/eurollm-fp16`, `models/eurollm-q8` ve `models/eurollm-q4` konumlarında bulunması gerekir. `python scripts/score_pilot.py --model fp16` komutunda `fp16`, `q8` veya `q4` seçilir. `python scripts/summarize_pilot.py` mevcut pilot sonuçlarını ve yerel tokenizer’ları karşılaştırır. Puanlama betiğini yeniden çalıştırmak seçilen modelin pilot sonuç dosyalarının üzerine yazar. Bu kopyalama sırasında deneyler yeniden çalıştırılmadı.

Tam deneyin tekrarlanabilmesi için model hazırlama adımları, ana deney komutları, sonuçları ve analiz kodları da yayımlanmalıdır. Süre ve bellek ölçümleri donanım ve sistem yüküne göre değişebilir.

## Dosyalar ve çalışma düzeni

- `scripts/`: ana/pilot veri hazırlama, pilot puanlama ve karşılaştırma kodları.
- `results/pilot/`: üç modelin pilot kayıtları, özetleri ve karşılaştırma sonucu.
- `ProjeMarkdown/`: tamamlanan ve planlanan aşamaların notları.
- `data/processed/manifest.json`: yerel veri hash kayıtları.
- `requirements.txt`: yerel Python paket sürümleri.

`.venv/`, `models/`, ham veriler ve işlenmiş JSONL dosyaları `.gitignore` ile dışlanır. Bu depo mevcut deney klasöründen alınmış bir kopyadır; deney klasöründeki sonraki değişiklikler buraya otomatik aktarılmaz.
