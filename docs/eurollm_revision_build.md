# EuroLLM ağırlıklarının yeniden üretimi

Kaynak, [utter-project/EuroLLM-1.7B modelinin sabit revizyonu](https://huggingface.co/utter-project/EuroLLM-1.7B/tree/e00c97a9a389cc184174163cff4ba58facdb8e28) olan `e00c97a9a389cc184174163cff4ba58facdb8e28` commit’idir. Bu commit, ilk deneyin yerel `refs/pr/5` indirme kaydındaki commit ile aynıdır. İndirilen safetensors dosyası, dönüşümden önce başlangıç denetiminde kaydedilen kaynak SHA-256 değeriyle karşılaştırılır.

## Yeniden çalıştırma

Depo kökünde, `requirements.txt` içindeki sürümleri kullanan MLX ortamıyla:

```bash
mkdir -p outputs/revision_v2/eurollm
(set -o pipefail; bash scripts/rebuild_eurollm_revision.sh 2>&1 | tee -a outputs/revision_v2/eurollm/rebuild.log)
```

[Derleme betiği](../scripts/rebuild_eurollm_revision.sh) sabit revizyondan indirilecek dosyaları ve tam dönüşüm komutlarını içerir. FP16 için açıkça `--dtype float16` kullanılır. Q8 ve Q4, bu FP16 dizininden ayrı ayrı, sırasıyla `--q-bits 8` ve `--q-bits 4` ile üretilir. İki komutta da `--q-group-size 64 --q-mode affine --dtype float16` kullanılır.

Yerel kaynak ve yeni modeller `models/revision_v2/eurollm/` altında tutulur. İlk deneyin `models/eurollm-*` dizinleri karşılaştırma için okunur. [İlk dönüşüm kaydı](eurollm_revision_conversion_trace.txt), çalıştırılan komutları ve o sıradaki model yollarını arşivler. [Doğrulama manifesti](eurollm_revision_manifest.json), güncel model yollarını, ortam sürümlerini, dosyaların SHA-256 değerlerini, eski ağırlıklarla karşılaştırmayı ve küçük puanlama testlerini kaydeder.

## Doğrulama sonucu

Sabit kaynaktan indirilen kaynak ve yeniden üretilen FP16, Q8 ve Q4 ağırlık dosyalarının SHA-256 değerleri ilk deneydeki karşılıklarıyla eşleşti. Karşılaştırılan tokenizer dosyaları da eşleşti.

FP16, Q8 ve Q4 ayrı ayrı MLX ile yüklendi. Ağırlık indeksleri, tensör veri türleri ve kuantizasyon ayarları doğrulandı. İki örnek cümlenin token dizisi üç yapılandırmada aynı kaldı ve her yapılandırma sonlu Mean ve Total Log Probability skorları üretti.

Ana revizyon deneyinde 3.662 çiftin tamamı bu ağırlıklarla yeniden puanlandı.
