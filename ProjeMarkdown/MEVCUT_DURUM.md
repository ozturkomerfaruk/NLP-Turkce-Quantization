# Projenin Mevcut Durumu

**Son güncelleme:** 25 Ekim 2026  
**Proje:** Bir Çok Dilli Dil Modelinde Nicemlemenin Türkçe Morfosentaktik Yeterliliğe Etkisinin Deneysel İncelenmesi

## Genel durum

EuroLLM-1.7B Base modelinin **FP16, Q8 ve Q4 sürümleri hazır**, TurBLiMP deney veri setinin hazırlanması tamamlandı. `scripts/prepare_dataset.py` ile indirilen ve işlenen veriler yerel dosyalar üzerinden kontrol edildi. **Sıradaki deney aşaması 20 çiftlik pilot değerlendirme.** Henüz dilbilgisel doğruluk sonucu veya nicemlemenin doğruluğa etkisine ilişkin deneysel bulgu yok.

| Aşama | Durum | Açıklama |
| --- | --- | --- |
| Ortam, model kurulumu ve nicemleme | Tamamlandı | Üç model mevcut; önceki kurulum notlarında Türkçe üretim testleri başarılı. |
| TurBLiMP veri hazırlama | Tamamlandı | Dört kategoriden 250'şer, toplam 1.000 benzersiz çift doğrulandı. |
| Tekrarlanabilirlik manifesti | Eksik | Kaynak revizyonu ve seed kodda var; ayrı `manifest.json` yok. |
| 20 çiftlik pilot deney | Sıradaki | Puanlama kodu ve pilot sonuçları henüz bulunmuyor. |
| 1.000 çiftlik ana deney | Bekliyor | Pilot doğrulandıktan sonra yapılacak. |
| İstatistiksel analiz ve raporlama | Bekliyor | Ana deneyin kaydedilmiş puanlarını kullanacak. |

## Hazır olan modeller ve çalışma ortamı

Kurulum kayıtlarına göre sistem MacBook Pro M2 / 16 GB RAM; araçlar Python 3.12, `.venv`, MLX-LM 0.32.2 ve Apple Metal. Kaynak model `utter-project/EuroLLM-1.7B`, kullanılan indirme revizyonu `refs/pr/5` olarak kaydedilmiş.

| Model | Yerel klasör | Doğrulanan özellik |
| --- | --- | --- |
| Kaynak | `models/eurollm-source/` | Ağırlık, tokenizer ve yapılandırma dosyaları mevcut. |
| FP16 | `models/eurollm-fp16/` | Ağırlık başlıklarında `F16`; referans model. |
| Q8 | `models/eurollm-q8/` | Yapılandırmada 8-bit, `group_size=64`, `affine`. |
| Q4 | `models/eurollm-q4/` | Yapılandırmada 4-bit, `group_size=64`, `affine`. |

Kurulum notlarına göre Q8 ve Q4 aynı FP16 modelinden ayrı ayrı üretildi. Yeniden indirme, dönüştürme veya nicemleme gerekmiyor. Bu dokümantasyon güncellemesinde modeller yeniden çalıştırılmadı.

## TurBLiMP hazırlığında tamamlananlar

`scripts/prepare_dataset.py`, TurBLiMP GitHub deposunun `297de13fb7a0ce524fe32e8b175c6b5255d66960` commit'ine sabitlenmiş dört CSV dosyasını kullanıyor. Boş cümleleri, doğru ve hatalı cümlesi aynı kayıtları ve yinelenen cümle çiftlerini eleyecek şekilde hazırlanmış. Mevcut ham verilerde geçersiz kayıt ve kategoriler arası tekrar bulunmadı; kategori içi tekrarlar temizlendi.

| Kategori anahtarı | Ham çift | Tekrar temizliği sonrası uygun çift | Seçilen |
| --- | ---: | ---: | ---: |
| `subject_agreement` | 1.000 | 934 | 250 |
| `argument_structure_transitive` | 1.000 | 835 | 250 |
| `nominalization` | 1.000 | 945 | 250 |
| `irregular_forms` | 1.000 | 948 | 250 |
| **Toplam** | **4.000** | **3.662** | **1.000** |

- Ham dosyalar: `data/raw/`.
- Sabit deney örneklemi: `data/processed/turblimp_1000.jsonl`.
- Örnekleme: `SEED = 42`, kategori başına 250 çift; mevcut dosyayla aynı seçim ve sıra yeniden doğrulandı.
- Çıktı: 1.000 kayıt, 1.000 benzersiz çift ve 1.000 benzersiz `pair_id`; toplam 2.000 cümle girdisi. Geçersiz çift yok.
- Kayıtlar kategori, doğru/hatalı cümle, kaynak dosya ve satır indeksi ile dilbilimsel ek alanları içeriyor.

Çıktının doğrulanan SHA-256 değeri:

```text
bcb2c7f25475f8290d1511d742e1e9f8f515fd29df11dd3e3f64f2f53bcb4835
```

Üç model **aynı 1.000 çift** üzerinde değerlendirilecek. Model başına ayrı örnekleme yapılmayacak. Kod ham dosyaların ve çıktının hash değerlerini hesaplıyor, ancak bunları kaynak bilgileriyle birlikte kalıcı bir manifest dosyasına yazmıyor. Veri seti kullanılabilir durumda; manifest, akademik izlenebilirlik için tamamlanacak belgelemeyi oluşturuyor.

## Ön ölçümler ve sıradaki adım

Önceki kurulum kayıtlarındaki tek seferlik 20 token üretim testleri:

| Ölçüm | FP16 | Q8 | Q4 |
| --- | ---: | ---: | ---: |
| Tepe bellek | 3,352 GB | 1,813 GB | 0,984 GB |
| Üretim hızı | 31,799 token/sn | 56,937 token/sn | 99,440 token/sn |

Bu değerler **TurBLiMP puanlama deneyinin süre/bellek sonuçları değildir** ve doğruluk hakkında bilgi vermez. Puanlama sırasında kaynak kullanımı ayrıca ölçülecek.

Önerilen sıra: manifesti belgelemek; ardından mevcut örneklemden kategori başına 5 çiftle 20 çiftlik pilotu yürütmek. Pilot üç modelde toplam **120 cümle puanlaması**, ana deney **6.000 cümle puanlaması** içerir. Pilot sırasında token başına ortalama ve toplam log-olasılık hesapları, token sınırları, karar kuralları, süre ve bellek ölçümü doğrulanacak. Bunlar bu güncelleme kapsamında uygulanmadı.

Tüm aşamaların amacı ve planı: [TUM_SUREC.md](TUM_SUREC.md).
