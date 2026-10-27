# Projenin Mevcut Durumu

> **SABİT KURAL — Bu dosya tarihli durum arşividir.** Önceki durum kayıtları silinmez, yeniden yazılmaz veya yeni sonuçlarla değiştirilmez. Her yeni durum, dosyanın **sonuna** tarih ve artan kayıt numarasıyla ayrı bir bölüm olarak eklenir. Böylece geçilen aşamalar ve o tarihte bekleyen işler sürüm geçmişi gibi korunur. En güncel durum, dosyanın en son kaydıdır. Bu kural en başta tutulur.
>
> Her yeni kayıt; önceki kayda göre ilerlemeyi, tamamlanan aşamaları, o tarihteki sonuçları ve kanıt dosyalarını, bekleyen işleri ve sıradaki adımı içerir. Aynı gün birden fazla kayıt eklenebilir; kayıt numarası artırılır, biliniyorsa saat de yazılır. Eski bir kayıtta hata bulunursa geçmiş kayıt değiştirilmez; düzeltme, ilgili kayda atıf yapan yeni bir kayıt olarak sona eklenir. Yeni sonuçlar geçmiş tarihlere uygulanmaz.
>
> **Kapsam:** Yalnızca `ProjeMarkdown/` altındaki `.md` belgeleri düzenlenir; kod, veri, sonuç ve model dosyaları değiştirilmez, eklenmez veya silinmez.

## Kayıt 001 — 25 Ekim 2026 — Model ve ana veri hazırlığı tamamlandı

**Arşiv notu:** Bu bölüm, 27 Ekim’de dosyanın tarihli arşive dönüştürülmesi sırasında, önceki 25 Ekim belgesindeki bilgilerden özetlenerek geri eklendi. Önceki belgenin birebir kopyası değildir; aşağıdaki durum 25 Ekim kaydını temsil eder.

| Aşama | O tarihteki durum | Açıklama |
| --- | --- | --- |
| Ortam ve model hazırlığı | Tamamlandı | EuroLLM-1.7B Base modelinin FP16, Q8 ve Q4 sürümleri hazır. |
| TurBLiMP veri hazırlığı | Tamamlandı | Dört kategoriden 250’şer, toplam 1.000 benzersiz çift hazır. |
| Tekrarlanabilirlik manifesti | Eksik | Ayrı `manifest.json` henüz yok. |
| 20 çiftlik pilot | Sıradaki | Pilot seçimi, puanlama betiği ve sonuçları henüz yok. |
| 1.000 çiftlik ana deney | Bekliyor | Pilot doğrulandıktan sonra yapılacak. |
| İstatistiksel analiz ve raporlama | Bekliyor | Ana deney sonuçlarını kullanacak. |

**Kanıt ve önceki kayıt bilgisi:** `scripts/prepare_dataset.py`, `data/processed/turblimp_1000.jsonl`, dört model klasörü ve önceki kurulum notları. Ana veri SHA-256 değeri `bcb2c7f25475f8290d1511d742e1e9f8f515fd29df11dd3e3f64f2f53bcb4835`. Ön üretim testleri mevcut; dilbilgisel doğruluk sonucu henüz yok.

**Sıradaki adım:** Manifesti tamamlamak, kategori başına 5 çiftle pilot veri setini oluşturmak ve üç modelde puanlama yöntemini doğrulamak.

## Kayıt 002 — 27 Ekim 2026 — Manifest ve FP16 pilotu tamamlandı

**Önceki kayda göre ilerleme:** Manifest ve 20 çiftlik pilot veri seti oluşturuldu; FP16 puanlama akışı doğrulandı ve pilot %75 doğrulukla tamamlandı. Q8/Q4 pilotları, karşılaştırma ve puanlama süre/bellek ölçümleri bekliyor.

**Son güncelleme:** 27 Ekim 2026  
**Proje:** Bir Çok Dilli Dil Modelinde Nicemlemenin Türkçe Morfosentaktik Yeterliliğe Etkisinin Deneysel İncelenmesi

### Genel durum

EuroLLM-1.7B Base modelinin **FP16, Q8 ve Q4 sürümleri hazır**, TurBLiMP deney veri setinin hazırlanması tamamlandı. `scripts/prepare_dataset.py` ile indirilen ve işlenen veriler yerel dosyalar üzerinden kontrol edildi. **Manifest ve 20 çiftlik pilot veri seti hazır; FP16 pilot değerlendirmesi tamamlandı.** FP16, 20 çiftte 15 doğru ve 5 yanlış kararla **%75 pilot doğruluğu** elde etti; eşit puan yok. **Sıradaki adım aynı 20 çift üzerinde Q8 pilot değerlendirmesi**, ardından Q4 ve üç modelin karşılaştırılması. Henüz nicemlemenin doğruluğa etkisine ilişkin karşılaştırmalı bulgu yok.

| Aşama | Durum | Açıklama |
| --- | --- | --- |
| Ortam, model kurulumu ve nicemleme | Tamamlandı | Üç model mevcut; önceki kurulum notlarında Türkçe üretim testleri başarılı. |
| TurBLiMP veri hazırlama | Tamamlandı | Dört kategoriden 250'şer, toplam 1.000 benzersiz çift doğrulandı. |
| Tekrarlanabilirlik manifesti | Tamamlandı | `data/processed/manifest.json` kaynak revizyonunu, seed değerlerini, kategori sayılarını ve veri hash değerlerini içeriyor. |
| FP16 — 20 çiftlik pilot | Tamamlandı | 15 doğru, 5 yanlış, 0 eşit puan; doğruluk %75. |
| Q8 ve Q4 — 20 çiftlik pilot | Bekliyor | Aynı pilot veri seti kullanılacak. |
| FP16/Q8/Q4 karşılaştırması | Bekliyor | Tokenlaştırma tutarlılığı ve karar değişimleri incelenecek. |
| Puanlama süresi ve bellek ölçümleri | Bekliyor | Mevcut FP16 sonuçları bu ölçümleri içermiyor. |
| 1.000 çiftlik ana deney | Bekliyor | Pilot doğrulandıktan sonra yapılacak. |
| İstatistiksel analiz ve raporlama | Bekliyor | Ana deneyin kaydedilmiş puanlarını kullanacak. |

### Hazır olan modeller ve çalışma ortamı

Kurulum kayıtlarına göre sistem MacBook Pro M2 / 16 GB RAM; araçlar `.venv`, Python 3.12.15, MLX 0.32.4, MLX-LM 0.32.2 ve Apple Metal. Sürüm bilgileri FP16 pilot özetinde kayıtlı. Kaynak model `utter-project/EuroLLM-1.7B`, kullanılan indirme revizyonu `refs/pr/5` olarak kaydedilmiş.

| Model | Yerel klasör | Doğrulanan özellik |
| --- | --- | --- |
| Kaynak | `models/eurollm-source/` | Ağırlık, tokenizer ve yapılandırma dosyaları mevcut. |
| FP16 | `models/eurollm-fp16/` | Ağırlık başlıklarında `F16`; referans model. |
| Q8 | `models/eurollm-q8/` | Yapılandırmada 8-bit, `group_size=64`, `affine`. |
| Q4 | `models/eurollm-q4/` | Yapılandırmada 4-bit, `group_size=64`, `affine`. |

Kurulum notlarına göre Q8 ve Q4 aynı FP16 modelinden ayrı ayrı üretildi. Yeniden indirme, dönüştürme veya nicemleme gerekmiyor. Bu dokümantasyon güncellemesinde modeller yeniden çalıştırılmadı.

### TurBLiMP hazırlığında tamamlananlar

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

Üç model **aynı 1.000 çift** üzerinde değerlendirilecek. Model başına ayrı örnekleme yapılmayacak. `scripts/finalize_pilot_dataset.py` ile ana veri setinin hash değeri korunarak kategori başına 5 çift seçilmiş ve `data/processed/manifest.json` oluşturulmuş. Manifest kaynak revizyonunu, ana veri ve pilot seed değerlerini (`42`), kategori sırası/sayılarını, ham dosya hash değerlerini ve iki işlenmiş veri dosyasının hash değerlerini içeriyor. Ana veri ve pilot hash değerleri bu güncellemede yeniden hesaplanarak doğrulandı.

Pilot dosyası `data/processed/turblimp_pilot_20.jsonl`: 20 çift, kategori başına 5 çift. SHA-256 değeri:

```text
420407b129d0e997680a50ab11ff49319e933c1b82b07068042b5d1249239f27
```

### FP16 pilot değerlendirmesi — tamamlandı

Paylaşılan terminal çıktısı ve yerel `results/pilot/fp16_pilot_20.jsonl` ile `results/pilot/fp16_summary.json` kayıtları birbiriyle uyumlu. Sonuç dosyasındaki 20 çiftin kimlikleri ve sırası pilot veri setiyle eşleşiyor.

| Kategori | Doğru / Toplam | Doğruluk |
| --- | ---: | ---: |
| Subject Agreement | 2/5 | %40 |
| Argument Structure Transitive | 3/5 | %60 |
| Nominalization | 5/5 | %100 |
| Irregular Forms | 5/5 | %100 |
| **Genel** | **15/20** | **%75** |

Puanlama `add_special_tokens=False` ile yapılıyor; BOS (`<s>`, ID `1`) elle başa ekleniyor ve kendisi puanlanmıyor. EOS eklenmiyor. Kaydırılmış input/target eşleşmesiyle ilk cümle token’ı da puanlanıyor. FP16 logits, log-softmax öncesinde FP32’ye çevriliyor. Ana puan tüm cümlenin **token başına ortalama log-olasılığı**; karar farkı doğru cümlenin puanı eksi hatalı cümlenin puanı. Pozitif fark doğru karar, negatif fark yanlış karar; sıfır fark eşitlik olarak sayılıyor ve başarıya dahil edilmiyor.

İlk çift `subject_agreement:0066` için iki cümle de 11 token. Doğru cümle “Ah sen ne kadar güzel resim yapıyorsun.” için ortalama puan `-4.622046`, hatalı cümle “Ah o ne kadar güzel resim yapıyorsun.” için `-4.566949`; fark `-0.055097` ve karar yanlış. Terminalde input/target boyutu `(1, 11)`, logits boyutu `(1, 11, 128000)` ve bağımsız hesapla puanlama fonksiyonunun `1e-4` toleransında uyuştuğu görülüyor. Token konumlarının karşılaştırılması dilbilimsel hizalama anlamına gelmiyor.

JSONL kayıtları çift kimliği, kategori, model, veri hash değeri, puanlama yöntemi, iki ortalama puan, token sayıları, fark ve doğru/yanlış kararını saklıyor. Toplam log-olasılık terminalde hesaplanmış olsa da mevcut JSONL’ye kaydedilmiyor. Özet dosyası puanlama/tokenlaştırma kurallarını, ortam sürümlerini ve genel başarıyı içeriyor; süre ve bellek ölçümlerini içermiyor.

**Bu %75 değeri yalnızca 20 çiftlik FP16 pilotuna aittir.** Ana deney sonucu değildir; Q8/Q4 değerlendirmesi olmadan nicemlemenin etkisi hakkında sonuç çıkarılamaz.

### Ön ölçümler ve sıradaki adım

Önceki kurulum kayıtlarındaki tek seferlik 20 token üretim testleri:

| Ölçüm | FP16 | Q8 | Q4 |
| --- | ---: | ---: | ---: |
| Tepe bellek | 3,352 GB | 1,813 GB | 0,984 GB |
| Üretim hızı | 31,799 token/sn | 56,937 token/sn | 99,440 token/sn |

Bu değerler **TurBLiMP puanlama deneyinin süre/bellek sonuçları değildir** ve doğruluk hakkında bilgi vermez. Puanlama sırasında kaynak kullanımı ayrıca ölçülecek.

Kalan sıra: aynı 20 çift üzerinde Q8 pilotu → Q4 pilotu → FP16/Q8/Q4 tokenlaştırma ve karar karşılaştırması → süre/bellek ölçümlerinin doğrulanması → 1.000 çiftlik ana deney → istatistiksel analiz ve makale bulguları. Üç modelde pilot toplam **120 cümle puanlaması**, ana deney **6.000 cümle puanlaması** içerir.

Mevcut çalıştırma komutu `python scripts/score_pilot.py`; betik FP16 yolunu sabit kullanıyor. `--model fp16/q8/q4` seçimi, otomatik ısınma ve ölçüm akışı, kesinti sonrası devam desteği ve `scripts/summarize_pilot.py` henüz mevcut değil. Bunlar sonraki geliştirme işleri; mevcut komutlara eklenmiş özellikler olarak değerlendirilmemeli. Betik sonuçları 20 çift bittikten sonra `fp16_pilot_20.jsonl` ve `fp16_summary.json` adlarıyla yazıyor; her çift sonrasında diske yazmıyor ve yeniden çalıştırıldığında bu dosyaların üzerine yazıyor. Mevcut FP16 sonuçları korunarak sonraki değerlendirmeler hazırlanmalı.

Bu güncellemede yalnızca `ProjeMarkdown/` altındaki mevcut iki `.md` dosyası düzenlendi; deneyler yeniden çalıştırılmadı.

Tüm aşamaların amacı ve planı: [TUM_SUREC.md](TUM_SUREC.md).


## Kayıt 003 — 27 Ekim 2026 — Q8 ve Q4 pilotları tamamlandı

**Önceki kayda göre ilerleme:** Aynı 20 çift Q8 ve Q4 ile değerlendirildi; üç modelin sonuçları yerel JSONL ve özet dosyalarından karşılaştırıldı. FP16 ve Q8 %75, Q4 %70 pilot doğruluğuna ulaştı. Önceki kayıtlar o tarihteki durumu yansıttığından aynen korunmuştur.

### Tamamlanan aşamalar ve sonuçlar

| Aşama | Durum | Açıklama |
| --- | --- | --- |
| Ortam ve FP16/Q8/Q4 model hazırlığı | Tamamlandı | Üç modelin pilot çıktıları mevcut. |
| Ana veri, pilot veri ve manifest | Tamamlandı | Ana veri 1.000 çift; pilot dört kategoride 5’er çift. |
| FP16 — 20 çiftlik pilot | Tamamlandı | 15 doğru, 5 yanlış; %75. |
| Q8 — 20 çiftlik pilot | Tamamlandı | 15 doğru, 5 yanlış; %75. |
| Q4 — 20 çiftlik pilot | Tamamlandı | 14 doğru, 6 yanlış; %70. |
| Genel/kategori doğruluğu ve karar karşılaştırması | Tamamlandı | Kayıtlar üzerinden hesaplandı; aşağıdaki tablolarla belgelendi. |
| Tokenlaştırma tutarlılığının tam doğrulanması | Kısmen tamamlandı | Tüm çiftlerde modelden modele token sayıları eşit; tüm token ID dizileri henüz karşılaştırılmadı. |
| Puanlama süresi ve bellek ölçümleri | Bekliyor | Mevcut özetler bu ölçümleri içermiyor. |
| 1.000 çiftlik ana deney | Bekliyor | Kalan pilot kontrollerinden sonra yapılacak. |
| İstatistiksel analiz ve makale bulguları | Bekliyor | Ana deney sonuçlarını kullanacak. |

| Kategori | FP16 | Q8 | Q4 |
| --- | ---: | ---: | ---: |
| Subject Agreement | 2/5 (%40) | 2/5 (%40) | 2/5 (%40) |
| Argument Structure Transitive | 3/5 (%60) | 3/5 (%60) | 3/5 (%60) |
| Nominalization | 5/5 (%100) | 5/5 (%100) | 4/5 (%80) |
| Irregular Forms | 5/5 (%100) | 5/5 (%100) | 5/5 (%100) |
| **Genel** | **15/20 (%75)** | **15/20 (%75)** | **14/20 (%70)** |

Üç modelde de eşit puan sayısı sıfır. Her modelde 40 cümle, üç modelde toplam 120 cümle pilot kapsamında puanlandı; ilk çiftin ek tanılama hesapları bu sayıya dahil değil.

### FP16’ya göre karar değişimleri

| Model | Doğru → yanlış | Yanlış → doğru | Aynı karar | Doğruluk farkı |
| --- | ---: | ---: | ---: | ---: |
| Q8 | 0 | 0 | 20/20 | 0 yüzde puan |
| Q4 | 1 | 0 | 19/20 | −5 yüzde puan |

Q8’in tüm kararları FP16 ile aynı; sayısal puanlar değişmiş olsa da karar yönleri değişmemiş. Q4’te değişen tek çift `nominalization:0577`:

- Doğru cümle: “Beyaz meyvelerin erken olgunlaşmasına sıcak havalar sebep oldu.”
- Hatalı cümle: “Beyaz meyvelerin erken olgunlaştığına sıcak havalar sebep oldu.”
- FP16 doğru–hatalı ortalama puan farkı: `+0.064758`; Q4 farkı: `−0.021774`.

**Yorum sınırı:** Q4 bu pilotta FP16’dan bir çift daha az doğru karar verdi. Yalnızca 20 çift üzerinden nicemlemenin genel Türkçe dilbilgisel yeterliliğe etkisi, istatistiksel anlamlılık veya ana deney sonucu hakkında hüküm verilemez. Tokenlaştırma kontrolü de tamamlanmalıdır.

### Doğrulanan kanıtlar ve kontrol sınırları

Sonuç dosyaları:

- `results/pilot/fp16_pilot_20.jsonl` ve `results/pilot/fp16_summary.json`.
- `results/pilot/q8_pilot_20.jsonl` ve `results/pilot/q8_summary.json`.
- `results/pilot/q4_pilot_20.jsonl` ve `results/pilot/q4_summary.json`.

Her JSONL dosyasında 20 kayıt var; çift kimlikleri ve sırası pilot veri setiyle eşleşiyor. Kayıtlardaki veri hash değerleri pilot dosyasının yeniden hesaplanan SHA-256 değeriyle aynı:

```text
420407b129d0e997680a50ab11ff49319e933c1b82b07068042b5d1249239f27
```

Her kayıtta `delta = good_score - bad_score` ve `correct = (delta > 0)` ilişkisi doğrulandı. Başarı toplamları özet dosyaları ve paylaşılan Q8/Q4 terminal çıktılarıyla uyuşuyor. Q8/Q4 özetleri kategori sonuçlarını da içeriyor. Terminalde iki modelin ilk çift puanlama fonksiyonu doğrulaması başarılı; logits FP16, log-olasılıklar FP32 olarak gösteriliyor.

Tüm pilot çiftlerinde doğru ve hatalı cümlelerin token sayıları modeller arasında ayrı ayrı eşleşiyor. Bununla birlikte sayı eşitliği token ID dizilerinin eşitliğini kanıtlamaz. Yerel `tokenizer.json` dosyalarının karşılaştırmasında FP16 için `pre_tokenizer.prepend_scheme=always`, Q8/Q4 için `first` bulundu. Bu dosyalardaki diğer JSON alanları aynı; `tokenizer_config.json` dosyaları da bayt düzeyinde aynı. Bu ayar farkının pilot token ID dizilerini etkileyip etkilemediği bu güncellemede çalıştırılarak kontrol edilmedi; kalan doğrulama işidir.

### Güncel komut durumu ve önceki kayda ilişkin açıklama

Kayıt 002’de model seçiminin bulunmadığı belirtilmişti. **Bu yeni kayıt itibarıyla** yerel `scripts/score_pilot.py`, `argparse` ile zorunlu `--model` seçimini ve `fp16`, `q8`, `q4` değerlerini destekliyor. Kullanıcının çalıştırdığı ve çıktısını paylaştığı komutlar:

```bash
python scripts/score_pilot.py --model q8
python scripts/score_pilot.py --model q4
```

FP16 için güncel komut biçimi `python scripts/score_pilot.py --model fp16`. Çıktılar seçilen modele göre `<model>_pilot_20.jsonl` ve `<model>_summary.json` adlarıyla ayrı tutuluyor. Paylaşılan metnin başındaki eski “FP16 PİLOT ÖZETİ” başlığı Q8 dosya yoluyla birlikte görünüyor; sonraki tam Q8 çalışmasının başlıkları ve yerel özetin `precision=Q8` değeri tutarlı. Karşılaştırmada yerel kayıtlar ve tam çalışma çıktıları esas alındı.

Son kontrolde `scripts/summarize_pilot.py` boş bir dosya olarak mevcut; karşılaştırma işlevi henüz uygulanmamış. Burada verilen karşılaştırma mevcut sonuç dosyaları okunarak yapıldı. Süre/bellek ölçümü, ayrı bir ısınma ölçüm akışı ve kesinti sonrası devam desteği mevcut betikte henüz bulunmuyor. Sonuçlar tüm çiftler tamamlandıktan sonra yazılıyor; aynı modeli yeniden çalıştırmak ilgili sonuç dosyalarının üzerine yazar.

### Sıradaki adım

1. Tokenizer ayar farkını dikkate alarak aynı 40 pilot cümlenin token ID dizilerini üç model arasında doğrulamak ve sonucu belgelemek.
2. Yükleme/ısınma/puanlama sürelerinin kapsamını belirlemek, bellek ölçümlerini toplamak ve kaynak tüketimini doğrulamak.
3. Bu kontrollerden sonra aynı puanlama kurallarıyla 1.000 çiftlik ana deneye geçmek; ardından istatistiksel analiz ve makale bulgularını hazırlamak.

Bu kayıt eklenirken yalnızca `ProjeMarkdown/MEVCUT_DURUM.md` dosyasının sonuna içerik eklendi. Önceki kayıtlar değiştirilmedi; kod, veri, sonuç ve model dosyalarına dokunulmadı. Modeller veya deney betikleri yeniden çalıştırılmadı.


## Kayıt 004 — 27 Ekim 2026 — Pilot karşılaştırma ve tokenlaştırma kontrolleri tamamlandı

**Önceki kayda göre ilerleme:** `scripts/summarize_pilot.py` artık uygulanmış ve kullanıcı tarafından çalıştırılmış. Son terminal çıktısı “PİLOT KARŞILAŞTIRMA KONTROLLERİ BAŞARILI.” mesajıyla tamamlanmış; `results/pilot/pilot_comparison.json` oluşturulmuş. Kayıt 003’te bekleyen tüm pilot cümlelerinin token ID karşılaştırması tamamlandı. Önceki kayıtlar aynen korunmuştur.

### Doğrulama kapsamı ve kanıtlar

- Pilot veri SHA-256 değeri ve 20 benzersiz çift kontrolü başarılı. Kullanılan SHA-256: `420407b129d0e997680a50ab11ff49319e933c1b82b07068042b5d1249239f27`.
- Üç modelin sonuç/özet dosyaları; çift kimlikleri ve sırası, kategori/model/yöntem etiketleri, sonlu puanlar, puan farkları, doğru/yanlış kararları, token sayıları, genel başarı ve eşitlik sayıları bakımından karşılaştırma betiğiyle doğrulanmış.
- Kullanıcının başarılı çalıştırma çıktısına göre **BOS ID ve 40 pilot cümlenin token ID dizileri FP16, Q8 ve Q4’te aynı**. Yeniden tokenlaştırılan cümlelerin uzunlukları da kaydedilmiş sonuçların token sayılarıyla eşleşmiş.
- Kayıt 003’te belirtilen `prepend_scheme` ayar farkı yerel tokenizer dosyalarında bulunuyor; buna rağmen bu 40 cümlede farklı token ID dizisi üretmemiş. Bu doğrulama pilot cümleleriyle sınırlı; ana deneydeki tüm cümleler için henüz aynı kontrol yapılmış değil.
- Kalıcı karşılaştırma kaydı `results/pilot/pilot_comparison.json`; veri hash değeri, toplam çift sayısı, genel/kategori doğrulukları, FP16’ya göre karar değişimleri ve `token_ids_equal_across_models: true` alanını içeriyor. Dosyanın hash referansı ve başarı/kategori değerleri bu dokümantasyon güncellemesinde pilot verisi ve kayıtlı sonuçlarla ayrıca karşılaştırıldı. Tokenizer doğrulaması için kullanıcının başarılı çalışma çıktısı ve ilgili betik incelendi; betik bu güncellemede yeniden çalıştırılmadı.

### Doğrulanan pilot karşılaştırması

| Kategori | FP16 | Q8 | Q4 |
| --- | ---: | ---: | ---: |
| Subject Agreement | 2/5 (%40) | 2/5 (%40) | 2/5 (%40) |
| Argument Structure Transitive | 3/5 (%60) | 3/5 (%60) | 3/5 (%60) |
| Nominalization | 5/5 (%100) | 5/5 (%100) | 4/5 (%80) |
| Irregular Forms | 5/5 (%100) | 5/5 (%100) | 5/5 (%100) |
| **Genel** | **15/20 (%75)** | **15/20 (%75)** | **14/20 (%70)** |

Q8’de FP16’ya göre karar değişimi yok. Q4’te tek değişim `nominalization:0577`: FP16 farkı `+0.064758` iken Q4 farkı `−0.021774`; doğru karardan yanlış karara dönüşmüş. Q4’ün doğruluk farkı −5 yüzde puan; üç modelde de eşit puan yok. Bu bulgular 20 çiftlik pilota aittir; genel nicemleme etkisi veya istatistiksel anlamlılık sonucu olarak yorumlanmaz.

### Terminal mesajlarına ilişkin kayıt

İlk çıktılardan birinde karar değişimleri başlığı kategori satırları arasında tekrarlanmış; son paylaşılan çalıştırmada başlık bir kez, kategori tablosundan sonra yer alıyor ve karşılaştırma dosyası başarıyla kaydedilmiş. Esas alınan çıktı bu son başarılı çalıştırmadır.

Terminaldeki PyTorch bulunamadı mesajı bu tokenizer ve kayıtlı sonuç karşılaştırmasını engellememiş. İncelenen betik model ağırlıklarını yüklemiyor; yerel tokenizer’ları ve mevcut sonuçları kullanıyor. Başarılı karşılaştırma çıktısı bu çalışma için bir PyTorch kurulumu gerektiren hata göstermiyor.

### Güncel aşamalar ve sıradaki adım

| Aşama | Durum |
| --- | --- |
| Model, ana veri, pilot veri ve manifest hazırlığı | Tamamlandı |
| FP16/Q8/Q4 — 20 çiftlik pilot puanlama | Tamamlandı |
| Sonuç dosyaları ve genel/kategori karşılaştırması | Tamamlandı |
| Pilotun tüm cümlelerinde token ID ve BOS eşitliği | Tamamlandı |
| Karar değişimlerinin kalıcı karşılaştırma dosyasına kaydı | Tamamlandı |
| Yükleme/ısınma/puanlama süreleri ve bellek ölçümleri | Bekliyor |
| 1.000 çiftlik ana deney ve ana veri genelinde tokenlaştırma kontrolü | Bekliyor |
| İstatistiksel analiz ve makale bulguları | Bekliyor |

**Sıradaki adım:** Süre/bellek ölçüm kapsamını belirlemek ve kaynak tüketimini doğrulamak; ardından ana deneyin aynı puanlama kuralları ve veri bütünlüğü kontrolleriyle yürütülmesine geçmek. Pilotun puanlama ve karşılaştırma kontrolleri tamamlandı; kaynak ölçümleri henüz kaydedilmiş değil.

Bu güncellemede yalnızca `ProjeMarkdown/MEVCUT_DURUM.md` sonuna bu kayıt eklendi. Önceki kayıtlar korundu; kod, veri, sonuç ve model dosyaları değiştirilmedi, deney veya karşılaştırma betikleri çalıştırılmadı.
