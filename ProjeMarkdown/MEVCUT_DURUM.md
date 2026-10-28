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


## Kayıt 005 — 27 Ekim 2026 — Üç modelde 1.000 çiftlik ana deney tamamlandı

**Önceki kayda göre ilerleme:** Kullanıcı `scripts/score_main.py` ve `scripts/summarize_main.py` betiklerini ekleyip çalıştırdı. FP16, Q8 ve Q4 aynı 1.000 çift üzerinde, model başına 20 adet 50 çiftlik parça halinde değerlendirildi. Ana deney dosya ve token kontrolleri başarılı; `results/main/main_comparison.json` kaydedildi. Puanlama süreleri, yükleme/ısınma süreleri ve bellek ölçümleri artık kayıtlı. Bu belge güncellemesinde kod eklenmedi veya değiştirilmedi; önceki durum kayıtları korundu.

### Bu kayıt içinde belgelenen ilerleme adımları

Aşağıdaki ara durumlar kullanıcının paylaştığı terminal çıktılarından geriye dönük kaydedilmiştir; ayrı zamanlarda eklenmiş arşiv sürümleri değildir.

1. **FP16 ilk parça:** `--max-chunks 1` ile 50 çift, 42 doğru (%84); puanlama 6,0393 sn, MLX tepe bellek 3,4055 GB. Yükleme 2,3746 sn, ısınma 0,5885 sn. Tamamlanan 1/20 parça.
2. **FP16 devamı:** `--max-chunks 19` önceki 1 parçayı tanıyıp kalan 19 parçayı tamamladı; sonuç 912/1.000 (%91,2). İlk 50 çift tüm deneyin yalnızca bir alt kümesidir.
3. **Q8 ilk parça ve devamı:** İlk 50 çiftte 43 doğru (%86); ardından kalan 19 parça tamamlandı. Sonuç 915/1.000 (%91,5).
4. **Q4:** `--max-chunks 20` ile 20 parça tek oturumda tamamlandı. Sonuç 884/1.000 (%88,4).
5. **Ana karşılaştırma:** `python scripts/summarize_main.py` dosya bütünlüğü ve token kontrollerini tamamlayıp genel/kategori sonuçları, karar geçişleri ve kaynak ölçümlerini kaydetti.

### Ana deney sonuçları

| Ölçüm | FP16 | Q8 | Q4 |
| --- | ---: | ---: | ---: |
| Tamamlanan parça | 20/20 | 20/20 | 20/20 |
| Değerlendirilen çift | 1.000 | 1.000 | 1.000 |
| Doğru karar | 912 | 915 | 884 |
| Yanlış karar | 88 | 85 | 116 |
| Eşit puan | 0 | 0 | 0 |
| Doğruluk | %91,2 | %91,5 | %88,4 |
| FP16’ya göre fark | — | +0,3 yüzde puan | −2,8 yüzde puan |
| Toplam parça puanlama süresi | 123,0381 sn | 97,8844 sn | 95,2026 sn |
| Ortalama puanlama süresi/çift | 0,1230 sn | 0,0979 sn | 0,0952 sn |
| Parçalardaki en yüksek MLX ayırıcı tepe belleği | 3,4161 GB | 2,0682 GB | 1,2367 GB |
| Kayıtlı en yüksek süreç RSS değeri | 1,2474 GB | 1,8393 GB | 1,5735 GB |
| Puanlanan token sayısı | 26.879 | 26.879 | 26.879 |

Süreler ham metrik dosyalarından toplanmıştır; paylaşılan FP16 özetindeki 123,03 sn yerine standart iki ondalık yuvarlama 123,04 sn’dir. Toplam iş yükü **1.000 çift × 2 cümle × 3 model = 6.000 cümle puanlaması**; ısınma hesapları bu sayıya dahil değil.

### Kategori sonuçları

| Kategori | FP16 | Q8 | Q4 |
| --- | ---: | ---: | ---: |
| Subject Agreement | 221/250 (%88,4) | 222/250 (%88,8) | 203/250 (%81,2) |
| Argument Structure Transitive | 233/250 (%93,2) | 233/250 (%93,2) | 229/250 (%91,6) |
| Nominalization | 210/250 (%84,0) | 212/250 (%84,8) | 203/250 (%81,2) |
| Irregular Forms | 248/250 (%99,2) | 248/250 (%99,2) | 249/250 (%99,6) |

Bu örneklemde Q8 doğruluğu FP16’dan 0,3 yüzde puan yüksek; bu gözlem bir üstünlük veya eşdeğerlik kanıtı olarak yorumlanmaz. Q4’te en büyük kategori düşüşü Subject Agreement: −7,2 yüzde puan. Diğer Q4 farkları sırasıyla −1,6, −2,8 ve +0,4 yüzde puan. Kategori düzeyinde anlamlılık testleri henüz yapılmadı.

### Eşleşmiş karar geçişleri ve ilk istatistiksel hesaplar

| FP16’ya göre geçiş | Q8 | Q4 |
| --- | ---: | ---: |
| Doğru → yanlış | 0 | 42 |
| Yanlış → doğru | 3 | 14 |
| Aynı karar | 997 | 944 |
| Net doğru karar farkı | +3 | −28 |

Q8’de düzelmiş çiftler `subject_agreement:0031`, `nominalization:0536` ve `nominalization:0857`. Q4’te değişen 56 çiftin kimlikleri karşılaştırma dosyasında kayıtlı.

İki yönlü kesin McNemar hesabı, uyumsuz çiftlerin sayısı `n = aşağı + yukarı` için `min(1, 2 × Σ[k=0..min(aşağı,yukarı)] C(n,k) / 2^n)` ile bu belge güncellemesinde ayrıca doğrulandı:

- **Q8:** 0 ve 3 geçiş için `p = 0,25`. %5 düzeyinde anlamlı fark saptanmıyor; bu sonuç eşdeğerlik veya nicemlemenin etkisizliği kanıtı değildir.
- **Q4:** 42 ve 14 geçiş için `p = 0,00023444586091919684`. Düzeltme uygulanmamış bu genel karşılaştırmada %5 düzeyinde anlamlı doğruluk düşüşü görülüyor.

Bunlar mevcut karar geçişlerinden doğrulanan **ilk hesaplar**. `main_comparison.json` p değerlerini içermiyor; güven aralıkları, çoklu karşılaştırma yaklaşımı ve ayrıntılı hata analizi henüz tamamlanmış değil. Önceki plan bu yöntemlerin analizden önce belirlenmesini öngörüyordu; bu ilk hesapların sonradan yapıldığı açıkça belgelenmeli. Bulgular bu model, yöntem ve seçilmiş dört kategorilik örneklemle sınırlı tutulmalı.

### Veri, token ve puanlama doğrulaması

Ana veri SHA-256 değeri değişmemiş:

```text
bcb2c7f25475f8290d1511d742e1e9f8f515fd29df11dd3e3f64f2f53bcb4835
```

Yöntem etiketi `mean_logp_float32_bos_context_no_eos_v1`: BOS bağlam olarak kullanılıyor, EOS eklenmiyor; tüm cümle token’larının FP32 log-olasılık ortalaması karar puanı. Toplam log-olasılık ve token sayısı da ana deney kayıtlarında mevcut. Eşitlik toleransı sıfır; yalnızca tam sıfır fark eşitlik.

Bu dokümantasyon güncellemesinde model ağırlıkları yüklenmeden, deney betikleri yeniden çalıştırılmadan **60 JSONL parçası, 60 metrik dosyası ve 3.000 çift kaydı** okundu. Kayıtların veri sırası/kimlikleri, kategori/model/yöntem/hash etiketleri, ortalama puan farkları, karar ve eşitlik kuralları kontrol edildi. Genel/kategori sayıları, karar geçiş listeleri, puanlama süreleri, MLX tepe bellekleri ve token toplamları karşılaştırma dosyasıyla uyuşuyor. Her ana veri cümlesinin kayıtlı token ID SHA-256 değeri ve token sayısı üç model arasında eşleşiyor.

### Kaynak ölçümlerinin kapsamı

Puanlama süresi parça döngüsünü kapsıyor; tokenlaştırma, puanlama, kayıtların bellekte hazırlanması ve çiftler arası önbellek temizleme bu döngüde. Model yükleme, ısınma ve tamamlanan parçanın diske yazılması bu süreye dahil değil. Yükleme/ısınma süreleri metriklerde oturum bilgisi olarak her parçada tekrarlandığından 20 dosyadaki bu alanları toplamak aynı oturumu birden fazla sayar. FP16 ve Q8 iki, Q4 tek oturumda tamamlanmış; süre karşılaştırması bu çalışma koşullarıyla sınırlı.

Bellek GB değerleri 10⁹ bayt üzerinden hesaplanıyor. MLX ayırıcı tepe belleği toplam sistem RAM’i değildir. Süreç RSS ölçümü süreç boyunca biriken üst sınırdır; MLX ölçümüyle toplanarak toplam RAM veya GPU belleği elde edilemez. Kullanıcının paylaştığı açıklamada swap ve yüksek bellek baskısından söz ediliyor; bu mesajda sistem bellek görüntüsü veya swap ölçümü bulunmadığından burada bağımsız doğrulanmış ölçüm olarak kaydedilmiyor. Performans bulguları mevcut koşullardaki tek deneyin betimsel ölçümleridir; kontrollü tekrarlı performans ölçümü henüz yapılmadı.

### Kanıt dosyaları ve güncel aşamalar

- `scripts/score_main.py`: model seçimi, 50 çiftlik parçalar, tamamlanan parçaları tanıyarak devam etme ve parça bazlı puan/metrik kaydı.
- `scripts/summarize_main.py`: tamamlanan üç modelin dosya/token kontrolleri ve karşılaştırma özeti.
- `results/main/{fp16,q8,q4}/chunk_00.jsonl` … `chunk_19.jsonl`: her modelde 1.000 çiftin puanları.
- Aynı klasörlerde `chunk_00.metrics.json` … `chunk_19.metrics.json`: yükleme/ısınma ve parça kaynak ölçümleri, ortam ve yapılandırma bilgileri.
- `results/main/main_comparison.json`: ana deney sonuçları, kategori doğrulukları, karar geçişleri ve kaynak özeti.

| Aşama | Durum |
| --- | --- |
| Ortam, üç model ve ana/pilot veri hazırlığı | Tamamlandı |
| Üç modelde pilot ve karşılaştırma kontrolleri | Tamamlandı |
| Aşama 4 — Üç modelde 1.000 çiftlik ana deney | Tamamlandı |
| Ana veri genelinde dosya/token tutarlılığı | Tamamlandı |
| Süre ve bellek ölçümlerinin kaydı/özetlenmesi | Tamamlandı; ölçüm sınırları yukarıda |
| Genel karşılaştırmalar için kesin McNemar hesabı | İlk hesaplar doğrulandı |
| Güven aralıkları ve nihai istatistiksel raporlama | Bekliyor |
| Ayrıntılı hata analizi ve makale bulguları | Bekliyor |

**Sıradaki adım:** Eşleşmiş doğruluk farklarının güven aralıklarını ve çoklu karşılaştırma yaklaşımını belgelemek; istatistiksel raporu tamamlamak; karar değiştiren çiftlerde hata analizi yaparak bulguları makale düzenine taşımak. Ana deney puanlamasının tekrar çalıştırılması bu kayıt için gerekli olmadı.

Bu güncellemede yalnızca `ProjeMarkdown/MEVCUT_DURUM.md` sonuna Kayıt 005 eklendi. Önceki kayıtlar korunmuş; kod, veri, sonuç ve model dosyaları değiştirilmemiştir.


## Kayıt 006 — 28 Ekim 2026 — Ana deneyin ayrıntılı MLX peak ve süre dökümü

**Bu kaydın amacı:** Yeni bir deney çalıştırılmadı. Kullanıcının ayrıntı kontrolü talebi üzerine Kayıt 005’te özetlenen ana deneyin parça ve oturum ölçümleri açık tablolarla eklendi. Kayıt 005’te MLX peak en yüksek değerleri zaten vardı; 60 parçanın tek tek dökümü ve tüm oturum yükleme/ısınma değerleri yoktu. “Tüm ayrıntılar var” ifadesi bu nedenle eksiksiz bir terminal dökümü olarak anlaşılmamalı. Önceki kayıtlar değiştirilmedi.

### MLX peak — üç modelin açık karşılaştırması

| Ölçüm | FP16 | Q8 | Q4 |
| --- | ---: | ---: | ---: |
| **MLX peak — en yüksek parça değeri (GB)** | 3.416117920 | 2.068173544 | 1.236718312 |
| MLX peak — iki ondalık gösterim (GB) | 3.42 | 2.07 | 1.24 |
| Toplam puanlama süresi (sn) | 123.038089 | 97.884444 | 95.202621 |

Tablolarda nokta ondalık ayırıcıdır; terminalle doğrudan karşılaştırılabilir. MLX peak toplam sistem RAM’i değildir; MLX ayırıcısının raporladığı tepe değerdir. GB = 10⁹ bayt. Süreler yükleme ve ısınma hariç parça puanlama süreleridir; ayrıntılı ölçüm kapsamı Kayıt 005’te korunmuştur.

### Oturumlar — model yükleme ve ısınma

Terminaldeki çalışma sırası FP16 için 1 + 19, Q8 için 1 + 19, Q4 için 20 parçadır. Aşağıdaki değerler bu oturumların ilk parça metriklerinden alınmıştır; aynı oturuma ait tekrarlanan alanlar yeniden toplanmamıştır.

| Model | Oturum | Terminaldeki komut | Parçalar | Yükleme (sn) | Isınma (sn) |
| --- | ---: | --- | --- | ---: | ---: |
| FP16 | 1 | `python scripts/score_main.py --model fp16 --max-chunks 1` | 01–01 | 2.374633 | 0.588512 |
| FP16 | 2 | `python scripts/score_main.py --model fp16 --max-chunks 19` | 02–20 | 2.332323 | 0.451830 |
| Q8 | 1 | `python scripts/score_main.py --model q8 --max-chunks 1` | 01–01 | 1.803497 | 0.323872 |
| Q8 | 2 | `python scripts/score_main.py --model q8 --max-chunks 19` | 02–20 | 1.340210 | 0.308231 |
| Q4 | 1 | `python scripts/score_main.py --model q4 --max-chunks 20` | 01–20 | 1.137677 | 0.083085 |

Üç modelin yükleme mesajlarında BOS=1 ve yöntem `mean_logp_float32_bos_context_no_eos_v1` bildirildi. FP16/Q8’in ilk oturumundan sonra 19, son oturumlarından sonra 0 parça kaldı. Q4 oturumundan sonra 0 parça kaldı. FP16 ve Q8’in ikinci oturumları 1/20 parçanın önceden tamamlandığını tanıdı; tamamlanan ilk parça yeniden puanlanmadı.

### FP16 — 20 parçanın tamamı

| Parça (terminal sırası) | Çift | Doğru | Yanlış | Eşit | Puanlama (sn) | **MLX peak (GB)** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01/20 | 50 | 42 | 8 | 0 | 6.039261 | 3.405533796 |
| 02/20 | 50 | 46 | 4 | 0 | 6.014656 | 3.412038284 |
| 03/20 | 50 | 44 | 6 | 0 | 6.107158 | 3.394507304 |
| 04/20 | 50 | 45 | 5 | 0 | 6.109001 | 3.388297728 |
| 05/20 | 50 | 44 | 6 | 0 | 6.001169 | 3.377713604 |
| 06/20 | 50 | 46 | 4 | 0 | 5.962579 | 3.388297728 |
| 07/20 | 50 | 49 | 1 | 0 | 6.061793 | 3.388297728 |
| 08/20 | 50 | 44 | 6 | 0 | 6.272844 | 3.388297728 |
| 09/20 | 50 | 47 | 3 | 0 | 6.180485 | 3.388297728 |
| 10/20 | 50 | 47 | 3 | 0 | 6.141356 | 3.380957656 |
| 11/20 | 50 | 46 | 4 | 0 | 6.599416 | 3.402568272 |
| 12/20 | 50 | 40 | 10 | 0 | 6.320230 | 3.416117920 |
| 13/20 | 50 | 40 | 10 | 0 | 6.239047 | 3.416117920 |
| 14/20 | 50 | 42 | 8 | 0 | 6.224128 | 3.388297728 |
| 15/20 | 50 | 42 | 8 | 0 | 6.356617 | 3.397751356 |
| 16/20 | 50 | 49 | 1 | 0 | 6.022309 | 3.402568272 |
| 17/20 | 50 | 50 | 0 | 0 | 6.075876 | 3.391541780 |
| 18/20 | 50 | 50 | 0 | 0 | 6.091625 | 3.402568272 |
| 19/20 | 50 | 50 | 0 | 0 | 6.140926 | 3.391541780 |
| 20/20 | 50 | 49 | 1 | 0 | 6.077613 | 3.394507304 |

**Toplam:** 912/1.000 doğru; puanlama 123.038089 sn; en yüksek MLX peak 3.416117920 GB.

### Q8 — 20 parçanın tamamı

| Parça (terminal sırası) | Çift | Doğru | Yanlış | Eşit | Puanlama (sn) | **MLX peak (GB)** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01/20 | 50 | 43 | 7 | 0 | 4.934231 | 2.036552364 |
| 02/20 | 50 | 46 | 4 | 0 | 4.822878 | 2.056393428 |
| 03/20 | 50 | 44 | 6 | 0 | 4.950228 | 2.004046448 |
| 04/20 | 50 | 45 | 5 | 0 | 4.965930 | 1.984205384 |
| 05/20 | 50 | 44 | 6 | 0 | 4.911094 | 1.952584204 |
| 06/20 | 50 | 46 | 4 | 0 | 4.843420 | 1.984205384 |
| 07/20 | 50 | 49 | 1 | 0 | 4.877297 | 1.984205384 |
| 08/20 | 50 | 44 | 6 | 0 | 4.898453 | 1.984205384 |
| 09/20 | 50 | 47 | 3 | 0 | 4.847555 | 1.984205384 |
| 10/20 | 50 | 47 | 3 | 0 | 4.953732 | 1.962807840 |
| 11/20 | 50 | 46 | 4 | 0 | 4.886363 | 2.026934936 |
| 12/20 | 50 | 41 | 9 | 0 | 4.874253 | 2.068173544 |
| 13/20 | 50 | 40 | 10 | 0 | 4.868384 | 2.068173544 |
| 14/20 | 50 | 43 | 7 | 0 | 4.891363 | 1.984205384 |
| 15/20 | 50 | 42 | 8 | 0 | 4.725428 | 2.013647492 |
| 16/20 | 50 | 49 | 1 | 0 | 4.938555 | 2.026934936 |
| 17/20 | 50 | 50 | 0 | 0 | 4.943448 | 1.994429020 |
| 18/20 | 50 | 50 | 0 | 0 | 4.895766 | 2.026934936 |
| 19/20 | 50 | 50 | 0 | 0 | 4.881275 | 1.994429020 |
| 20/20 | 50 | 49 | 1 | 0 | 4.974789 | 2.004046448 |

**Toplam:** 915/1.000 doğru; puanlama 97.884444 sn; en yüksek MLX peak 2.068173544 GB.

### Q4 — 20 parçanın tamamı

| Parça (terminal sırası) | Çift | Doğru | Yanlış | Eşit | Puanlama (sn) | **MLX peak (GB)** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01/20 | 50 | 37 | 13 | 0 | 4.488683 | 1.205408428 |
| 02/20 | 50 | 41 | 9 | 0 | 4.784035 | 1.225052884 |
| 03/20 | 50 | 41 | 9 | 0 | 4.684206 | 1.171346032 |
| 04/20 | 50 | 44 | 6 | 0 | 4.826151 | 1.151865416 |
| 05/20 | 50 | 40 | 10 | 0 | 4.781706 | 1.120801292 |
| 06/20 | 50 | 45 | 5 | 0 | 4.732933 | 1.151865416 |
| 07/20 | 50 | 47 | 3 | 0 | 4.745318 | 1.151865416 |
| 08/20 | 50 | 45 | 5 | 0 | 4.763661 | 1.151865416 |
| 09/20 | 50 | 46 | 4 | 0 | 4.809017 | 1.151865416 |
| 10/20 | 50 | 46 | 4 | 0 | 4.760474 | 1.130828320 |
| 11/20 | 50 | 45 | 5 | 0 | 4.793096 | 1.193841304 |
| 12/20 | 50 | 38 | 12 | 0 | 4.810309 | 1.236718312 |
| 13/20 | 50 | 37 | 13 | 0 | 4.741715 | 1.236718312 |
| 14/20 | 50 | 42 | 8 | 0 | 4.848814 | 1.151865416 |
| 15/20 | 50 | 41 | 9 | 0 | 4.798112 | 1.180783236 |
| 16/20 | 50 | 50 | 0 | 0 | 4.639989 | 1.193841304 |
| 17/20 | 50 | 50 | 0 | 0 | 4.822024 | 1.161892444 |
| 18/20 | 50 | 50 | 0 | 0 | 4.814655 | 1.193841304 |
| 19/20 | 50 | 50 | 0 | 0 | 4.804803 | 1.161892444 |
| 20/20 | 50 | 49 | 1 | 0 | 4.752922 | 1.171346032 |

**Toplam:** 884/1.000 doğru; puanlama 95.202621 sn; en yüksek MLX peak 1.236718312 GB.

### Kapsam ve diğer karşılaştırmaların yeri

Kayıt 005’te genel doğruluk/doğru/yanlış/eşit sayıları, FP16’ya göre farklar, dört kategorinin sonuçları, karar geçişleri, ilk kesin McNemar hesapları, token/dosya kontrolleri ve kaynak ölçümlerinin kapsamı bulunuyor. Bu ek kayıt, o sonuçları değiştirmeden ana deneyin **60 parça satırını ve 5 oturumun yükleme/ısınma değerlerini** tamamlıyor. Kaynaklar `results/main/{fp16,q8,q4}/chunk_00..19.jsonl`, bunlara karşılık gelen `.metrics.json` dosyaları ve `results/main/main_comparison.json`.

Bu belge, gönderilen terminal metinlerinin birebir transkripti değildir: tekrarlanan komut istemleri/`clear` satırları ve önceki pilotların tüm token bazlı tanılama çıktıları burada tek tek yeniden basılmamıştır. Grafik eklenmemiştir. Swap/bellek baskısı açıklaması Kayıt 005’te kullanıcı gözlemi olarak, bağımsız ölçüm bulunmadığı belirtilerek kayıtlıdır.

Bu güncellemede yalnızca `ProjeMarkdown/MEVCUT_DURUM.md` sonuna Kayıt 006 eklendi. Eski kayıtlar aynen korundu; kod, veri, model veya sonuç dosyaları değiştirilmedi; deneyler yeniden çalıştırılmadı.
