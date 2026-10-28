# Projenin Tüm Süreci

**Son güncelleme:** 27 Ekim 2026  
**Proje:** Bir Çok Dilli Dil Modelinde Nicemlemenin Türkçe Morfosentaktik Yeterliliğe Etkisinin Deneysel İncelenmesi

## 1. Amaç ve çalışma kapsamı

Araştırma sorusu: **EuroLLM-1.7B Base modelinde 8-bit ve 4-bit ağırlık nicemlemesi, seçilmiş Türkçe morfosentaktik minimal çiftlerde doğruluğu ve cümle tercihlerini nasıl değiştiriyor?**

Çalışma, aynı çok dilli modelin FP16 referansı ile Q8 ve Q4 sürümlerini karşılaştırıyor. TurBLiMP'teki her kayıtta dilbilgisel açıdan doğru ve hatalı bir cümle bulunuyor. Modelin bu iki cümleye verdiği puanlar karşılaştırılarak dilbilgisel tercihi ölçülecek; ardından nicemleme ile değişen kararlar incelenecek.

Kapsam bir model, Türkçe, dört kategori, 1.000 çift ve MLX afin ağırlık nicemlemesiyle sınırlı. Eğitim, ince ayar, QAT, başka modeller/diller ve GPTQ/AWQ gibi başka nicemleme yöntemleri bu deneyin dışında. Ücretli API gerektirmeyen yerel bir çalışma düzeni benimsenmiş.

## 2. Ortam kurulumu — tamamlandı

Paylaşılan kurulum kayıtlarına göre MacBook Pro M2 / 16 GB RAM üzerinde VS Code, Python 3.12 ve `.venv` kullanılıyor. MLX-LM 0.32.2 kurulmuş; Apple Metal desteği ve `mlx_lm.convert` aracı doğrulanmış. Proje konumu `~/Desktop/turkce-quantization/`.

Bu güncellemede kurulum tekrarlanmadı. Önceki çalışma testleri kullanıcının kurulum notlarına dayanıyor; veri ve model dosyaları ise yerel olarak incelendi.

## 3. Kaynak model ve nicemlenmiş sürümler — tamamlandı

Hugging Face üzerindeki `utter-project/EuroLLM-1.7B` Base modeli, safetensors ağırlıklarını içeren `refs/pr/5` revizyonundan indirilmiş. Kaynak dosyalar `models/eurollm-source/` altında tutuluyor. Yerel yapılandırma Llama mimarisini, 24 katmanı ve 4.096 token bağlam uzunluğunu gösteriyor.

Kaynak modelden MLX uyumlu FP16 sürümü hazırlanmış; ardından aynı FP16 kaynaktan Q8 ve Q4 ayrı ayrı üretilmiş. İki nicemleme koşulunda da `group_size=64` ve `affine` yöntemi kullanılıyor. Böylece karşılaştırmanın temel değişkeni ağırlık bit genişliği oluyor.

| Sürüm | Konum | Ağırlık dosyalarının toplamı | Görev |
| --- | --- | ---: | --- |
| FP16 | `models/eurollm-fp16/` | Yaklaşık 3,31 GB | Nicemlenmemiş referans |
| Q8 | `models/eurollm-q8/` | Yaklaşık 1,76 GB | 8-bit karşılaştırma koşulu |
| Q4 | `models/eurollm-q4/` | Yaklaşık 0,93 GB | 4-bit karşılaştırma koşulu |

Tablodaki boyutlar yerel `.safetensors` dosyalarının toplamıdır; GB için 10⁹ bayt kullanılmıştır. Kurulum notlarındaki yaklaşık 3,1 GB / 1,6 GB / 889 MB gösterimleriyle birim ve yuvarlama farkı vardır. Bunlar RAM tüketimi değildir. FP16 ağırlık başlıkları `F16`; Q8/Q4 yapılandırmaları belirtilen nicemleme ayarlarını doğruluyor.

Önceki kayıtlara göre üç model de Türkçe metin üretmiş. Aynı başlangıç metniyle tek seferlik 20 token üretim testinde FP16/Q8/Q4 için sırasıyla 31,799 / 56,937 / 99,440 token/sn ve 3,352 / 1,813 / 0,984 GB tepe bellek kaydedilmiş. Bu sonuçlar çalışabilirlik ve ön kaynak ölçümü niteliğinde; araştırmanın dilbilgisel başarı veya puanlama performansı sonuçları henüz değil.

## 4. TurBLiMP veri hazırlama — tamamlandı

`scripts/prepare_dataset.py` ile dört kategoriye ait CSV dosyaları indirilmiş ve sabit deney örneklemi oluşturulmuş. Gerçek indirme kaynağı TurBLiMP GitHub deposunun şu commit'idir:

```text
297de13fb7a0ce524fe32e8b175c6b5255d66960
```

| Kategori | İncelenen özellik | Ham dosya | Seçilen çift |
| --- | --- | --- | ---: |
| Subject Agreement | Özne–yüklem kişi/sayı uyumu | `augmented_subject_verb_agreement.csv` | 250 |
| Argument Structure Transitive | Geçişli fiillerle durum eki kullanımı | `augmented_argument_structure_transitive.csv` | 250 |
| Nominalization | Adlaştırma eki seçimi | `augmented_nominalization.csv` | 250 |
| Irregular Forms | Geniş zaman ekinin biçim seçimi | `augmented_irregular_forms.csv` | 250 |
| **Toplam** | | | **1.000** |

Hazırlama akışı:

1. Ham CSV dosyalarını `data/raw/` altında saklama ve noktalı virgül ayıracıyla okuma.
2. Cümlelerin başındaki/sonundaki boşlukları temizleme; boş veya doğru/hatalı cümlesi aynı kayıtları eleme.
3. `(doğru cümle, hatalı cümle)` ikilisine göre kategori içi ve kategoriler arası tekrarları kontrol etme.
4. Sabit kategori sırası ve `random.Random(42)` ile her kategoriden 250 çift seçme.
5. Kategori, kaynak satırı ve dilbilimsel alanlarla birlikte `data/processed/turblimp_1000.jsonl` dosyasına yazma; SHA-256 hesaplama.

Yerel kontrolde her ham dosyada 1.000 kayıt bulundu. Kategori içi tekrarlar çıkarıldığında sırasıyla 934, 835, 945 ve 948 uygun çift kaldı. Geçersiz kayıt veya kategoriler arası tekrar bulunmadı. Son örneklemde **1.000 benzersiz çift, 1.000 benzersiz kimlik ve toplam 2.000 cümle girdisi** doğrulandı. `seed=42` seçimi dosyadaki kayıt ve sırayla eşleşti.

İşlenmiş dosyanın SHA-256 değeri:

```text
bcb2c7f25475f8290d1511d742e1e9f8f515fd29df11dd3e3f64f2f53bcb4835
```

**Manifest ve pilot hazırlığı tamamlandı.** `scripts/finalize_pilot_dataset.py`, ana veri setinin beklenen hash değerini, 1.000 kayıt/benzersiz kimlik koşulunu ve kategori başına 250 çift bulunduğunu kontrol ediyor. `pilot_seed=42` ile her kategoriden 5 çift seçerek `data/processed/turblimp_pilot_20.jsonl` ve `data/processed/manifest.json` dosyalarını oluşturuyor; ana veri setini değiştirmiyor.

Manifest kaynak revizyonunu, ana veri ve pilot seed değerlerini, kategori sırasını/sayılarını, ham dosya hash değerlerini ve ana/pilot veri dosyalarının kayıt sayısı ile hash değerlerini içeriyor. Temizleme istatistikleri önceki bölümde belgelenmiş; manifestte ayrı alan olarak yer almıyor. Kaynak revizyonu kaydı tek başına önceden indirilmiş ham dosyanın kökenini kanıtlamaz; hash kayıtları kullanılan dosyaları tanımlar.

Pilot dosyasının bu güncellemede yeniden doğrulanan SHA-256 değeri:

```text
420407b129d0e997680a50ab11ff49319e933c1b82b07068042b5d1249239f27
```

## 5. Cümle puanlama ve 20 çiftlik pilot — FP16 tamamlandı, Q8/Q4 bekliyor

Amaç, ana deneye geçmeden puanlama yönteminin doğru çalıştığını ve bilgisayardaki süre/bellek maliyetini görmek. Pilot veri seti kategori başına 5 çiftle hazırlandı; aynı 20 çift üç modelde kullanılacak. FP16 değerlendirmesi tamamlandı. Sonuçlar paylaşılan terminal çıktısı ve yerel sonuç dosyalarından doğrulandı; bu dokümantasyon güncellemesinde model yeniden çalıştırılmadı.

### Uygulanan FP16 puanlama yöntemi

Model metin üretmiyor; her cümle token’ının önceki token’lar verildiğindeki koşullu log-olasılığı hesaplanıyor. `scripts/score_pilot.py` içindeki mevcut kurallar:

- `add_special_tokens=False`; BOS (`<s>`, ID `1`) elle başa ekleniyor. BOS puanlanmıyor, EOS eklenmiyor.
- Input dizisi son token hariç, target dizisi BOS hariç tutuluyor; ilk cümle token’ı BOS koşuluyla puanlanıyor. Batch boyutu `1`.
- FP16 model logits çıktısı FP32’ye çevrilerek log-softmax hesaplanıyor ve target token olasılıkları seçiliyor.
- Toplam log-olasılık ile token sayısına bölünmüş ortalama hesaplanıyor. Ana karar puanı **tüm cümlenin token başına ortalama log-olasılığı**; yalnızca kritik bölge puanlanmıyor.
- `delta = good_score - bad_score`: pozitif fark doğru, negatif fark yanlış karar. Tam sıfır fark eşitlik sayılıyor; başarıya dahil edilmiyor. FP16 pilotunda eşitlik yok.
- Token sayıları farklıysa konumsal karşılaştırma atlanıyor. Sayılar eşit olsa da konumsal tablo dilbilimsel token hizalaması sayılmıyor.

İlk çiftte input/target boyutları `(1, 11)`, FP16 logits boyutu `(1, 11, 128000)`, log-olasılık veri tipi FP32. Ayrı hesaplanan ilk çift puanları ile `score_sentence` fonksiyonu `1e-4` toleransında uyuşmuş; fonksiyon doğrulaması terminalde başarılı.

### FP16 pilot sonuçları

| Kategori | Doğru / Toplam | Doğruluk |
| --- | ---: | ---: |
| Subject Agreement | 2/5 | %40 |
| Argument Structure Transitive | 3/5 | %60 |
| Nominalization | 5/5 | %100 |
| Irregular Forms | 5/5 | %100 |
| **Genel** | **15/20** | **%75** |

Toplam 15 doğru, 5 yanlış, 0 eşit puan. İlk çift `subject_agreement:0066` için “Ah sen ne kadar güzel resim yapıyorsun.” doğru cümlesinin toplam puanı `-50.842498779296875`, ortalaması `-4.622045516967773`; “Ah o ne kadar güzel resim yapıyorsun.” hatalı cümlesinin toplamı `-50.2364387512207`, ortalaması `-4.566948890686035`. İki cümle de 11 token; ortalama farkı `-0.055097` ve modelin kararı yanlış.

Kaydedilmiş çıktılar:

- `results/pilot/fp16_pilot_20.jsonl`: pilot veriyle kimlik ve sıra bakımından eşleşen 20 kayıt. Her kayıtta çift kimliği, kategori, model, veri hash değeri, puanlama yöntemi, iki ortalama puan, token sayıları, fark ve karar var. Toplam log-olasılıklar terminalde hesaplanıyor; mevcut JSONL bunları içermiyor.
- `results/pilot/fp16_summary.json`: model/veri yolları, veri hash değeri, puanlama ve tokenlaştırma kuralları, ortam sürümleri, toplam/doğru/başarısız/eşit sayıları ve doğruluk. Ortam Python 3.12.15, MLX 0.32.4, MLX-LM 0.32.2 olarak kayıtlı. Süre ve bellek ölçümleri yok.

**%75 yalnızca 20 çiftlik FP16 pilotunun doğruluğudur.** Ana deney sonucu veya nicemlemenin etkisine ilişkin bulgu olarak yorumlanmayacak.

### Kalan pilot işleri ve komutların mevcut durumu

1. Mevcut FP16 sonuçlarını koruyarak aynı 20 çifti Q8 ile değerlendirmek.
2. Aynı veri ve puanlama kurallarıyla Q4 pilotunu tamamlamak.
3. Üç model arasında tokenlaştırma tutarlılığını, genel/kategori doğruluğunu ve FP16’ya göre karar değişimlerini karşılaştırmak.
4. Yükleme, ısınma ve puanlama süreleri ile bellek ölçümlerinin kapsamını belirleyip ölçümleri doğrulamak; ardından ana deneye geçmek.

Mevcut FP16 çalıştırma komutu:

```bash
python scripts/score_pilot.py
```

Betikte model yolu FP16’ya sabit. Paylaşılan sonraki aşama planındaki `--model fp16`, `--model q8` ve `--model q4` seçenekleri henüz uygulanmamış; bu seçenekler mevcut betikte model seçmez. `scripts/summarize_pilot.py` de henüz bulunmuyor. Otomatik ısınma/ölçüm akışı ve kesinti sonrası devam desteği planlanan işlerdir. Mevcut betik 20 çiftin sonucunu bellekte topladıktan sonra dosyalara yazar; her çiftte diske yazma veya tamamlananları atlama yapmaz. Yeniden çalıştırma mevcut FP16 sonuç dosyalarının üzerine yazar.

Planlanan `fp16.jsonl`, `fp16.metrics.json`, `q8.jsonl`, `q8.metrics.json`, `q4.jsonl` ve `q4.metrics.json` adları mevcut çıktılar değildir. Doğrulanan FP16 çıktıları yukarıdaki `fp16_pilot_20.jsonl` ve `fp16_summary.json` dosyalarıdır. Bu güncellemede bu planı uygulayacak kod eklenmedi veya değiştirilmedi.

Pilot iş yükü üç modelde **20 çift × 2 cümle × 3 model = 120 cümle puanlaması**. Tamamlanan FP16 pilotu 40 cümlelik değerlendirmedir; betiğin ilk çift için yaptığı ek tanılama hesapları bu sayıya dahil değildir.

## 6. Ana deney — bekliyor

Pilot doğrulandıktan sonra sabit 1.000 çiftin tamamı aynı puanlama kurallarıyla FP16, Q8 ve Q4 üzerinde değerlendirilecek:

**1.000 çift × 2 cümle × 3 model = 6.000 cümle puanlaması.**

Planlanan çalışma düzeni aynı anda tek model yüklemek, `batch_size=1` kullanmak ve model değiştirirken önceki süreci kapatmak. Ana değerlendirme 50 çiftlik parçalarda kaydedilerek kesinti sonrasında devam edebilir olmalı. Isınma, bellek baskısı ve arka plan yükünün ölçümlere etkisi gözetilmeli; yükleme, ısınma ve puanlama sürelerinin sınırları önceden belirlenmeli.

Üç model için ayrı veri seti oluşturulmayacak. Puanlama süresi ve tepe bellek bu deneyde ayrıca ölçülecek. Pilot süreleri, ana deneyin yaklaşık süresini öngörmek için kullanılabilir; şu anda toplam süreye ilişkin doğrulanmış bir tahmin yok. Mevcut FP16 pilot çıktıları süre/bellek ölçümü içermediğinden bu tahmin için henüz kullanılabilir ölçüm bulunmuyor.

## 7. İstatistiksel analiz ve raporlama — bekliyor

Kaydedilmiş puanlar üzerinden genel ve kategori bazında doğruluk, Q8/Q4'ün FP16'ya göre yüzde puan farkları, doğru→yanlış ve yanlış→doğru karar geçişleri incelenecek. Doğru–hatalı cümle puan farkları, özellikle FP16'da birbirine yakın puanlanan çiftlerin nicemleme sonrası değişip değişmediğini değerlendirmek için kullanılacak.

Karşılaştırma aynı çiftler üzerinde olduğundan doğruluk farklarına eşleştirilmiş güven aralıkları eklenecek. Kullanılacak istatistiksel yöntem ve varsa çoklu karşılaştırma yaklaşımı analizden önce belgelenecek. Karar değiştiren yaklaşık 20 çift elle incelenecek; yeterli sayıda değişen çift yoksa mevcut sayı raporlanacak. Dosya boyutu, puanlama süresi ve tepe bellek, başarı değişimiyle birlikte sunulacak.

Bulgular yalnızca bu model, nicemleme yöntemi ve seçilen TurBLiMP örneklemi için yorumlanacak. Kategori başına 250 çift olması küçük farkların güvenilir biçimde saptanacağını garanti etmez. Dört kategori üzerindeki minimal çift başarısı modelin tüm Türkçe yeteneğine genellenmeyecek.

## 8. Bugün gelinen nokta ve dosya düzeni

```text
turkce-quantization/
├── .venv/
├── models/
│   ├── eurollm-source/
│   ├── eurollm-fp16/
│   ├── eurollm-q8/
│   └── eurollm-q4/
├── scripts/
│   ├── prepare_dataset.py
│   ├── finalize_pilot_dataset.py
│   └── score_pilot.py
├── data/
│   ├── raw/                         # Dört kaynak CSV
│   └── processed/
│       ├── turblimp_1000.jsonl
│       ├── turblimp_pilot_20.jsonl
│       └── manifest.json
├── results/
│   └── pilot/
│       ├── fp16_pilot_20.jsonl
│       └── fp16_summary.json
└── ProjeMarkdown/
    ├── MEVCUT_DURUM.md               # Güncel durum ve kontroller
    └── TUM_SUREC.md                  # Tamamlanan ve planlanan süreç
```

Model hazırlığı, veri hazırlığı, manifest ve FP16 pilotu tamamlandı. Sıradaki adım Q8 pilotu; ardından Q4, üç modelin karşılaştırılması, süre/bellek doğrulaması, ana deney ve analiz gelecek. Bu dokümantasyon güncellemesi kapsamında yalnızca `ProjeMarkdown/` altındaki mevcut iki Markdown dosyası düzenlendi; klasör yapısı, deney kodu, veri, sonuçlar ve modeller değiştirilmedi.

Güncel durumun kısa ve somut kaydı: [MEVCUT_DURUM.md](MEVCUT_DURUM.md).
