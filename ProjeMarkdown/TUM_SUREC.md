# Projenin Tüm Süreci

**Son güncelleme:** 25 Ekim 2026  
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

**Kalan belgeleme:** `manifest.json` henüz oluşturulmamış. Kaynak commit, seed, kategori sırası/sayıları, temizleme istatistikleri ve ham/işlenmiş dosya hash değerleri bu dosyada bir araya getirilmeli. Kod mevcut ham dosyayı yeniden indirmeden kullandığından, kaynak revizyonunun kodda yazması tek başına önceden indirilmiş dosyanın kökenini kanıtlamaz; hash kayıtları bu nedenle değerlidir. Manifest eksikliği mevcut veri setinin puanlanmasına engel değil.

## 5. Cümle puanlama ve 20 çiftlik pilot — sıradaki aşama

Amaç, ana deneye geçmeden puanlama yönteminin doğru çalıştığını ve bilgisayardaki süre/bellek maliyetini görmek. Önerilen pilot, hazırlanmış örneklemden kategori başına 5 çift seçilerek oluşturulacak; aynı 20 çift üç modelde kullanılacak. Henüz pilot seçimi, değerlendirme betiği veya sonuç dosyası yok.

Planlanan ölçümde model metin üretmeyecek. Cümlenin her token'ının önceki token'lar verildiğindeki koşullu log-olasılığı hesaplanacak. Ana puan **token başına ortalama log-olasılık**; uzunluk etkisini ayrıca incelemek için toplam log-olasılık ve puanlanan token sayısı da saklanacak. Doğru cümle daha yüksek ana puan alırsa çift doğru değerlendirilmiş sayılacak.

Pilot sırasında şu kararlar netleştirilmeli: BOS/EOS ve özel token kullanımı, ilk token'ın nasıl ele alınacağı, tokenizer davranışı, puanlanan token sayısı, eşit puan kuralı ve sayısal doğrulama yöntemi. Bu kurallar üç modelde aynı uygulanmalı ve ana deneyden önce sabitlenmeli.

Pilot iş yükü **20 çift × 2 cümle × 3 model = 120 cümle puanlaması**. Sonuç kayıtları en az çift kimliği, kategori, model, iki cümlenin toplam/ortalama puanları, token sayıları, karar ve puan farkını içermeli. Süre ve tepe bellek ölçümünün kapsamı da açıkça kaydedilmeli. Bunlar henüz uygulanmış özellikler değil, yapılacak işlerin tanımıdır.

## 6. Ana deney — bekliyor

Pilot doğrulandıktan sonra sabit 1.000 çiftin tamamı aynı puanlama kurallarıyla FP16, Q8 ve Q4 üzerinde değerlendirilecek:

**1.000 çift × 2 cümle × 3 model = 6.000 cümle puanlaması.**

Planlanan çalışma düzeni aynı anda tek model yüklemek, `batch_size=1` kullanmak ve model değiştirirken önceki süreci kapatmak. Ana değerlendirme 50 çiftlik parçalarda kaydedilerek kesinti sonrasında devam edebilir olmalı. Isınma, bellek baskısı ve arka plan yükünün ölçümlere etkisi gözetilmeli; yükleme, ısınma ve puanlama sürelerinin sınırları önceden belirlenmeli.

Üç model için ayrı veri seti oluşturulmayacak. Puanlama süresi ve tepe bellek bu deneyde ayrıca ölçülecek. Pilot süreleri, ana deneyin yaklaşık süresini öngörmek için kullanılabilir; şu anda toplam süreye ilişkin doğrulanmış bir tahmin yok.

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
│   └── prepare_dataset.py
├── data/
│   ├── raw/                         # Dört kaynak CSV
│   └── processed/
│       └── turblimp_1000.jsonl
└── ProjeMarkdown/
    ├── MEVCUT_DURUM.md               # Güncel durum ve kontroller
    └── TUM_SUREC.md                  # Tamamlanan ve planlanan süreç
```

Model hazırlığı ve veri hazırlığı tamamlandı. Öncelik tekrarlanabilirlik kaydını tamamlamak ve pilot puanlamayı doğrulamak; ardından ana deney ve analiz gelecek. Bu dokümantasyon güncellemesi kapsamında yalnızca `ProjeMarkdown/` altındaki iki Markdown dosyası oluşturuldu; deney kodu, veri ve modeller değiştirilmedi.

Güncel durumun kısa ve somut kaydı: [MEVCUT_DURUM.md](MEVCUT_DURUM.md).
