# Türkçe Dilbilgisel Yargılarda Model Nicemleme

EuroLLM-1.7B modelinin FP16, 8-bit (Q8) ve 4-bit (Q4) sürümlerini Türkçe dilbilgisel minimal çiftler üzerinde karşılaştıran deneyin kodları, verileri ve sonuçları.

Çalışma, model ağırlıklarını daha düşük bit genişliğiyle temsil etmenin Türkçe dilbilgisi ayrımlarına etkisini inceler. TurBLiMP veri setinden seçilen 1.000 çiftte Q8, FP16 referansına yakın doğruluk verirken Q4’te 2,8 yüzde puanlık düşüş görülmüştür. Q4’teki en büyük kategori farkı özne–yüklem uyumundadır.

## Deney

Kaynak model **EuroLLM-1.7B Base**’dir. Q8 ve Q4 sürümleri aynı FP16 modelden ayrı ayrı, `group_size=64` ve `affine` nicemleme ile hazırlanmıştır. Deney, Apple Silicon M2 ve 16 GB RAM bulunan macOS ortamında MLX ile yürütülmüştür.

Her minimal çift, dilbilgisel açıdan doğru bir cümle ile belirli bir dilbilgisi kuralını ihlal eden karşılığından oluşur. Modelin doğru cümleye daha yüksek ortalama token log-olasılığı vermesi doğru karar sayılır. BOS yalnızca bağlam olarak kullanılır; EOS puanlamaya eklenmez. Log-softmax ve toplamalar float32 ile hesaplanır. Eşit puanlar doğru kabul edilmez.

TurBLiMP’in dört kategorisinden, tekrarlar ayıklandıktan sonra `seed=42` ile 250’şer çift seçilmiştir. Üç model aynı 1.000 çift ve aynı puanlama protokolüyle değerlendirilmiştir.

## Sonuçlar

| Model | Doğru / toplam | Doğruluk | FP16’ya göre fark | Ağırlık boyutu |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 912 / 1.000 | %91,2 | — | 3,314 GB |
| Q8 | 915 / 1.000 | %91,5 | +0,3 yüzde puan | 1,761 GB |
| Q4 | 884 / 1.000 | %88,4 | −2,8 yüzde puan | 0,932 GB |

Ağırlık boyutları `.safetensors` dosyalarının toplamıdır; GB, 10⁹ bayt olarak hesaplanmıştır. Bunlar RAM tüketimini göstermez. FP16’ya göre ağırlık boyutu Q8’de yaklaşık %47, Q4’te %72 azalmıştır.

**Q8**, FP16’nın doğru karar verdiği çiftlerin tamamında doğru kalmış, FP16’nın yanlış değerlendirdiği üç çifti doğru değerlendirmiştir. Bu küçük fark istatistiksel olarak anlamlı değildir (iki yönlü kesin McNemar testi, Holm düzeltmeli *p* = 0,25). Sonuç, bu örneklemde Q8’in referansa yakın kaldığını gösterir; bir üstünlük veya eşdeğerlik kanıtı değildir.

**Q4**, FP16’nın doğru değerlendirdiği 42 çiftte yanlış, yanlış değerlendirdiği 14 çiftte doğru karar vermiştir. Net kayıp 28 çifttir. Fark istatistiksel olarak anlamlıdır (Holm düzeltmeli *p* ≈ 0,000469); doğruluk farkı için muhafazakâr en az %95 kapsamalı eşleştirilmiş güven aralığı −5,15 ile −0,42 yüzde puandır.

### Kategorilere göre doğruluk

| Kategori | Çift sayısı | FP16 | Q8 | Q4 | Q4 − FP16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Özne–yüklem uyumu | 250 | %88,4 | %88,8 | %81,2 | −7,2 yüzde puan |
| Geçişli fiillerde argüman yapısı | 250 | %93,2 | %93,2 | %91,6 | −1,6 yüzde puan |
| Adlaştırma | 250 | %84,0 | %84,8 | %81,2 | −2,8 yüzde puan |
| Düzensiz biçimler | 250 | %99,2 | %99,2 | %99,6 | +0,4 yüzde puan |

Kategori sonuçları betimleyicidir. Q4’teki kayıp dilbilgisel kategorilere eşit dağılmamıştır: toplam 28 çiftlik net kaybın 18’i özne–yüklem uyumundan gelmektedir. Ayrıca Q4 ile FP16 arasında kararın değiştiği 56 çiftin 46’sı, FP16’nın iki cümle arasında en küçük puan farkını verdiği dörtte birlik gruptadır. Karar değişimleri ağırlıklı olarak referans modelin de ayrımı daha zayıf yaptığı çiftlerde toplanmıştır.

Tam sayısal sonuçlar: [ana istatistikler](results/analysis/main_statistics.json), [kategori tablosu](results/analysis/category_results.csv) ve [çift düzeyindeki puanlar](results/analysis/per_pair_scores.csv).

## Analizleri tekrar çalıştırma

Depoda seçilmiş veri, ham CSV dosyaları ve üç modelin kayıtlı puanları bulunur. Ana istatistikleri bu puanlardan yeniden hesaplamak için model ağırlıkları veya MLX kurulumu gerekmez. Depo kökünde:

```bash
python3 scripts/analyze_main.py
```

Ana analiz yalnızca Python standart kütüphanesini kullanır. Komut `results/analysis/` altındaki dosyaları yeniden yazar. Ağırlık dosyaları bulunmadığında yeniden üretilen rapordaki ağırlık boyutu alanları `null` olur; depodaki kayıtlı rapor deney sırasında ölçülen boyutları içerir.

### Veri örneklemini yeniden oluşturma

Veri hazırlama betikleri de yalnızca standart kütüphaneyi kullanır. Ham dosyalar yoksa internetten indirilir; varsa mevcut dosyalar kullanılır.

```bash
python3 scripts/prepare_dataset.py
python3 scripts/finalize_pilot_dataset.py
shasum -a 256 data/processed/turblimp_1000.jsonl
```

Beklenen SHA-256:

```text
bcb2c7f25475f8290d1511d742e1e9f8f515fd29df11dd3e3f64f2f53bcb4835
```

TurBLiMP kaynak revizyonu `297de13fb7a0ce524fe32e8b175c6b5255d66960`’tır. Veri seçimi ve dosya hash’leri [manifestte](data/processed/manifest.json) kayıtlıdır.

### Model puanlamasını yeniden çalıştırma

Referans ortam Python 3.12.15, MLX 0.32.4 ve MLX-LM 0.32.2’dir. `requirements.txt` yerel deney ortamındaki paket sürümlerini kaydeder; temiz ortam kurulumu ayrıca doğrulanmamıştır.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Model ağırlıkları depoya dahil değildir. Puanlama için ağırlıkların ilgili yapılandırma ve tokenizer dosyalarıyla birlikte `models/eurollm-fp16/`, `models/eurollm-q8/` ve `models/eurollm-q4/` konumlarında bulunması gerekir. Betik her çalıştırmada tamamlanmış parçaları denetler ve kaldığı yerden devam eder; her parça 50 çift içerir.

```bash
python scripts/score_main.py --model fp16 --max-chunks 20
python scripts/score_main.py --model q8 --max-chunks 20
python scripts/score_main.py --model q4 --max-chunks 20
python scripts/summarize_main.py
python scripts/analyze_main.py
```

Kurulum notlarında kaynak model indirme revizyonu `refs/pr/5` olarak geçer. Değişmez model commit’i ve tam dönüştürme komutları eksiksiz arşivlenmediği için, model hazırlama aşamasının birebir tekrarlanabilirliği sınırlıdır. Kayıtlı puanlardan istatistiksel analizlerin yeniden üretilmesi ise ağırlıkların indirilmesine bağlı değildir.

## Dosya düzeni

| Konum | İçerik |
| --- | --- |
| `scripts/` | Veri hazırlama, pilot ve ana puanlama, özetleme ve analiz betikleri |
| `data/` | Ham CSV dosyaları, seçilmiş JSONL örneklemi ve manifest |
| `models/` | Model yapılandırmaları ve tokenizer dosyaları; ağırlıklar hariç |
| `results/main/` | Üç modelin parça bazlı puanları, metrikleri ve karşılaştırması |
| `results/analysis/` | Ana istatistikler, kategori sonuçları ve çift düzeyindeki puanlar |
| `results/deep_analysis/` | Ek analizler, denetimler ve yeniden çalıştırma kayıtları |
| `results/fragmentation_analysis/` | Alt sözcük parçalanmasına ilişkin keşifsel analizler |
| `tests/` | Analiz doğrulama testleri |
| `ProjeMarkdown/` | Yöntem, süreç ve çalışma notları |
| `outputs/` | Makale dosyaları, şekiller ve tekrar paketleri |

[7 Kasım 2026 tekrar paketinin açıklaması](outputs/Socioinformatics_Final_2026_11_07/replication_package/README.md), ek materyallerin kapsamını ve kullanımını açıklar.

## Kapsam ve sınırlılıklar

Bu deney tek bir modelin üç ağırlık hassasiyetini, TurBLiMP’ten seçilmiş dört kategoride karşılaştırır. Sonuçlar Türkçenin tüm dilbilgisel özelliklerine, başka modellere veya serbest metin üretim kalitesine doğrudan genellenemez. İstatistiksel belirsizlik, sabit model ve veri oluşturma koşulları altında değerlendirilmiştir. Ana karşılaştırmalar FP16–Q8 ve FP16–Q4’tür; iki test için Holm düzeltmesi uygulanmıştır.

Raporlanan doğruluklar ortalama token log-olasılığına dayanır. Toplam log-olasılığı kullanan duyarlılık analizi farklı doğruluklar verir; ana sonuçlarla aynı puanlama ölçütü gibi ele alınmamalıdır. Alt sözcük parçalanması analizleri keşifseldir. Süre ve bellek ölçümleri kullanılan donanıma ve çalışma koşullarına bağlıdır.

## Kaynaklar

- Başar, Padovani, Jumelet ve Bisazza (2025), **TurBLiMP**: [makale](https://doi.org/10.18653/v1/2025.emnlp-main.834), [veri deposu](https://github.com/ezgibasar/TurBLiMP).
- Martins ve diğerleri (2024), **EuroLLM**: [makale](https://doi.org/10.48550/arXiv.2409.16235), [model kartı](https://huggingface.co/utter-project/EuroLLM-1.7B).

TurBLiMP verileri için kaynak deponun belirttiği CC BY 4.0 koşulları geçerlidir. Buradaki örneklem, kaynak veriden tekrarlar ayıklanarak ve örnekleme yapılarak oluşturulmuştur. Kaynak veri ve modelin lisans koşulları kendi yayınlarından kontrol edilmelidir. Bu depodaki özgün kod için ayrıca bir lisans tanımlanmamıştır.
