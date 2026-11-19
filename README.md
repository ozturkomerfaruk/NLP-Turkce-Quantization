# Türkçe dilbilgisel yargılarda model nicemleme

*Weight Quantization and the Stability of Turkish Grammatical Judgments in EuroLLM and Qwen* makalesinin yeniden üretim paketi.

Bu araştırma paketi, EuroLLM-1.7B ve Qwen3-1.7B-Base modellerinin FP16, Q8 ve Q4 yapılandırmalarını aynı 3.662 TurBLiMP minimal çifti üzerinde karşılaştırır. Her çift için toplam ve ortalama token log-olasılığı, modelin veri kümesinde tercih edilen cümleyi seçip seçmediğini belirlemek üzere ayrı ayrı değerlendirilir.

| Model | Skor | FP16 | Q8 | Q4 |
| --- | --- | ---: | ---: | ---: |
| EuroLLM | Mean | %91,04 | %91,02 | %88,80 |
| EuroLLM | Total | %94,13 | %94,13 | %91,13 |
| Qwen | Mean | %82,55 | %82,44 | %79,46 |
| Qwen | Total | %86,40 | %86,43 | %83,18 |

Tablo, 3.662 çiftin eşit ağırlıklı doğruluğunu gösterir. Q8 doğruluğu, her iki model ve puanlama kuralında önceden belirlenen ±1 yüzde puanlık marj içinde FP16’ya eşdeğer bulundu. Q4 doğruluğu dört karşılaştırmada 2,24–3,22 yüzde puan düştü. Her iki bulgu 2.662 önceden puanlanmamış çiftte de gözlendi. Tekil tercih değişimleri, kategoriye eşit ağırlık veren Macro Accuracy ve belirsizlik aralıkları [istatistik tablolarında](results/revision_v2/statistics/) yer alır.

## Sonuçları yeniden üretme

```bash
git clone --single-branch --branch revision/major-revision-v2 \
  https://github.com/ozturkomerfaruk/NLP-Turkce-Quantization.git
cd NLP-Turkce-Quantization
```

- **Kayıtlı skorlardan analiz.** Python 3.12 ve R ile 444 kontrol noktasının SHA-256 değerlerini doğrulayın, dokuz istatistik tablosunu yeniden hesaplayın ve arşivle karşılaştırın. Bu yol macOS veya Linux üzerinde model ağırlığı ve GPU gerektirmeden çalışır.
- **Ağırlıklardan tam deney.** Apple Silicon ve Metal destekli MLX ortamında kaynak ağırlıkları sabit sürümlerden indirip altı yapılandırmayı oluşturun. Küçük doğrulama koşularının ardından 3.662 çifti yeniden puanlayın. Karşılaştırma komutu her model, hassasiyet ve skor kuralı için doğru sayısını, değişen tercihleri ve en büyük cümle skoru farkını verir.

Kurulum ve komutlar [yeniden üretim kılavuzunda](docs/reproducibility_v2.md). Makaledeki altı tablonun kaynakları [tablo eşlemesinde](docs/reproducibility_v2.md#manuscript-table-sources) listelenir. Kayıtlı skorlar, veri, model manifestleri ve deneyde kullanılan tokenizer ile ayar dosyaları Git’te yer alır. Model ağırlıkları, önbellekler ve yeni koşu çıktıları yerelde oluşturulur.

Makale için bu araştırma paketine atıf verirken `git rev-parse HEAD` ile aldığınız tam commit kimliğini kullanın.

## Araştırma kayıtları

| Kayıt | İçerik |
| --- | --- |
| [Veri manifesti](data/revision_v2/manifest.json) ve [veri kapsamı](docs/revision_scope_v2.md) | Veri sürümü, kategori sayıları ve ilk örnekleme ait çiftlerin işareti |
| [Analiz protokolü](docs/revision_analysis_protocol_v1.md) ve [istatistik yöntemi](results/revision_v2/statistics/methods.md) | Araştırma soruları, eşdeğerlik marjı, heterojenlik ve sağlamlık testleri |
| [EuroLLM](docs/eurollm_revision_build.md) ve [Qwen](docs/qwen_revision_build.md) | Sabit model sürümleri, dönüşüm komutları ve model dosyalarının SHA-256 kayıtları |
| [Revizyon modeli dosyaları](models/revision_v2/) | Makalede kullanılan kaynak, FP16, Q8 ve Q4 yapılandırmaları ve tokenizer’lar |
| [Skor arşivi](results/revision_v2/) ve [doğrulama kaydı](results/revision_v2/verification.json) | Altı yapılandırmanın 21.972 çift değerlendirmesi |
| [Sonuç raporu](results/revision_v2/statistics/report.md) ve [duyarlılık incelemesi](results/revision_v2/diagnostics/report.md) | İstatistik bulguları, yüklem filtresi ve tek koşulu kaynak ölçümleri |

İlk 1.000 çiftlik EuroLLM deneyinin [başlangıç denetimi](docs/baseline_audit_v2.md), [ana tabloları](results/analysis/) ve [keşifsel parçalanma analizi](results/fragmentation_analysis/fragmentation_report.md) tarihsel karşılaştırma için erişilebilir.

Özgün kod [MIT](LICENSE) lisanslıdır. Veri ve yukarı akış model bileşenlerinin lisansları [üçüncü taraf bildirimlerinde](THIRD_PARTY_NOTICES.md) yer alır.
