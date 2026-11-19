# Revizyon deneyinin istatistik yöntemi

[Önceden sabitlenen protokol](../../../docs/revision_analysis_protocol_v1.md) ve [skor doğrulama kaydı](../verification.json) analiz girdilerini tanımlar. [Veri hazırlama betiği](../../../scripts/prepare_revision_analysis.py) 444 kontrol noktasının SHA-256 değerlerini doğrular, çift kimliklerini ve kategori üyeliklerini eşleştirir ve her model ailesi için üç hassasiyet düzeyinin token dizilerini karşılaştırır. R betiği doğrulanmış tek bir satırın üzerinde bir çiftin tüm konfigürasyon ve skor kararlarını okur.

## Ölçütler ve testler

- **RQ1.** Her konfigürasyon için Micro ve Macro Accuracy, her kuantizasyon düzeyi için eşleştirilmiş FP16 farkı, %95 aralıklar ve iki yönlü karar geçişleri. McNemar kesin p değerleri ayrı, ikincil ve düzeltilmemiş kontrollerdir.
- **RQ2.** Aynı çiftte Qwen FP16 farkı eksi EuroLLM FP16 farkı. Pozitif değer, Qwen’in doğruluk farkının daha yüksek olduğunu gösterir.
- **RQ3.** Q4 eksi FP16 çift farkları için dört kategori ortalamasını karşılaştıran CR2 temelli HTZ Wald testi. Yeni 2.662 çiftte Subject Agreement ayrıca diğer üç kategorinin çift sayısıyla ağırlıklandırılmış ortalamasıyla karşılaştırılır.
- **RQ4.** Aynı çiftte Total eksi Mean doğruluğu ve iki ölçütün kuantizasyon etkileri arasındaki fark. Uzunluk grupları modelin token sayılarıyla belirlenir ve betimleyici olarak raporlanır. [TurBLiMP değerlendirme kodu](https://github.com/ezgibasar/TurBLiMP/blob/297de13fb7a0ce524fe32e8b175c6b5255d66960/evaluation.py#L17-L22)
- **RQ5.** Q8 eksi FP16 için ±1 yüzde puan marjlı eşleştirilmiş TOST. İki tek yönlü testin büyük p değeri Holm düzeltmesine girer. %90 aralıklar nominaldir. Karar, düzeltilmiş p < 0,05 ile verilir.

Tek kontrastlarda CR2 standart hataları ve Satterthwaite serbestlik dereceleri kullanılır. Küme, eski analizdeki ortak söylem öneki anahtarıyla oluşturulur ve olası şablon bağımlılığını temsil eder. Her çifti tek küme kabul eden duyarlılık sonuçları ayrıca verilir. Holm aileleri tam örneklem ve yeni alt küme içinde ayrı ayrı RQ1 için 8, RQ2 için 4, RQ3 için 4, RQ4 için 10 ve RQ5 için 4 test içerir. Eski 1.000 çift betimleyici olarak raporlanır. Ana 3.662 çift ve yeni 2.662 çift örtüşür. Eşdeğerlik yalnızca genel doğruluk farkı için değerlendirilir. [CR2 ve HTZ yöntemi](https://doi.org/10.1080/07350015.2016.1247004) · [clubSandwich HTZ belgesi](https://github.com/jepusto/clubSandwich/blob/main/vignettes/Wald-tests-in-clubSandwich.Rmd)

## Yeniden çalıştırma

Kayıtlı skorlardan ayrı bir çıktı dizininde tabloları üretmek ve arşivle karşılaştırmak için [yeniden üretim kılavuzundaki komutları](../../../docs/reproducibility_v2.md#recompute-tables-from-saved-scores) kullanın. Referans çalıştırma conda-forge üzerinden R 4.5.3 ve CRAN üzerinden `clubSandwich` 0.7.0 kullandı. `clubSandwich` kaynak paketinin SHA-256 değeri `f3cd9cd5840022b8d1354dce3e7e4a820edcd47411b9f95bdd200c4e040032ed`. Sürümler [R oturum kaydında](session-info.txt) yer alır.

## Sayısal doğrulama

İlk HTZ çıktısında, EuroLLM Total Irregular Forms hücresindeki matematiksel olarak sıfır varyans sayısal yuvarlama nedeniyle pozitif görünüyordu. Sabit kategori etkileri ve kovaryansın göreli özdeğerleri kontrol edilerek protokolün tekillik koşulu uygulandı. Dört ilgili omnibus hücresi ve üç kategori güven aralığı hesaplanamadı olarak kaydedildi. RQ3 Holm aile büyüklüğü dört olarak korundu. Doğruluklar, karar geçişleri, McNemar p değerleri, CR2 standart hataları, TOST aralıkları ve Holm sonuçları ayrı hesaplarla denetlendi.
