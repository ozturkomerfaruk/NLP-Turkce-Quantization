# Revizyon deneyinin istatistik sonuçları

Ana analiz, EuroLLM ve Qwen için aynı **3.662 minimal çifti** kapsar. Önceki 1.000 çift betimleyici karşılaştırma, daha önce puanlanmamış 2.662 çift ise planlı tekrar kontrolüdür. Aşağıdaki doğruluklar Micro Accuracy’dir. Dört kategoriyi eşit ağırlıklandıran Macro Accuracy ve nominal %95 güven aralıkları [doğruluk tablosunda](accuracy.csv) yer alır.

## Genel doğruluk

| Model | Skor | FP16 | Q8 | Q4 |
| --- | --- | ---: | ---: | ---: |
| EuroLLM | Mean | %91,04 | %91,02 | %88,80 |
| EuroLLM | Total | %94,13 | %94,13 | %91,13 |
| Qwen | Mean | %82,55 | %82,44 | %79,46 |
| Qwen | Total | %86,40 | %86,43 | %83,18 |

[FP16’ya göre farklar ve karar geçişleri](paired.csv) Micro ve Macro farkı, kümelenmiş %95 aralığı, doğru→yanlış ve yanlış→doğru sayıları, tercih değişimi ve kesin McNemar p değerini içerir. İlk deneyin EuroLLM sonuçları aynı 1.000 çiftte yeniden elde edildi. Mean doğru sayıları FP16, Q8 ve Q4 sırasında 912, 915 ve 884. Total doğru sayıları sırasıyla 946, 948 ve 907.

## Cümle uzunluğu

| Model, FP16 | Tercih edilen cümle | Çift | Mean | Total |
| --- | --- | ---: | ---: | ---: |
| EuroLLM | Daha kısa | 789 | %76,43 | %96,58 |
| EuroLLM | Eşit uzunlukta | 2.224 | %93,66 | %93,66 |
| EuroLLM | Daha uzun | 649 | %99,85 | %92,76 |
| Qwen | Daha kısa | 512 | %40,43 | %87,30 |
| Qwen | Eşit uzunlukta | 2.396 | %86,73 | %86,73 |
| Qwen | Daha uzun | 754 | %97,88 | %84,75 |

Her iki modelde Total, tercih edilen cümle daha kısa olduğunda genel olarak daha yüksek, daha uzun olduğunda daha düşük doğruluk verdi. Eşit token uzunluğunda Mean ve Total hiçbir konfigürasyonda tercihi değiştirmedi. Uzunluk grupları modelin kendi tokenizer’ına göre tanımlandığı için grup büyüklükleri değişiyor. Bu betimleyici örüntü, Total’in kısa dizilere avantaj sağlayabildiği yorumuyla uyumlu. [Tüm hassasiyet ve örneklem grupları](length.csv) · [TurBLiMP’in toplama kullanan değerlendirme kodu](https://github.com/ezgibasar/TurBLiMP/blob/297de13fb7a0ce524fe32e8b175c6b5255d66960/evaluation.py#L17-L22)

## Q8 doğruluk eşdeğerliği

±1,00 yüzde puanlık önceden tanımlanmış marj içinde eşdeğerlik ana örneklemde dört model ve skor birleşiminin tamamında gösterildi. Dördü de yeni 2.662 çiftte aynı kararı verdi. Aralıklar nominal %90, p değerleri dört test için Holm düzeltmelidir.

| Model | Skor | Q8 − FP16, yüzde puan | %90 GA | Düzeltilmiş p | Tercih değişimi |
| --- | --- | ---: | --- | ---: | ---: |
| EuroLLM | Mean | −0,03 | [−0,16, 0,11] | 2,66 × 10⁻³⁰ | 9 |
| EuroLLM | Total | 0,00 | [−0,14, 0,14] | 4,44 × 10⁻²⁹ | 10 |
| Qwen | Mean | −0,11 | [−0,29, 0,07] | 8,97 × 10⁻¹⁶ | 16 |
| Qwen | Total | +0,03 | [−0,19, 0,24] | 1,16 × 10⁻¹³ | 23 |

Eşdeğerlik toplu doğruluk için geçerli. Örneğin EuroLLM Total’de beş doğru karar yanlışa, beş yanlış karar doğruya döndü ve toplam doğruluk değişmedi. [TOST ve duyarlılık sonuçları](tests.csv) · [Eşleştirilmiş geçişler](paired.csv)

## Q4 doğruluk kaybı

| Model | Skor | Q4 − FP16, yüzde puan | %95 GA | Düzeltilmiş p | Doğru → yanlış | Yanlış → doğru |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| EuroLLM | Mean | −2,24 | [−3,21, −1,27] | 0,0000348 | 140 | 58 |
| EuroLLM | Total | −3,00 | [−3,92, −2,09] | 1,65 × 10⁻⁹ | 146 | 36 |
| Qwen | Mean | −3,09 | [−4,34, −1,83] | 0,00000920 | 215 | 102 |
| Qwen | Total | −3,22 | [−4,47, −1,97] | 0,00000357 | 239 | 121 |

Dört Q4 kaybı da yeni 2.662 çiftte Holm düzeltmesinden sonra gözlendi. Model aileleri arasındaki kuantizasyon etkisi farkı RQ2’de anlamlı bulunmadı. Total’in genel doğruluğu altı yapılandırmada da Mean’in üzerindeydi. Total ile Mean arasındaki **Q4 kaybı farkı** EuroLLM’de −0,76, Qwen’de −0,14 yüzde puandı. Bu farklar RQ4’ün Holm düzeltmesinden sonra anlamlı bulunmadı. [Eşleştirilmiş farklar](paired.csv) · [RQ1, RQ2 ve RQ4 testleri](tests.csv)

## Kategori heterojenliği ve Subject Agreement

| Model | Skor | Kümelenmiş HTZ, 3.662 çift | Kümelenmiş HTZ, yeni 2.662 çift |
| --- | --- | ---: | ---: |
| EuroLLM | Mean | 0,0167 | 0,1458 |
| EuroLLM | Total | Hesaplanamadı | Hesaplanamadı |
| Qwen | Mean | 0,0394 | 0,0656 |
| Qwen | Total | 0,00705 | 0,02285 |

Tablo Holm-düzeltilmiş p değerlerini gösterir. EuroLLM Total’de Irregular Forms için tüm çiftlerde Q4−FP16 doğru karar farkı sıfırdı. Dört kategoriyi içeren kovaryans tekil olduğu için protokol uyarınca HTZ sonucuna karar verilmedi. Gözlenen kategori farkı 0,00 yüzde puan ve bu kategori için CR2 güven aralığı hesaplanamadı. Holm ailesi dört testlik boyutunu korudu.

Yeni 2.662 çiftte Subject Agreement için Q4 farkı eksi diğer kategorilerin çift ağırlıklı Q4 farkı dört model ve skor birleşiminde de negatifti. Nominal %95 aralık yalnızca EuroLLM Total’de **−4,08** [−6,86, −1,31] ve Qwen Mean’de **−3,78** [−7,32, −0,24] yüzde puan için sıfırın altında kaldı. EuroLLM Mean’de fark **−0,61** [−3,67, 2,44], Qwen Total’de **−2,93** [−6,33, 0,46] yüzde puandı. Bu örüntü, özne uyumunun tüm model ve skor birleşimlerinde aynı biçimde öne çıktığı sonucunu desteklemiyor. Örneğin Qwen Total’de yeni çiftler arasında geçişli argüman yapısındaki kayıp, özne uyumundakinden daha büyüktü. [Q4 kategori etkileri](q4_category_effects.csv) · [Yeni çiftlerde Subject Agreement kontrastı](subject_agreement.csv)

## Bağımlılık ve sağlamlık

Tam verideki 2.273 kümenin 1.934’ü tek çift içeriyor ve en büyük kümede 10 çift var. Eski ve yeni alt kümelerin 257 ortak küme anahtarı var. Ortak önek kuralı olası benzerliği yakalayan bir göstergedir. Sonuçlar bu şablon tanımına bağlı değerlendirilir.

Çiftlerin bağımsız kabul edildiği duyarlılık analizinde Q8 eşdeğerliği ve Q4 kaybı kararları korundu. Planlanan test ailelerinde kararın değiştiği tek hücre yeni 2.662 çiftteki Qwen Mean heterojenliğiydi. Kümelenmiş HTZ düzeltilmiş p değeri 0,0656, bağımsız çift varsayımındaki karşılığı 0,0115. Ana yorum kümelenmiş sonucu izler. [Küme dağılımı](cluster_sizes.csv) · [Kümelenmiş ve bağımsız çift testleri](tests.csv)

Analiz [sabit protokolü](../../../docs/revision_analysis_protocol_v1.md) izledi. İstatistik paketinin sürümü, kullanılan testler ve tekrar çalıştırma komutları [yöntem kaydında](methods.md) bulunuyor.
