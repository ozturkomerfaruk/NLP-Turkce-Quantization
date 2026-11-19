# Kaynak ölçümleri ve yüklem filtresi

## Kaynak ölçümleri

| Model | Hassasiyet | Ağırlık (GB) | Skorlama (s) | Token/s | Tepe MLX (GB) | Tepe süreç RSS (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EuroLLM | FP16 | 3,314 | 642,21 | 154,1 | 3,419 | 2,403 |
| EuroLLM | Q8 | 1,761 | 538,28 | 183,8 | 2,078 | 1,246 |
| EuroLLM | Q4 | 0,932 | 351,61 | 281,4 | 1,247 | 1,681 |
| Qwen | FP16 | 3,441 | 565,26 | 206,9 | 3,561 | 2,418 |
| Qwen | Q8 | 1,828 | 443,08 | 264,0 | 2,136 | 0,875 |
| Qwen | Q4 | 0,968 | 439,00 | 266,4 | 1,276 | 1,020 |

GB ondalık birimdir. Ağırlık, manifestteki `.safetensors` dosyasının disk boyutudur. Süre, 74 parçanın kayıtlı skorlama sürelerinin toplamıdır. Model yükleme, ısınma ve kayıt yazma işlemleri bu toplamın dışında kalır. Token/s, puanlanan cümle token’larının bu süreye oranıdır. Modellerin tokenizer’ları farklı olduğundan token/s öncelikle her model ailesi kendi içinde yorumlanır.

MLX ölçümü, her parçadan önce sıfırlanan aktif MLX ayırıcı tepesinin parçalar arasındaki maksimumudur. Ayrı izlenen ayırıcı önbelleği bu ölçüme dahil değildir. Süreç RSS, macOS `ru_maxrss` süreç yüksek su işaretidir ve yükleme ile ısınma sırasındaki tepeyi de yansıtabilir. Ölçümler ayrı sütunlarda tutulur. Her yapılandırma için tek, sabit sırada yürütülen süreç kullanıldığı için süre ve bellek değerleri koşuya özgü betimleyici ölçümlerdir. [MLX tepe bellek tanımı](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html) · [Darwin kaynak kullanımı tanımı](https://github.com/apple/darwin-xnu/blob/main/bsd/sys/resource.h)

İlk 1.000 çiftlik EuroLLM koşusunda Q4’ün aktif MLX tepesi FP16’ya göre %63,8 düşerken süreç RSS 1,247 GB’den 1,573 GB’ye yükselmişti. Revizyon koşusunda iki modelde de Q8 ve Q4 süreç RSS değerleri FP16’dan düşüktü. Her iki modelin Q4 RSS değeri ise Q8’in üzerindeydi. Tarihsel %63,8 oranı yalnızca ilk koşunun MLX ölçümüyle ilişkilendirilir. [İlk koşu ölçümleri](../../deep_analysis/extended_statistics.json) · [Revizyon ölçümleri](resource_measurements.csv)

## Predicate screen

Kural yalnızca **Nominalization** kategorisinde `critical_region` alanı **`şaşırdı`**, **`tepki gösterdi`** veya **`inanıyor`** değerlerinden birine tam eşit olan kayıtları işaretler. Kural skor veya tercih değişimini kullanmaz. Eski 1.000 çiftte aynı **17 kimlik** yeniden bulundu. Tam verideki **48 çiftin** 31’i önceki örneklemin dışındadır.

| Kritik bölge | Eski 1.000 | Yeni 2.662 | Tam 3.662 |
| --- | ---: | ---: | ---: |
| şaşırdı | 6 | 13 | 19 |
| tepki gösterdi | 8 | 11 | 19 |
| inanıyor | 3 | 7 | 10 |
| **Toplam** | **17** | **31** | **48** |

Önceki keşifsel incelemede ilk iki yüklemde olay odaklı `-mA` ve olgu odaklı `-DIK` tümleçlerinin farklı okumalarda yorumlanabileceği gözlendi. `inanıyor` çiftlerinde olguya inanma ile davranışa ilişkin daha bağlama duyarlı bir okuma arasındaki ayrım incelendi. İkinci okuma bağlam gerektirebilir ve daha az doğal olabilir. Bu gerekçe yüklem bazlı bir duyarlılık kontrolünü destekler. Ana analizde 48 çift ve özgün etiketleri kullanıldı. [İlk analiz kuralı](../../deep_analysis/analyze_extended.py#L199-L203) · [İşaretlenen 48 çift](predicate_screen_pairs.csv)

| Model | Skor | Q4 − FP16, 3.662 çift (yp) | Q4 − FP16, filtre sonrası 3.614 çift (yp) |
| --- | --- | ---: | ---: |
| EuroLLM | Mean | −2,24 | −1,99 |
| EuroLLM | Total | −3,00 | −2,91 |
| Qwen | Mean | −3,09 | −3,10 |
| Qwen | Total | −3,22 | −3,24 |

Q4 farkı dört model ve skor birleşiminde de negatif kaldı. Tablo, işaretlenen kayıtların çıkarılmasına dayanan keşifsel bir betimlemedir. [Tüm örneklem, hassasiyet ve skor sonuçları](predicate_sensitivity.csv)

## Açıklayıcı örnekler

Bu üç örnek yeni 2.662 çift arasındaki karar değişimlerinden amaçlı seçildi. Her yüklem için bir örnek verildi. Okumalar AI destekli ön yorumlardır. Kabullenilebilirlik çalışması TurBLiMP yazarlarına aittir. Bu revizyonda aşağıdaki çiftler için bağımsız uzman yargısı alınmadı. [TurBLiMP insan değerlendirmesi](https://aclanthology.org/2025.emnlp-main.834/)

| Çift | TurBLiMP’in tercih ettiği cümle | Alternatif | FP16 → Q4 değişimi ve ön yorum |
| --- | --- | --- | --- |
| `nominalization:0139` | Ancak müdürün herkesin fikrini almasına çalışanlar çok şaşırdı. | Ancak müdürün herkesin fikrini aldığına çalışanlar çok şaşırdı. | EuroLLM Mean ve Total alternatifi seçmekten etiketle uyumlu seçime geçti. Olay ve olgu okumaları iki biçimi de bağlama göre yorumlanabilir kılabilir. |
| `nominalization:0578` | Tüm kanunların uygulanmamasına vatandaşlar tepki gösterdi. | Tüm kanunların uygulanmadığına vatandaşlar tepki gösterdi. | Qwen Mean ve Total alternatifi seçmekten etiketle uyumlu seçime geçti. Tepki, uygulamama olayına veya uygulanmadığı olgusuna yöneltilebilir. |
| `nominalization:0108` | Kız kardeşinin doğruyu söylediğine canı gönülden inanıyor. | Kız kardeşinin doğruyu söylemesine canı gönülden inanıyor. | İki modelde Mean ve Total etiketle uyumlu seçimden alternatife geçti. İlk biçimin olgusal okuması daha açık. İkinci biçimin kabulü, davranışa yönelik bir okuma ve ek bağlam gerektirebilir. |

## Tekrar üretme

```bash
.venv/bin/python scripts/summarize_revision_diagnostics.py
```

Betik doğrulanmış kontrol noktalarını ve model manifestlerini okur.
