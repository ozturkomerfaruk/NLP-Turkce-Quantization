# Veri kapsamı

Dört TurBLiMP kategorisi seçili Türkçe biçimbilim ve biçimsözdizim ayrımlarını kapsar. Subject Agreement kişi ve sayı uyumunu, Argument Structure (Transitive) geçişli yapılardaki durum işaretlemeyi, Nominalization adlaştırmayı, Irregular Forms düzensiz biçimbilimsel gerçekleşmeleri temsil eder. Bu gerekçe revizyonda incelenen kapsamı açıklar. İlk 1.000 çiftlik çalışmadaki kategori seçiminin önceden belirlenmiş bir kuramsal gerekçesi belgelenmemiştir.

| Kategori | İlk deney | Revizyon |
| --- | ---: | ---: |
| Subject Agreement | 250 | 934 |
| Argument Structure (Transitive) | 250 | 835 |
| Nominalization | 250 | 945 |
| Irregular Forms | 250 | 948 |
| **Toplam** | **1.000** | **3.662** |

Dört kategorideki tüm tekrarsız çiftler revizyona dahildir. [Veri manifesti](../data/revision_v2/manifest.json) tekrarsızlaştırmayı ve kategori sayılarını belgeler. İlk örneklemin 1.000 çifti `in_baseline` alanıyla işaretlenmiştir. Kalan 2.662 çift, önceden puanlanmamış alt kümedir.

EuroLLM-1.7B ve Qwen3-1.7B-Base aynı çiftlerde FP16, Q8 ve Q4 olarak değerlendirilir. Mean ve Total Log Probability eşit ağırlıkla raporlanır. Micro Accuracy her çifte, Macro Accuracy her kategoriye eşit ağırlık verir. Bulgular bu dört kategori, iki model ailesi ve kullanılan nicemleme yöntemiyle sınırlıdır.
