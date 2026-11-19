# İlk EuroLLM deneyinin denetimi

İlk deney `036acec81a1f3bd0cf31697cab4c75c09e4bc5c3` commit’inde sabitlenmiştir. Her hassasiyet düzeyinde 20 kontrol noktası ve 1.000 benzersiz çift doğrulandı. Ham veri ile örneklem dosyalarının hash’leri [ilk deneyin manifestiyle](../data/processed/manifest.json) eşleşti.

| Dosya | SHA-256 |
| --- | --- |
| `turblimp_1000.jsonl` | `bcb2c7f25475f8290d1511d742e1e9f8f515fd29df11dd3e3f64f2f53bcb4835` |
| `turblimp_pilot_20.jsonl` | `420407b129d0e997680a50ab11ff49319e933c1b82b07068042b5d1249239f27` |
| `augmented_subject_verb_agreement.csv` | `8eb3b48ecf76dac8ca6db608480004ba7bd6c5f7d6e521ab431389e85f4b00f6` |
| `augmented_argument_structure_transitive.csv` | `8f1b38b3fb540d987bcac784dde054c9bcf3469455e15a65903caad6de16a6c3` |
| `augmented_nominalization.csv` | `57944471822eba1aa97bb707d63e1807a84c4a227a25d24dbffc273256ed3c12` |
| `augmented_irregular_forms.csv` | `68d2b1189e65fac792b7cbcba947453ae52b349b991e53e47b203592418bca35` |

| Model | Ortalama log-olasılık | Toplam log-olasılık |
| --- | ---: | ---: |
| FP16 | 912/1000 | 946/1000 |
| Q8 | 915/1000 | 948/1000 |
| Q4 | 884/1000 | 907/1000 |

## Model dosyaları

| Model dizini | Boyut, bayt | SHA-256 |
| --- | ---: | --- |
| `eurollm-source` | 3313726248 | `658a7ea60f10d4e283c2cc61b4685898b98df25768f4ee9d34450a9efe3eafbd` |
| `eurollm-fp16` | 3313725829 | `179b442ef5a4a22cad87640f6c89a11d9dafe81674cdcf9df6d11a7f4545bac9` |
| `eurollm-q8` | 1760560359 | `69553d7d3fcd6e5cab96a4227e877c83b63df16efc5bfa12f5c9f56364a3a942` |
| `eurollm-q4` | 932184842 | `a653c92f03b133b79f57b05cf680ace89ee6f2e5313baca8ccde1fd4e2581577` |

İlk deneyde kaydedilen indirme referansı `refs/pr/5`, sabit kaynak commit’i `e00c97a9a389cc184174163cff4ba58facdb8e28` ile eşleştirildi. [EuroLLM model yapım kaydı](eurollm_revision_build.md), aynı kaynaktan yeniden oluşturulan revizyon dosyalarının ilk deney ağırlıklarıyla eşleştiğini belgeler.

## Kayıtlı skorlardan yeniden hesaplama

İlk deneyin kayıtlı skorlarını ayrı bir çalışma kopyasında Python ile analiz edin.

```bash
python3 scripts/analyze_main.py
```

Komut [ilk deneyin ana tablolarını](../results/analysis/) yeniden oluşturur. Yerel ağırlık dosyaları yoksa yeniden hesaplanan raporda ağırlık boyutları boş kalır. [Genişletilmiş tarihsel analiz](../ProjeMarkdown/ISTATISTIKSEL_ANALIZ_VE_DENEY_RAPORU.md), ilk 1.000 çiftin ek istatistiklerini ve doğrulama kayıtlarını içerir. Tarihsel SHA-256 envanteri özgün denetim kopyasını belgeler.

İlk koşunun ortamı Python 3.12.15, MLX 0.32.4 ve MLX-LM 0.32.2 idi. Genişletilmiş revizyon deneyi için [yeniden üretim kılavuzuna](reproducibility_v2.md) bakın.
