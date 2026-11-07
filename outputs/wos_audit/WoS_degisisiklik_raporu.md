# WoS kaynak ve metin düzenleme raporu

6 Kasım 2026

## Kapsam

Kullanıcının açıklaması doğrultusunda deneysel çalışma korunmuştur. Özet, deney düzeni, örneklem, model yapılandırmaları, istatistik yöntemleri, sonuçlar, sayısal tablolar, denklemler, şekiller ve sınırlılıklar değiştirilmemiştir. Girişte iki paragraf, ilgili çalışmalarda üç paragraf ve bir alt başlık, tartışmada bir paragraf düzenlenmiştir. Orijinal belgeye dokunulmamıştır.

## Kaynak denetimi

Aşağıdaki sekiz bilimsel yayın Web of Science Core Collection içindeki gerçek kayıtlarıyla doğrulanmıştır. WoS kaydı, bütün çalışmaların SCI/SSCI dergi makalesi olduğu anlamına gelmez; konferans bildirileri de bu kapsamdadır. Yalnız SCI/SSCI şartı ayrıca istenmemiştir.

| Yayın | WoS kayıt numarası | Kullanımı |
|---|---|---|
| Dettmers ve ark. 2022 LLM.int8 | [001202259103056](https://www.webofscience.com/wos/woscc/full-record/WOS:001202259103056) | Giriş ve 2.1–2.2: karma hassasiyet ve bellek |
| Xiao ve ark. 2023 SmoothQuant | [001372555100003](https://www.webofscience.com/wos/woscc/full-record/WOS:001372555100003) | Giriş ve 2.1–2.2: ağırlık/aktivasyon nicemlemesi |
| Toraman ve ark. 2023 | [000998929700024](https://www.webofscience.com/wos/woscc/full-record/WOS:000998929700024) | Giriş, 2.2–2.3 ve tartışma: Türkçe tokenizasyon karşılaştırmaları |
| Warstadt ve ark. 2020 BLiMP | [000736531900025](https://www.webofscience.com/wos/woscc/full-record/WOS:000736531900025) | 2.3: minimal çiftlerle dilbilgisel değerlendirme |
| Clopper ve Pearson 1934 | [000200863400002](https://www.webofscience.com/wos/woscc/full-record/WOS:000200863400002) | Mevcut istatistik atfı korundu |
| Efron 1979 | [A1979GL00300001](https://www.webofscience.com/wos/woscc/full-record/WOS:A1979GL00300001) | Mevcut bootstrap atfı korundu |
| Holm 1979 | [A1979JY78700003](https://www.webofscience.com/wos/woscc/full-record/WOS:A1979JY78700003) | Mevcut çoklu test atfı korundu |
| McNemar 1947 | [000203876700008](https://www.webofscience.com/wos/woscc/full-record/WOS:000203876700008) | Mevcut eşleştirilmiş test atfı korundu |

## Düzenlemelerin bilimsel gerekçesi

Marchisio 2024 ve Bayram 2025 üzerinden verilen çok dilli/Türkçe nicemleme sonuçları çıkarıldı. Yerlerine nicemleme yöntemlerinin verimlilik bulguları ile bu deneyin farklı kapsamı açıklandı. Yeni metin Türkçenin diğer dillerden daha kırılgan olduğunu veya insanlarca değerlendirmeyi ölçtüğünü ileri sürmez.

Başar ve Bisazza 2026 üzerine kurulu tokenizasyon/biçimbirim açıklamaları, WoS içinde konu aramasıyla bulunan Toraman ve ark. 2023 çalışması kullanılarak yeniden yazıldı. WoS tam kaydındaki özet incelendi: beş tokenizasyon yöntemi, Türkçe OSCAR, RoBERTa eğitimi ve altı sonraki görev. Yeni metin yalnız bu kapsamda iddiada bulunur; bu yayını nicemleme veya TurBLiMP çalışması olarak tanıtmaz. Tam metin üzerinden ayrıntılı sonuç karşılaştırması yapılmamıştır. Yazar bilgileri ve sayfalar Crossref DOI kaydıyla tamamlandı.

Frantar 2023 GPTQ ve Lin 2024 AWQ teknik karşılaştırmaları kaldırıldı; yöntem ayrımları doğrulanmış Dettmers ve Xiao çalışmalarıyla açıklandı. AWQ için GetMobile dergi sürümü bulundu ancak WoS kaydında türü Editorial Material olarak gösterildiği ve konferans kaynağıyla bibliyografik farklılıklar bulunduğu için metne eklenmedi.

Clopper–Pearson DOI sorgusunda çıkmadı, başlık sorgusunda bulundu. Bu nedenle mevcut DOI yalnızca arama uyuşmazlığı yüzünden değiştirilmedi.

## Deney kökeni için korunan dört kayıt

Başar ve ark. 2025 TurBLiMP; Martins ve ark. 2024 EuroLLM; MLX Contributors teknik dokümantasyonu; UTTER Project model kartı. Bunlar kullanılan veri/model/yazılımın gerçek kaynağını belirtmek için korunmuştur ve WoS doğrulaması yapılmış yayınlar olarak sunulmamıştır. Kaynakçanın toplamı 12 kayıttır: sekiz WoS doğrulanmış bilimsel yayın ve dört deney kökeni kaydı.

## Kontroller

Orijinal ve yeni DOCX karşılaştırıldı: tüm tabloların OOXML içeriği, denklemler, çizimler ve düzenleme kapsamı dışındaki paragraflar birebir korunmuştur. Metinde kaldırılan beş kaynağa artık atıf yoktur. Ayrıntılı önce/sonra metinleri changes.json içinde saklanmıştır.
