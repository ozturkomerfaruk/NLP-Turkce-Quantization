# Revizyon deneyi analiz protokolü

**Sürüm 1.1 · 18 Kasım 2026**

Bu protokol yeni değerlendirme sonuçları görülmeden hazırlanmıştır. İlk EuroLLM deneyinin 1.000 çiftlik sonuçları biliniyor. Protokol bu sonuçların ön kaydı olarak sunulmayacaktır.

## Araştırma soruları

1. **RQ1** — FP16’dan Q8 ve Q4’e geçiş Türkçe minimal çift doğruluğunu nasıl etkiliyor?
2. **RQ2** — FP16’ya göre doğruluk farkları iki model arasında değişiyor mu?
3. **RQ3** — Q4’ün FP16’ya göre doğruluk farkı dört dilbilgisel kategori arasında değişiyor mu?
4. **RQ4** — Mean ve Total Log Probability doğruluğu ve kuantizasyon etkisini nasıl değiştiriyor?
5. **RQ5** — Q8’in FP16’ya göre doğruluk farkı önceden belirlenen kabul sınırları içinde mi?

## Veri ve değerlendirme

Her iki model FP16, Q8 ve Q4 olarak aynı 3.662 çiftte değerlendirilir. İkinci model ve sabit kaynak sürümleri ana puanlamadan önce kaydedilir. Her çiftte iki cümle aynı model, tokenizer ve bağlam kuralıyla puanlanır. Yalnız cümlenin kendi token’ları skora girer. Teknik başlangıç bağlamı ve sona eklenen özel token’lar skor ve token sayısı dışında tutulur. Bir modelin üç hassasiyet düzeyinde aynı token dizisi doğrulanır.

```math
S_{\mathrm{total}}(x)=\sum_{i=1}^{n}\log P(x_i\mid x_{<i})
```

```math
S_{\mathrm{mean}}(x)=\frac{1}{n}\sum_{i=1}^{n}\log P(x_i\mid x_{<i})
```

Tercih edilen cümlenin skoru kesin olarak yüksekse karar doğru sayılır. Eşitlikler yanlış olarak kodlanır ve ayrıca sayılır. Eksik, yinelenen ya da sonlu olmayan skor varsa ilgili koşu düzeltilip denetlenmeden analiz başlatılmaz. Teknik pilotta skorlanan çiftler ve analizde kullanılan çıktı sürümleri kaydedilir.

Eski 1.000 çift, tam veriyle kategori ve iki cümlenin sıralı metni üzerinden bire bir eşleştirilir. Eşleşmeler ve geriye kalan 2.662 çift, skorlamadan önce bir üyelik listesiyle sabitlenir. Sayılar veya eşleşmeler uyuşmazsa analizden önce veri kimliği sorunu çözülür.

| Veri | Rol |
| --- | --- |
| 3.662 çift | İki model için ana etki tahminleri ve protokoldeki testler |
| Eski 1.000 çift | Gönderilen örneklemle betimleyici karşılaştırma |
| Önceden puanlanmamış 2.662 çift | Yeni çiftlerde örüntünün tekrarlanmasını değerlendiren ayrı kontrol |

Yeni alt küme aynı kaynak veri kümesinden gelir. İki alt kümenin benzer cümle şablonlarını paylaşma olasılığı, yeni çiftlerdeki tekrarın yorumunda belirtilir. Yeni alt kümede de aynı etki yönleri, testler ve Q8 kabul marjı kullanılır. Ana örneklem ile yeni alt küme örtüştüğü için iki analiz birer bağımsız doğrulama olarak sayılmaz.

## Raporlama ve eşleştirilmiş karşılaştırmalar

Her model, hassasiyet, skor yöntemi ve veri grubu için iki genel doğruluk özeti birlikte sunulur. **Micro Accuracy**, doğru karar sayısının toplam çift sayısına oranıdır. **Macro Accuracy**, dört kategori doğruluğunun ağırlıksız aritmetik ortalamasıdır. FP16’ya göre Micro ve Macro farkları kendi ölçeklerinde ayrı raporlanır. Ana hipotez testlerinde, Q8 kabul marjı dahil, çift ağırlıklı Micro fark kullanılır. Macro sonuçlar kategori büyüklüklerinin genel özete etkisini göstermek için betimleyici olarak sunulur. Bu raporlama kararı 1.1 sürümünde, yeni skorlar üretilmeden eklendi.

İki skor yöntemi eşit ağırlıkla sunulur. Her model, hassasiyet düzeyi ve veri grubu için doğruluk ve %95 güven aralığı verilir. Q8 ve Q4 için aynı çiftteki doğru karar göstergelerinin farkı D = doğruluk(kuantize) − doğruluk(FP16) olarak kodlanır. Ortalama D, doğruluk farkını verir ve 100 ile çarpılarak yüzde puanla raporlanır. Doğrudan yanlış karara ve yanlıştan doğru karara geçişler, toplam tercih değişimi ve %95 fark aralığı da verilir. Genel doğruluk her çifte eşit ağırlık verir. Kategori doğrulukları ayrıca sunulur.

- **RQ1** — Her model, kuantizasyon seviyesi ve skor yöntemi için ortalama D’nin sıfırdan farkı sınanır. Önceki çalışmayla karşılaştırma için kesin McNemar testi ikincil kontrol olarak raporlanır.
- **RQ2** — Aynı çiftte D(ikinci model) − D(EuroLLM) hesaplanır. Ortalama fark Q8 ve Q4 için iki skor yöntemiyle ayrı sınanır.
- **RQ4** — Her model ve hassasiyet düzeyinde aynı çiftte doğru(Total) − doğru(Mean) hesaplanır. Ayrıca her model ve kuantizasyon seviyesi için D(Total) − D(Mean) sınanır. Tercih edilen cümlenin token sayısı daha az, eşit veya daha fazla olan çiftler model bazında ayrı betimlenir.

Tek örnekli modellerde sabit terim, ortalama doğruluğu veya ilgili eşleştirilmiş farkı tahmin eder. Küme dayanıklı CR2 standart hatası ve tek kontrastlar için Satterthwaite serbestlik dereceli t yaklaşımı kullanılır. Kümeler eski çalışmadaki [ortak önek kuralıyla](../results/deep_analysis/analyze_extended.py#L134-L149), yalnız kategori ve cümle metinlerinden, skorlar görülmeden oluşturulur. Kural Türkçe metni normalize eder ve iki cümlede ortak olan belirli söylem öneklerini kaldırır. Küme sayısı ve büyüklük dağılımı raporlanır. Bu gruplar olası şablon benzerliğini temsil eder. Çift düzeyinde bağımsızlık varsayan sonuçlar duyarlılık kontrolü olarak ayrıca verilir. [Küme dayanıklı çıkarım yöntemi](https://doi.org/10.1080/07350015.2016.1247004)

## Q8 eşdeğerliği

Kabul marjı **±1,0 yüzde puan** olarak sabitlenir. Ölçeği, 100 minimal çift başına yaklaşık bir net doğru karar farkıdır. Bu, yerel değerlendirmede FP16 yerine Q8 kullanımı için çalışmanın benimsediği pratik doğruluk toleransıdır. Uygulamaya özgü doğrulanmış bir endüstri eşiği bulunmadığından yorum bu benchmark ve kullanım amacıyla sınırlıdır. Marj yeni sonuçlara göre değiştirilmeyecektir.

Her model ve skor yöntemi için eşleştirilmiş D üzerinde iki tek yönlü test (TOST) yürütülür. Sıfır hipotezleri sırasıyla ortalama D’nin −1 yüzde puan veya altında ve +1 yüzde puan veya üstünde olmasıdır. İki tek yönlü p değerinin büyüğü TOST p değeridir. Düzeltilmemiş %90 güven aralığı ve iki tek yönlü p değeri de sunulur. Eşdeğerlik kararı, dört TOST p değerine uygulanan Holm düzeltmesinden sonra p < 0,05 ise verilir. Bu koşul sağlanmazsa eşdeğerliğin gösterilemediği bildirilir. Doğruluk eşdeğerliği toplu bir ölçüttür. Çift düzeyinde değişen kararlar ayrıca raporlanır. [TOST yöntemi](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/)

## Kategoriler arası heterojenlik

Her model ve skor yöntemi için çift düzeyinde Q4 − FP16 farkı, kategoriyi dört düzeyli bir değişken olarak içeren doğrusal modelle analiz edilir. Dört kategori ortalamasının eşitliği, üç bağımsız kontrastı birlikte sınayan CR2 temelli yaklaşık Hotelling T² (HTZ) Wald testiyle değerlendirilir. Böylece kategorilerin FP16’ya göre değişimleri doğrudan karşılaştırılır. Kategori ortalama etkileri ve %95 aralıkları raporlanır. Küme kovaryansı tekilse veya standart hata ya da serbestlik derecesi geçersizse ilgili çıkarım hesaplanamadı olarak raporlanır. Bu durum sıfır genişlikli bir aralıktan eşdeğerlik veya fark kararı çıkarmak için kullanılmaz. [HTZ yöntemi](https://doi.org/10.1080/07350015.2016.1247004)

Subject Agreement yoğunlaşması eski örneklemden gelen keşifsel bir gözlemdir. Önceden puanlanmamış 2.662 çiftte Subject Agreement’ın Q4 farkı ile diğer üç kategorinin çift sayısına göre ağırlıklı Q4 farkı arasındaki kontrast ayrıca tahmin edilir. Negatif fark, daha büyük bir kayıpla uyumludur. Dört model ve skor birleşiminde kontrast ile %95 aralıkları ayrı sunulur. Bu kontrol genel heterojenlik testiyle birlikte yorumlanır.

## Karar kuralları ve kayıt

RQ1, RQ2 ve RQ4’ün fark testleri çift yönlüdür. Ana analizde Holm düzeltmesi her soru için ayrı uygulanır. RQ1 ailesi sekiz testi, RQ2 dört testi, RQ3 dört genel heterojenlik testini, RQ4 on doğruluk ve kuantizasyon farkı karşılaştırmasını, RQ5 dört TOST testini içerir. Karar sınırı düzeltilmiş p < 0,05’tir. Hesaplanamayan testlerde karar verilmez ve düzeltmede planlanan aile büyüklüğü korunur. Güven aralıkları aksi belirtilmedikçe nominaldir. Yeni 2.662 çiftte aynı aileler ayrı değerlendirilir.

Ana analiz planlıdır ve daha önce incelenen çiftleri içerir. Yeni 2.662 çift, planlı tekrar kontrolüdür. Uzunluk alt grupları ve eski 1.000 çiftin yeniden analizi betimleyicidir. Ek dışlamalar, sonradan tanımlanan alt gruplar ve diğer plan dışı analizler keşifsel olarak etiketlenir. Önceden puanlanmamış çiftlerin yeni skorları incelendikten sonra protokol değişirse değişiklik, gerekçe ve zamanlama ayrı kayda alınır.
