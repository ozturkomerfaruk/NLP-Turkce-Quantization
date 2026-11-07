# Alt sözcük parçalanması analizi: kodun, matematiğin ve sonuçların ayrıntılı Türkçe rehberi

**Tarih:** 29 Ekim 2026. **İlgili betik:** [analyze_fragmentation.py](/Users/omerfarukozturk/Desktop/turkce-quantization/scripts/analyze_fragmentation.py).

Bu belge yalnızca bir sonuç raporu değildir. Hangi dosyanın okunduğunu, hangi fonksiyonun neden yazıldığını, verinin nasıl dönüştürüldüğünü, matematiksel işlemlerin ne anlama geldiğini ve elde edilen sayıların hangi sınırlar içinde yorumlanabileceğini açıklayan bir çalışma rehberidir. Kodda fonksiyon adları korunmuştur; açıklamalar Türkçedir. Kodun bölüm numaraları aşağıdaki anlatımla birlikte okunabilir.

## 1. Asıl soru neydi ve önceki çalışmadan farkı ne?

Önceki ana deney şu soruyu inceliyordu: Aynı Türkçe minimal çiftler FP16, Q8 ve Q4 ile puanlandığında doğruluk değişiyor mu? Bu ek analiz ise aynı kayıtların başka bir özelliğini kullanıyor:

> Bir minimal çiftin cümlelerinde sözcük başına daha fazla token düşmesi, FP16 ile Q4 arasındaki karar değişimiyle ilişkili mi?

Minimal çift, biri iyi diğeri hatalı olarak etiketlenmiş iki cümleden oluşur. Model bu iki cümleye verdiği puanlarla bir tercih yapar. Buradaki “doğru karar”, veri setindeki iyi etiketli cümlenin daha yüksek ortalama log-olasılık almasıdır. Bu ifade, etiketlerin bağımsız dilbilimcilerce doğrulandığı anlamına gelmez.

Bu çalışmada yeni cümle üretilmedi, model indirilmedi, ağırlık yüklenmedi ve FP16/Q8/Q4 puanlaması yeniden yapılmadı. Yeni olan şey, mevcut cümlelerin sözcük sayılarının çıkarılması ve bunların kayıtlı token sayıları ile ilişkilendirilmesidir. Regresyon burada küçük bir istatistiksel modeldir; EuroLLM’yi yeniden eğitmek anlamına gelmez.

Analiz ana deneyden sonra tasarlandığı için **keşifseldir**. Bulgular önceden kaydedilmiş bağımsız bir doğrulama deneyinin sonucu olarak sunulamaz.

## 2. Dosya haritası: nereden ne okundu, nereye ne yazıldı?

### 2.1. Okunan kaynaklar

| Yol | Neden okunuyor? | Kullanılan bilgi |
|---|---|---|
| `data/processed/turblimp_1000.jsonl` | Hangi cümlelerin karşılaştırıldığını bilmek | `pair_id`, kategori, iyi ve hatalı cümle |
| `results/main/fp16/chunk_*.jsonl` | Referans modelin kararını ve karara yakınlığını belirlemek | Karar, ortalama log-olasılıklar, marj, token sayıları ve token hash’leri |
| `results/main/q8/chunk_*.jsonl` | Daha hafif nicemlemeyi tanımlayıcı karşılaştırmak | Kararlar ve token tutarlılığı |
| `results/main/q4/chunk_*.jsonl` | Araştırmanın hedefi olan karar değişimini belirlemek | Kararlar ve token tutarlılığı |
| `results/deep_analysis/template_clusters.csv` | Benzer şablonlardan gelen çiftlerin bağımlılığına duyarlılığı incelemek | Yaklaşık küme kimliği ve kümeye ait çiftler |

`chunk_*` birden fazla parçalı sonuç dosyasını ifade eder. Dosyalar okunup tek bir kayıt listesinde toplanır; eşleştirme satır numarasıyla değil `pair_id` ile yapılır.

### 2.2. Ana çıktılar

| Dosya | İçerik ve kullanım |
|---|---|
| [fragmentation_pairs.csv](/Users/omerfarukozturk/Desktop/turkce-quantization/results/fragmentation_analysis/fragmentation_pairs.csv) | 1.000 satırın her biri bir minimal çifttir. Cümleler, token/sözcük ölçüleri, kararlar, geçişler ve grup kimlikleri incelenebilir. |
| [fragmentation_statistics.json](/Users/omerfarukozturk/Desktop/turkce-quantization/results/fragmentation_analysis/fragmentation_statistics.json) | Tabloların sayısal kaynağıdır. Modeller, katsayılar, ölçekler, bootstrap aralıkları ve kaynak hash’leri bulunur. |
| [fragmentation_plot.png](/Users/omerfarukozturk/Desktop/turkce-quantization/results/fragmentation_analysis/fragmentation_plot.png) | Çeyreklere göre toplam Q4 karar değişimi oranı ve küme bootstrap aralığı. |
| [fragmentation_report.md](/Users/omerfarukozturk/Desktop/turkce-quantization/results/fragmentation_analysis/fragmentation_report.md) | Betiğin otomatik ürettiği kısa araştırma raporu. |
| Bu belge | `ProjeMarkdown` altındaki ayrıntılı Türkçe kod ve yöntem açıklaması. Kısa otomatik rapordan ayrı tutulur. |

`test_fragmentation.py` temel fonksiyonları, `verify_outputs.py` gerçek çıktı ve veri bütünlüğünü kontrol eder. `tests.log`, `verification.log` ve `run.log` çalıştırma kayıtlarıdır. `implementation_plan.md` ilk uygulama planıdır. Bunların tamamı ek analiz klasöründedir; eski araştırma raporları üzerine yazılmaz.

## 3. Programın baştan sona akışı

```text
main()
  ├─ load_pairs(): kaynakları oku, kimlikleri ve tokenları doğrula
  │    ├─ read_jsonl(): satırları Python kayıtlarına çevir
  │    ├─ unique_index(): pair_id ile eşleştirmeyi güvenceye al
  │    ├─ sha(): veri kimliğini denetle
  │    └─ word_count(): sözcük sayılarını hesapla
  ├─ quartiles() + summary(): genel ve kategori tablolarını oluştur
  ├─ model_result(): beş modelin başlangıç tahminlerini hesapla
  │    └─ design() → effect() → fit_ridge()
  ├─ resample_weights(): yeni örnekleme ağırlıkları üret
  │    └─ her tekrarda effect() → fit_ridge(): modeli yeniden tahmin et
  ├─ interval(): bootstrap dağılımlarından aralıkları çıkar
  ├─ eski dosyaların hash’lerini tekrar kontrol et
  ├─ CSV ve JSON yaz
  ├─ plot(): PNG üret
  └─ report(): kısa Markdown raporunu üret
```

Burada `rows`, her elemanı bir çiftin bilgilerini taşıyan sözlüklerden oluşan listedir. `x`, sayısal açıklayıcı değişken matrisidir. `y`, her çift için karar değişimi olup olmadığını belirten 0/1 dizisidir. `w`, her satırın hesaplamaya ne kadar katkı vereceğini belirten ağırlıktır.

## 4. Başlangıç ayarları neden var?

`ROOT`, betiğin dosya konumundan proje kökünü bulur. Terminalin başka klasörde açık olması kaynak yollarını değiştirmez. `OUT`, yeni analiz dosyalarını `results/fragmentation_analysis/` altında toplar.

`MODELS`, okunacak üç kayıt grubunu belirler. `METHOD`, ana deneyin puanlama protokolünün kimliğidir. Başka yöntemle üretilmiş puanları bu analize yanlışlıkla karıştırmamak için kayıtlarla karşılaştırılır.

`SEED = 20261029`, rastgele yeniden örneklemeyi tekrarlanabilir yapar. Aynı veri, kod, seçenekler ve sayısal ortamla aynı seçimler yapılır. Tohum kullanılması örneklem yanlılığını gidermek veya istatistiksel doğruluğu garanti etmek değildir.

Matplotlib’in `Agg` arka ucu, grafik için masaüstü penceresi açılmadan PNG dosyası üretir. `MPLCONFIGDIR` ise grafik önbelleğini yazılabilir geçici bir dizine yönlendirir. Bunlar istatistiksel kararlar değil çalışma ortamı ayarlarıdır.

## 5. `word_count`: sözcük sayısını nasıl bulduk?

### Amaç

Parçalanma ölçütünün paydasında sözcük sayısı gerekir. Basit `split()` kullanılsaydı boşlukla ayrılmış `...` veya `—` gibi parçalar da sözcük kabul edilebilirdi. Düzenli ifade, harf ve rakam içeren dizileri sayar.

```python
r"[^\W_]+(?:['’][^\W_]+)*"
```

İlk parça Unicode sözcük karakterlerini alt çizgi hariç yakalar; böylece Türkçe harfler ve sayılar sayılır. İkinci parça sözcük içindeki düz veya kıvrık kesme işaretinden sonra gelen harf/rakam dizisini aynı birime bağlar. `Türkiye’nin` tek birimdir. Tire ise birleştirici olarak tanımlanmadığından `sözlük-test` iki birim olur.

Örnek: `Türkiye’nin 8 güzel şehri, değil mi? ...` altı sözcük birimi verir: `Türkiye’nin`, `8`, `güzel`, `şehri`, `değil`, `mi`. Virgül, soru işareti ve üç nokta sayılmaz.

### Neden bu bir morfolojik çözümleme değil?

`evlerimizden` tek sözcük birimidir. Fonksiyon bunu `ev + ler + imiz + den` diye ayırmaz. Tokenizer’ın sözcüğü hangi alt birimlere ayırdığını da yeniden hesaplamaz. Bu operasyonel sözcük tanımıdır; sayılar, kısaltmalar, tireler ve farklı Unicode yazımları başka tanımlarla farklı sayılabilir. Kod bu olasılığı çözen kapsamlı bir Türkçe dil işlemcisi içermez.

## 6. `unique_index`, `sha`, `read_jsonl`: neden küçük yardımcı fonksiyonlar yazıldı?

### `read_jsonl(path)`

JSONL biçiminde her dolu satır bir kayıttır. Fonksiyon satırları okur, boşları atlar ve JSON nesnelerini Python sözlüğüne çevirir. Hatalı JSON varsa normal ayrıştırma hatası oluşur; bozuk satır sessizce atlanmaz.

### `unique_index(rows, label, expected=None)`

Kayıtlar `pair_id → kayıt` sözlüğüne dönüştürülür. Önce kimlik sayısı ile benzersiz kimlik sayısı karşılaştırılır. Aksi hâlde aynı kimlik iki kez gelirse Python sözlüğü önceki kaydı sessizce ezebilir ve yanlış eşleştirme fark edilmeyebilirdi.

`expected` verildiyse gözlenen ve beklenen kimlik kümeleri birebir karşılaştırılır. Bu hem eksik hem fazla kimliği yakalar. Örneğin Q4 dosyalarında 999 doğru kimlik ve bir yabancı kimlik olması, toplam satır sayısı 1.000 olsa bile kabul edilmez.

### `sha(path)`

Dosya baytlarından SHA-256 özeti üretir. Kayıtların ait olduğunu söylediği veri dosyası ile gerçekten okunan veri dosyasının aynı olup olmadığı bu yolla sınanır. Ayrıca çalışma öncesi ve sonrası eski dosyaların değişip değişmediği kontrol edilir. Hash, dosyanın bilimsel açıdan doğru olduğunu veya etiketlerin dilsel olarak geçerli olduğunu kanıtlamaz.

## 7. `load_pairs`: tüm verinin bir araya getirildiği yer

Bu fonksiyonun çıktısı iki şeydir: analiz satırları ve kullanılan kaynak dosyalarının listesi. Önce 1.000 benzersiz çift ve dört kategori olduğu kontrol edilir. Ardından her modelin kimlik kümesinin veri setiyle aynı olması gerekir.

Yaklaşık şablon CSV’si için küme kimliklerinin tekilliği, yazılı küme boyutu, çiftlerin bilinen kimlikler olması ve kategori uyumu kontrol edilir. Bir çift iki kümeye giremez; hiçbir çift kümesiz kalamaz.

Her model/çift kaydında şu kontroller yapılır:

1. Model adı ve kategori beklenen kayıtla uyuşmalı.
2. Veri hash’i ve puanlama yöntemi beklenen değerleri taşımalı.
3. İyi ve hatalı cümlenin ortalama log-olasılık farkı kayıtlı marjla uyuşmalı.
4. Marj sonlu olmalı; kararın doğruluğu marjın işaretiyle aynı olmalı.
5. Beraberlik kaydı sıfır marjla tutarlı olmalı.
6. Token sayısı pozitif tam sayı olmalı.
7. Aynı cümlenin FP16/Q8/Q4 token sayıları ve token kimliği hash’leri aynı olmalı.

Böylece “nicemleme farkı” diye yorumlanacak şeyin aslında farklı tokenlaştırılmış cümlelerden kaynaklanma riski azaltılır. Bu kontrol, hash’i saklanmış token dizilerini karşılaştırır; tokenizer’ı tekrar çalıştırmaz.

### 7.1. Karar marjı

İyi cümlenin ortalama log-olasılığı `s_iyi`, hatalı cümleninki `s_hatalı` olsun:

$$m_i=s_{i,iyi}-s_{i,hatalı}.$$

`m_i > 0` iyi cümle tercih edildi, `m_i < 0` hatalı cümle tercih edildi demektir. `|m_i|`, kararın sınırdan uzaklığını gösteren vekil ölçüdür. Bu kalibre edilmiş bir güven olasılığı değildir. Çok küçük mutlak marjda puanlardaki küçük değişiklikler kararın işaretini değiştirebilir. Bu nedenle FP16 mutlak marjı regresyonda kritik kontrol olarak yer alır.

### 7.2. Token sayısından neden BOS çıkarılmadı?

Özgün puanlama kodunda `n_tokens`, `add_special_tokens=False` ile elde edilen cümle tokenlarının sayısıdır. BOS yalnızca bağlamda kullanılmıştır; kayıtlı sayıya dahil değildir. Bu yüzden `n_tokens - 1` yapmak bir düzeltme değil, hata olurdu. EOS da bu puanlama protokolünde sayılmaz.

Noktalama tokenları ise kayıtlı cümle token sayısında kalır. Sözcük sayısı noktalamayı dışladığı için kullanılan oran tam anlamıyla yalnızca sözcük içi bölünme sayısını ölçmez. İstenen kayıtlı token sayısını kullanma koşuluna uyan, cümle düzeyinde token/sözcük yoğunluğudur.

### 7.3. Parçalanma formülleri

İyi ve hatalı cümlelerin token sayıları `T`, sözcük sayıları `W` ile gösterilsin:

$$F_{i,iyi}=T_{i,iyi}/W_{i,iyi},\qquad F_{i,hatalı}=T_{i,hatalı}/W_{i,hatalı}.$$

$$F_i=(F_{i,iyi}+F_{i,hatalı})/2.$$

Örneğin iyi cümle 16 token/8 sözcük, hatalı cümle 18 token/8 sözcük içeriyorsa oranlar 2 ve 2,25; çiftin ana ölçüsü 2,125 olur. İki cümlenin sözcük sayıları farklıysa **oranların ortalaması**, toplam tokenın toplam sözcüğe bölünmesiyle aynı olmak zorunda değildir. Burada kullanıcının istediği oranların ortalaması uygulandı.

Ayrıca şu kontroller saklandı:

$$D_i=F_{i,iyi}-F_{i,hatalı},\qquad |D_i|=|F_{i,iyi}-F_{i,hatalı}|.$$

İşaretli fark iki cümleden hangisinin daha parçalı olduğunu, mutlak fark ise ayrılığın büyüklüğünü gösterir. `difference_control` modelinde **işaretli fark** kullanılır. Mutlak fark CSV’ye yazılır ancak bu regresyona ayrıca eklenmez.

Ortalama sözcük uzunluğu `(W_iyi + W_hatalı)/2`, ortalama token uzunluğu `(T_iyi + T_hatalı)/2` olarak hesaplanır.

### 7.4. Karar değişimleri

| FP16 | Q4 | `flip_q4` | `degradation_q4` | `improvement_q4` |
|---|---|---:|---:|---:|
| Doğru | Doğru | 0 | 0 | 0 |
| Doğru | Yanlış | 1 | 1 | 0 |
| Yanlış | Doğru | 1 | 0 | 1 |
| Yanlış | Yanlış | 0 | 0 | 0 |

Bu veri setinde beraberlik yoktur; son denetim beklenmedik beraberlikte analizi durdurur. Dolayısıyla burada `flip = degradation + improvement` eşitliği geçerlidir.

Q4 için 42 bozulma, 14 iyileşme, toplam 56 değişim; Q8 için toplam üç değişim elde edilmezse betik durur. Bu sabit kontroller **sonuçları bu sayılara zorlamaz**: farklı sayılar geldiğinde sonuç üretmeyi reddeder. Betik bu nedenle başka veri setine doğrudan uygulanacak genel amaçlı bir araç değildir.

## 8. `quartiles`, `summary` ve `binomial_ci`: ilk karşılaştırmalar

### Çeyreklere ayırmanın amacı

Regresyondan önce gözlenebilir basit bir soru sorulur: Parçalanması düşük ve yüksek örneklerde karar değişimi oranları nasıl? `quartiles`, %25, %50 ve %75 eşiklerini çıkarır. Genel eşikler yaklaşık **1,857143; 2,083333; 2,339286** değerleridir.

Eşikteki eşit değerler alt grupta tutulur. Bu yüzden gruplar 250–250–250–250 değil **256–247–247–250** olur. Aynı parçalanma değerindeki çiftleri yalnızca eşit grup boyutu için bölmekten kaçınılmıştır.

Genel çeyrekler tüm veri için ortak ölçek sağlar. Kategori-içi çeyrekler ise her kategorinin kendi dağılımına göre kurulur. İkisinin anlamı farklıdır: bir kategorinin “en yüksek” grubu başka bir kategorinin “en yüksek” grubuyla aynı mutlak parçalanma aralığında olmayabilir. CSV ve JSON her iki grup bilgisini de içerir.

### Oranların paydası

`summary`, doğru sayısını grup büyüklüğüne böler. Bozulma ve iyileşme oranları da grubun tüm çiftlerine bölünür. Örneğin 13 bozulma/256 çift = %5,08’dir. Bu sayı, “FP16’nın doğru bildikleri arasındaki bozulma oranı” değildir; böyle bir koşullu oran için farklı payda gerekir.

### Kesin binom aralığı

`binomial_ci(k,n)` tek bir olayın `n` çiftte `k` kez gözlenmesi için Clopper–Pearson aralığını hesaplar. Beta dağılımının ters birikimli dağılım fonksiyonu kullanılır:

$$L=B^{-1}_{0.025}(k,n-k+1),\qquad U=B^{-1}_{0.975}(k+1,n-k).$$

`k=0` durumunda alt uç 0; `k=n` durumunda üst uç 1 alınır. Boş grupta oran ve aralık `None` olur. “Kesin” sözcüğü binom modeli altındaki kapsama yaklaşımını belirtir; kümelenmiş örneklerde bağımsızlık varsayımını kendiliğinden sağlamaz. Grafikte bu aralıklar değil küme bootstrap aralıkları kullanılır.

## 9. `design`: regresyona hangi değişkenler verildi?

`design(rows, kind)` sayısal matrisi ve sütun adlarını üretir. `x` matrisinin her satırı bir çift, her sütunu bir değişkendir. Hedef `y=flip_q4` başka aşamada hazırlanır; Q8 için regresyon kurulmaz.

| `kind` | Açıklayıcı değişkenler | Sorduğu soru |
|---|---|---|
| `unadjusted` | Parçalanma | Başka kontrol olmadan ilişki nasıl? |
| `adjusted` | Parçalanma, mutlak FP16 marjı, ortalama sözcük sayısı, kategori | Ana kontrollü ilişki nasıl? |
| `token_length` | Ana modelde sözcük sayısı yerine token sayısı | Uzunluk tanımı değişince sonuç değişiyor mu? |
| `difference_control` | Ana model + işaretli parçalanma farkı | İyi/hatalı cümleler arası fark ilişkiyi açıklıyor mu? |
| `log_margin` | Ana modelde ham mutlak marj yerine `log(|marj|+0.0001)` | Çok düşük marjları farklı ölçekle ele alınca sonuç değişiyor mu? |

Kategoriler alfabetik sıralanır; ilk kategori referans bırakılır. Tam veri için referans `argument_structure_transitive` olur. Kalan üç kategori için 0/1 gösterge sütunu eklenir. Dört kategori göstergesi ve sabit terimin birlikte kullanılması doğrusal bağımlılık yaratacağından biri dışarıda bırakılır.

Sözcük sayısı ana kontrol seçildi: token sayısı zaten parçalanma oranının payında olduğu için iki değişken matematiksel olarak bağlantılıdır. Token uzunluğu kontrolü de makul başka bir soruyu yanıtlar ve ayrıca hesaplandı. İki kontrolün aynı şeyi ölçtüğü varsayılmadı.

## 10. `fit_ridge`: lojistik regresyonun matematiği

### 10.1. Neden lojistik regresyon?

Hedef 0 veya 1’dir: karar değişmedi/değişti. Model, açıklayıcı değişkenler verildiğinde değişim olasılığını 0–1 aralığında tahmin eder:

$$p_i=\sigma(\eta_i)=\frac{1}{1+e^{-\eta_i}},\qquad \eta_i=\beta_0+\sum_j \beta_j z_{ij}.$$

`expit`, sigmoid fonksiyonunun sayısal olarak güvenli uygulamasıdır. Burada tahmin amaçlı test başarısı veya çapraz doğrulanmış sınıflandırıcı geliştirilmedi; değişkenler arası koşullu ilişki özetlendi. Aynı veride model kurmak, nedensel etki tahmin edildiği anlamına gelmez.

### 10.2. Standardizasyon

Her sütunun ağırlıklı ortalaması ve standart sapması hesaplanır:

$$\mu_j=\frac{\sum_i w_i x_{ij}}{\sum_i w_i},\qquad s_j=\sqrt{\frac{\sum_i w_i(x_{ij}-\mu_j)^2}{\sum_i w_i}}.$$

$$z_{ij}=(x_{ij}-\mu_j)/s_j.$$

Bu, farklı birimdeki değişkenlerin ridge cezasıyla daha karşılaştırılabilir ölçeklerde ele alınmasını sağlar. Kod kategori göstergelerini de standartlaştırır; yalnızca sürekli değişkenleri değil. Sabit sütunda standart sapma sıfıra yaklaşırsa 1 kabul edilir; sıfıra bölünme önlenir. Başlangıç örnekleminde ağırlıklar 1’dir. Bootstrap’te ağırlıklar değiştiği için ortalama ve standart sapma her tekrarda yeniden hesaplanır.

### 10.3. Amaç fonksiyonu ve ceza

Küçültülen fonksiyon şudur:

$$L(\beta)=\sum_i w_i[\log(1+e^{\eta_i})-y_i\eta_i]+\frac{\lambda}{2}\sum_{j\geq1}\beta_j^2.$$

İlk terim veriye uyumu ölçen negatif log-olabilirliktir. İkinci terim büyük eğim katsayılarını cezalandırır. Sabit `β₀` cezalandırılmaz. `np.logaddexp(0, eta)` doğrudan `log(1+exp(eta))` hesaplamasının büyük sayılarda taşma sorununu azaltır.

Ana `λ=1` seçilmiştir; veri içinden en iyi λ’yı seçen çapraz doğrulama yapılmamıştır. `λ=0,1` ve `λ=10` duyarlılıkları ayrıca hesaplanır. Ceza toplam log-kayıp ölçeğinde tanımlıdır; başka bir yazılımın ortalama kayıp kullanan `lambda` parametresiyle doğrudan eşit olduğu varsayılamaz.

Ridge kullanma gerekçesi yalnızca 56 değişim olayı bulunması ve bazı kategorilerde çok az olay olmasıdır. Ayrışma veya çok büyük katsayılar riskini azaltır. Bunun bedeli katsayıların sıfıra doğru çekilmesi, yani düzenlileştirme yanlılığıdır. Bootstrap bu yanlılığı otomatik kaldırmaz.

### 10.4. `objective`, `gradient`, `hessian`

Bu üç iç fonksiyon aynı amaç fonksiyonunu farklı biçimlerde optimizer’a sunar:

- `objective(b)`: mevcut katsayılarla kaybın değerini verir.
- `gradient(b)`: kaybın katsayılara göre birinci türevini verir; hangi yönde azalacağını gösterir.
- `hessian(b)`: ikinci türev matrisidir; kayıp yüzeyinin eğriliğini gösterir.

Sabit sütunu da içeren matris `Z`, ceza matrisi `R=diag(0,λ,…,λ)` olmak üzere:

$$\nabla L=Z^T[w\odot(p-y)]+R\beta.$$

$$\nabla^2L=Z^T\operatorname{diag}(w\odot p\odot(1-p))Z+R.$$

`⊙`, eleman bazında çarpımdır. `Newton-CG`, bu türevlerle katsayıları günceller. Başlangıç sabiti gözlenen ağırlıklı olay oranının log-odds değeridir: `log(rate/(1-rate))`. Bir yeniden örnekte tüm sonuçlar aynıysa sonlu başlangıç ve anlamlı iki sınıflı ilişki kurulamayacağından fonksiyon hata verir.

Son kontrolde katsayıların sonlu olması ve mutlak gradyanın en büyük bileşeninin `1e-5` altında kalması istenir. Yalnızca optimizer’ın başarı mesajına güvenilmez. Başarısız tekrar sessizce atlanmaz; çalışma durur. Mevcut çalışmada böyle bir başarısızlık olmadı.

## 11. `effect` ve `model_result`: katsayı nasıl okunur?

`fit_ridge` standartlaştırılmış katsayı döndürür. `effect`, parçalanmanın katsayısını standart sapmaya bölerek özgün birime geri çevirir:

$$b_F=\beta_F/s_F.$$

Bir token/sözcük artış için odds oranı `exp(b_F)`, özgün örneklemin bir standart sapmalık artışı için `exp(b_F × SS_F)` olur. Bu veride `SS_F ≈ 0,3977 token/sözcük`.

Odds, `p/(1-p)` değeridir; olasılıkla aynı şey değildir. OR=0,735, bu model ve kontrol değişkenleri altında bir standart sapmalık artışa karşılık odds’un yaklaşık %26,5 daha düşük tahmin edilmesi demektir. **Karar değişimi olasılığı %26,5 azalır** demek değildir.

Yalnızca açıklayıcı bir örnek: başlangıç olasılığı %10 ise odds `0,10/0,90=0,1111` olur. Bunu 0,735 ile çarparız: yaklaşık 0,0817. Olasılığa geri dönüş `0,0817/(1+0,0817)≈%7,55` verir. Bu örnek gerçek bir çift için hesaplanmış tahmin değildir; OR yorumunu öğretmek içindir.

`model_result`, modelin sütun adlarını, katsayılarını, ortalama/ölçeklerini ve iki birimdeki OR değerlerini JSON’a uygun biçimde toplar. Böylece yalnızca nihai sayıyı değil sayıya nasıl ulaşıldığını da denetlemek mümkündür.

## 12. `resample_weights`: neden yazıldı, tam olarak ne yapıyor?

Bu fonksiyonun amacı “aynı tür örnekler yeniden seçilseydi tahmin ne kadar değişirdi?” sorusunu incelemektir. Yeni model puanları üretmez. Mevcut çiftleri veya kümeleri yerine koyarak yeniden seçmenin ağırlıklarını üretir.

### 12.1. Neden satırları kopyalamak yerine ağırlık?

Dört çiftin A, B, C, D olduğunu düşünelim. Bir bootstrap seçimi A, A, C, D ise eşdeğer ağırlıklar `[2,0,1,1]` olur. A’nın kayıp katkısı iki, B’ninki sıfır kez sayılır. Her çiftin FP16 ve Q4 bilgileri aynı satırda kaldığı için eşleşme bozulmaz. FP16 ve Q4 sonuçları birbirinden bağımsız örneklenmez.

### 12.2. `cluster=False`: kategori tabakalı çift bootstrap

Her kategori kendi içinde yeniden örneklenir. Bir kategoride `n_c` çift varsa, eşit seçilme olasılıklarıyla `n_c` seçim yapılır:

$$ (w_1,\ldots,w_{n_c})\sim\operatorname{Multinomial}(n_c;1/n_c,\ldots,1/n_c).$$

Ağırlıklar tam sayıdır ve toplamları `n_c` olur. Bu veri setinde kategoriler 250’şer çift içerdiğinden her kategorinin toplam ağırlığı 250 kalır. Böylece kategori bileşimindeki rastlantısal değişim yerine kategoriler içindeki örnekleme belirsizliği değerlendirilir.

### 12.3. `cluster=True`: yaklaşık şablon kümelerini birlikte seçmek

Aynı şablonun küçük değişikliklerinden gelen çiftler bağımsız olmayabilir. Onları bağımsız seçmek belirsizliği olduğundan küçük gösterebilir. Bu yüzden ikinci yaklaşımda seçim birimi tek çift değil yaklaşık kümedir.

Bir kategoride `K_c` küme varsa yerine koyarak `K_c` küme seçilir. Küme üç kez seçildiyse içindeki her çift önce ağırlık 3 alır; hiç seçilmediyse içindeki tüm çiftler ağırlık 0 alır. Böylece küme üyeleri birlikte hareket eder.

**Somut örnek:** Bir kategoride iki küme olsun. A kümesinde iki, B kümesinde bir çift bulunsun. İki küme seçimi A, A çıkarsa ham çift ağırlıkları `[2,2,0]`, toplam 4 olur. Özgün kategori üç çift içeriyordu. Kod ağırlıkları `3/4` ile çarpar ve `[1,5; 1,5; 0]` elde eder. Toplam yeniden 3’tür.

Genel formül:

$$w_i^*=w_i\frac{n_c}{\sum_{j\in c}w_j}.$$

Küme boyutları farklı olduğu için ham toplam değişebilir; bu normalizasyon kategorinin genel analizdeki ağırlığını korur. Bu nedenle küme bootstrap ağırlıkları tam sayı olmak zorunda değildir. Hesaplanan yaklaşım, kategori büyüklüğünü sabit tutan normalize edilmiş küme bootstrap duyarlılığıdır; her tür olası örnekleme tasarımını kapsadığı iddia edilmez.

### 12.4. Kümeler nereden geldi ve neyi garanti etmez?

Mevcut analizde **776 yaklaşık küme** vardır. Önceki çalışmada ortak baştaki söylem/zaman sözcükleri çıkarıldıktan sonra kalan cümle çiftlerinin benzerliği üzerinden oluşturulmuştur. Bu betik kümeleri yeniden keşfetmez; CSV’deki üyelikleri kullanır ve tutarlılığını denetler.

Bu gruplar gerçek veri üretim şablonlarının doğrulanmış kimlikleri değildir. Dolayısıyla küme bootstrap, yalnızca bu yaklaşık grupların yakaladığı bağımlılığa karşı duyarlılık incelemesidir. Başka kümeler arası bağımlılık devam edebilir.

## 13. `main` içindeki bootstrap döngüsü: her tekrarda ne yeniden hesaplandı?

Çift yaklaşımında 2.000, küme yaklaşımında 2.000 örnekleme yapılır. Her örneklemede:

1. `resample_weights` yeni ağırlıkları üretir.
2. Aynı ağırlıklar beş modelin tamamına uygulanır.
3. `fit_ridge`, ağırlıklı standardizasyonu ve katsayıları yeniden tahmin eder.
4. `effect`, parçalanma katsayısını özgün birime döndürür.
5. Her özgün çeyrekte ağırlıklı karar değişimi oranı hesaplanır.
6. En yüksek ve en düşük çeyrek arasındaki fark kaydedilir.

**Önemli ayrım:** 4.000 yeniden örnekleme vardır; beş model her örneklemede yeniden tahmin edildiğinden bootstrap içinde toplam **20.000 regresyon uyumu** yapılır. Buna özgün veri üzerindeki başlangıç ve diğer duyarlılık uyumları dahil değildir.

Çeyrek eşikleri her tekrarda yeniden hesaplanmaz. Başlangıçta tanımlanan grupların oranları değerlendirilir. Bu yüzden aralıklar, çeyrek sınırlarının seçilmesine ilişkin tüm belirsizliği kapsamıyor; mevcut gruplara koşullu belirsizliği gösteriyor.

Çift ve küme yaklaşımlarında farklı rastgele akışlar kullanılır: `SEED` ve `SEED+1`. Örnekleme içinde beş model için aynı ağırlıkların kullanılması, modeller arasında karşılaştırılabilir yeniden örnekler sağlar.

## 14. `interval`: güven aralıkları nasıl oluştu?

Bootstrap sonucunda aynı ölçütün 2.000 farklı tahmini elde edilir. %2,5 ve %97,5 yüzdelikleri alınır. Bu **yüzdelik bootstrap aralığıdır**; BCa veya standart hata üzerinden normal yaklaşım değildir.

Regresyon için önce özgün birimdeki eğim dağılımının yüzdelikleri alınır, sonra aynı özgün standart sapmayla çarpılıp üstel dönüşüm uygulanır. Böylece her tekrarın değişen standart sapmasının OR birimini değiştirmesi önlenir.

OR aralığında 1 nötr değerdir. Yüzde puan farkının aralığında 0 nötr değerdir. Aralığın nötr değeri içermemesi, kullanılan model ve örnekleme varsayımları altında bir yönü destekler; nedensellik veya her model seçeneğinde sağlamlık göstermez. Nötr değeri içermesi de ilişkinin kesinlikle sıfır olduğunu kanıtlamaz.

%95 güven aralığını, “gerçek değer %95 olasılıkla bu belirli aralıkta” diye yorumlamıyoruz. Bu, varsayımlar altında tekrar örneklemelerde aralık kurma yönteminin kapsama davranışına ilişkin bir ifadedir. Ridge, sınırlı olay sayısı ve yaklaşık kümeler nedeniyle bu kapsama da kusursuz bir garanti olarak ele alınmaz.

## 15. Duyarlılık analizleri neden yapıldı?

Tek bir modelin katsayısını sunmak bazı yöntem seçimlerini görünmez kılabilir. Bu nedenle uzunluk tanımı, marjın ölçeği, çift içi parçalanma farkı ve ceza gücü değiştirildi. Ayrıca her seferinde bir kategori çıkarılarak ilişki yalnızca bir kategoriye mi bağlı diye bakıldı.

Beş ana model için çift ve küme bootstrap aralıkları vardır. `λ=0,1/10` ve kategori çıkarma analizleri için **yalnızca nokta tahmini** hesaplandı; bunlar için aralık varmış gibi yorum yapılmamalıdır.

Kategori çıkarma analizinde parçalanmanın standart sapması kalan alt veri üzerinde hesaplanır. Bu nedenle “bir SS başına OR” değerleri tam olarak aynı fiziksel artış biriminde değildir. Kesin bir büyüklük karşılaştırmasında JSON’daki token/sözcük birimi başına OR kullanılmalıdır. Buradaki tablo özellikle yönün değişip değişmediğini anlamak içindir.

`log_margin` modelindeki `0,0001`, sıfırda logaritma sorunu oluşmasını önleyen sabittir. Bu seçimin de bir yöntem tercihi olduğu unutulmamalıdır. Modelde kategori × parçalanma etkileşimleri, spline veya başka doğrusal olmayan parçalanma eğrileri kurulmadı.

## 16. Gözlenen sonuçlar ve grafiğin okunması

### 16.1. Ana toplamlar

| Model | Doğru çift | Doğruluk |
|---|---:|---:|
| FP16 | 912/1000 | %91,2 |
| Q8 | 915/1000 | %91,5 |
| Q4 | 884/1000 | %88,4 |

Q4’te 42 doğru→yanlış ve 14 yanlış→doğru geçiş vardır. Toplam değişim 56, net doğru kaybı 28’dir. **56 değişim ile 28 net kayıp farklı niceliklerdir.** Regresyonun hedefi 56 değişim olayıdır. Q8’in üç olayı yalnızca tanımlayıcı tutulmuştur.

### 16.2. Parçalanma çeyrekleri

| Parçalanma grubu | Çift | FP16 doğruluk | Q4 doğruluk | Bozulma | İyileşme | Toplam değişim |
|---|---:|---:|---:|---:|---:|---:|
| En düşük | 256 | %91,02 | %86,72 | 13 | 2 | 15 (%5,86) |
| Düşük-orta | 247 | %93,52 | %91,50 | 8 | 3 | 11 (%4,45) |
| Yüksek-orta | 247 | %88,66 | %87,04 | 13 | 9 | 22 (%8,91) |
| En yüksek | 250 | %91,60 | %88,40 | 8 | 0 | 8 (%3,20) |

![Alt sözcük parçalanma çeyreklerine göre Q4 karar değişimi; çizgiler kategori tabakalı yaklaşık küme bootstrap %95 aralıklarını gösterir.](/Users/omerfarukozturk/Desktop/turkce-quantization/results/fragmentation_analysis/fragmentation_plot.png)

**Grafiğin üretildiği yer:** `plot(s)`. Çubuk yüksekliği `flip_q4` oranının 100 ile çarpımıdır. Çizgilerin alt ve üst uçları `bootstrap → cluster → quartile_flip_rate_ci95` alanından alınır. Çubuk üzerindeki `15/256`, o grupta 256 çiftin 15’inde karar değiştiğini gösterir.

Bu grafik FP16/Q4 doğruluğunu veya yalnızca bozulmayı göstermiyor. İki yöndeki değişimi birlikte gösteriyor. En yüksek oran üçüncü gruptadır; çeyrekler boyunca tek yönlü düzenli artış veya azalış yoktur.

En yüksek−en düşük grup farkı yaklaşık **−2,66 yüzde puan**; küme bootstrap %95 aralığı **[−6,50; 1,41] yüzde puan**. Aralık 0’ı içerdiğinden yalnızca bu uç grup karşılaştırmasıyla net fark iddia edilmez.

### 16.3. Regresyon sonuçları

| Model | Bir SS parçalanma artışı için OR | Çift bootstrap %95 GA | Küme bootstrap %95 GA |
|---|---:|---|---|
| Kontrolsüz | 0,789 | [0,611; 0,982] | [0,592; 1,027] |
| Marj + kategori + sözcük uzunluğu kontrollü | 0,735 | [0,548; 0,932] | [0,540; 0,946] |
| Token uzunluğu kontrolü | 0,900 | JSON’da mevcut | [0,649; 1,230] |
| Çift içi işaretli fark kontrolü | 0,742 | JSON’da mevcut | [0,544; 0,952] |
| Logaritmik marj kontrolü | 0,737 | JSON’da mevcut | [0,546; 0,953] |

Kontrolsüz modelin çift aralığı 1’in altında kalırken küme aralığı 1’i içeriyor. Bu, örnekler arasındaki bağımlılığın yorum için neden önemli olduğunu gösterir. Ana kontrollü model negatif yönü destekliyor; token uzunluğu kontrolünde ise nötr değer yeniden aralığa giriyor.

Bu nedenle “parçalanma nicemlemeye karşı koruyucudur” sonucu çıkarılmadı. Desteklenmeyen beklenti, daha fazla parçalanmanın daha fazla karar değişimiyle ilişkili olduğuydu. Gözlenen ters ilişki ise model seçimine duyarlı, keşifsel bir bulgudur.

### 16.4. Kategoriler neden ayrıca incelendi?

Genel eğilim farklı dilbilgisel kategorilerin karışmasından kaynaklanabilir. Kod hem her kategorinin genel özetini hem genel çeyreklerle kesişimini hem kategori-içi çeyreklerini üretir.

Örneğin özne–yüklem uyumu kategorisinde en düşük parçalanma grubunda 83 çiftin 13’ünde, en yüksek grupta 77 çiftin hiçbirinde değişim vardır. Adlaştırmada ise karşılık gelen gruplarda 61 çiftte 1 ve 68 çiftte 6 değişim bulunur. Bu tanımlayıcı örüntüler aynı değildir. Düzensiz biçimler kategorisindeki toplam değişim olayı yalnızca birdir; bu kategoride güçlü ilişki iddiaları için veri çok sınırlıdır.

Bir kategoriyi çıkaran modellerde parçalanma OR nokta tahminleri yaklaşık 0,668–0,749 arasında, aynı yöndedir. Bu, tek kategoriye bağımlılığı sorgulayan bir kontroldür; kategoriye özel eğimlerin eşitliğini veya tüm kategorilerde anlamlı ilişkiyi kanıtlamaz. Ayrıntılı kategori tabloları kısa rapor ve JSON’da korunmuştur.

## 17. `report`: hangi kısmı otomatik, hangi kısmı açıklama?

`report(s)` hesaplanmış istatistik sözlüğünü alır. İçindeki `pct` yardımcı fonksiyonu oranları yüzde olarak biçimler; `table` fonksiyonu grup sözlüklerinden Markdown tabloları oluşturur. Modeli tekrar tahmin etmez.

Ana sonuç cümlesinin bir bölümü kontrollü küme OR aralığının 1’e göre konumundan seçilir. Diğer yorum paragrafları bu sabit 1.000 çiftlik çalışmanın bulgularına özgü yazılmıştır. Dolayısıyla bu fonksiyon her yeni veri seti için kendi kendine bilimsel yorum üreten genel bir sistem değildir. Veri veya yöntem değişirse açıklamalar ayrıca gözden geçirilmelidir.

Bu `ProjeMarkdown` belgesi öğretici bir ek dokümandır; `main()` tarafından otomatik yazılmaz. Betik yeniden çalıştığında sayısal sonuçlar değişirse buradaki tablolar ve yorumlar da bilinçli olarak güncellenmelidir.

## 18. CSV ve JSON içinde bilgi arama rehberi

### CSV alan aileleri

| Alan | Anlam |
|---|---|
| `pair_id`, `category`, `cluster_id` | Çiftin, dilbilgisel grubun ve yaklaşık kümenin kimliği |
| `sentence_good`, `sentence_bad` | Özgün etiketli cümleler |
| `fp16_correct`, `q8_correct`, `q4_correct` | Veri etiketi açısından doğru tercih: 1/0 |
| `*_decision` | `good`, `bad` veya beraberlik etiketi; mevcut veride beraberlik yok |
| `*_margin` | İyi−hatalı ortalama log-olasılık farkı |
| `*_good_tokens`, `*_bad_tokens` | Her modelin kayıtlı, BOS hariç token sayıları |
| `good_words`, `bad_words` | Düzenli ifadeyle sayılmış sözcük birimleri |
| `good_fragmentation`, `bad_fragmentation` | Cümle bazında token/sözcük oranı |
| `fragmentation` | İki cümlenin oranlarının ortalaması |
| `fragmentation_difference`, `fragmentation_abs_difference` | İşaretli ve mutlak çift içi fark |
| `mean_words`, `mean_tokens`, `fp16_abs_margin` | Kontrol değişkenleri |
| `flip_*`, `degradation_*`, `improvement_*` | Toplam değişim, bozulma, iyileşme göstergeleri |
| `quartile`, `category_quartile` | Genel ve kategori-içi çeyrek numarası |

### JSON anahtarları

`overall` genel toplamları; `quartiles` dört genel grubu; `categories` kategori karşılaştırmalarını içerir. `models` özgün veri üzerinde katsayıları ve OR değerlerini, `bootstrap` yeniden örnekleme aralıklarını taşır. `penalty_sensitivity` ve `leave_one_category_out` alternatif nokta tahminleridir. `correlations`, parçalanmanın marj ve sözcük uzunluğuyla Pearson korelasyonlarını tanımlayıcı olarak saklar; bunlar ana regresyonun yerine geçmez.

`source_sha256` kaynak kimliklerini, `script_sha256` çalıştırılan betiğin kimliğini, `versions` kullanılan sayısal ortamı ve `protected_files_unchanged` koruma denetimini kaydeder.

## 19. Neler doğrulandı ve bunlar neyi kanıtlar?

`test_fragmentation.py` dört temel testi içerir: sözcük sayımı örnekleri, eksik/tekrarlı kimlik reddi, ilişkisiz simetrik sentetik veride sıfır katsayı ve optimizer çözümünde küçük gradyan.

`verify_outputs.py` gerçek veride 1.000 satırı, 42/14/56/3 geçiş sayılarını, kaynak hash’lerini, parçalanma aritmetiğini ve çeyrek toplamlarını denetler. Ana katsayıları farklı bir optimizasyon yöntemi olan BFGS ile yeniden hesaplayıp Newton-CG sonucu ile karşılaştırır. Geçici dosya kopyalarında bir kayıt silme/tekrarlama senaryolarının yükleyici tarafından reddedildiğini sınar. Geçici kopyalar kullanıldığı için gerçek sonuç kayıtları bozulmaz.

Bu testler yazılımın belirli mantıksal ve sayısal özelliklerini destekler. Araştırma tasarımının tüm varsayımlarını, etiketlerin dilsel geçerliliğini veya Türkçeye genellenebilirliği ispatlamaz.

Yorum satırları eklenirken önceki ve sonraki Python kodunun soyut sözdizim ağaçları (AST) karşılaştırıldı. Yalnızca yorum eklendiği, çalıştırılabilir hesaplama yapısının aynı kaldığı bu yolla denetlendi. Betik baytları değiştiği için betiğin SHA-256 değeri doğal olarak değişir; ek analiz yeniden çalıştırılarak üretim kaydı güncellenir. Bu, ana model puanlamasının tekrarı değildir.

## 20. Yeniden çalıştırma

Proje kökünde aşağıdaki komutlar kullanılır:

```bash
.venv/bin/python scripts/analyze_fragmentation.py --bootstrap 2000
.venv/bin/python results/fragmentation_analysis/test_fragmentation.py
.venv/bin/python results/fragmentation_analysis/verify_outputs.py
```

İlk komut ek analizin CSV, JSON, PNG ve kısa raporunu yeniden yazar. Ana deneyin FP16/Q8/Q4 kayıtlarını değiştirmez. `--bootstrap` her örnekleme türündeki tekrar sayısıdır; 2000 toplamda 4000 yeniden örnekleme anlamına gelir. Betik 200’den düşük değeri reddeder. Teslimat doğrulayıcısı özellikle 2000 tekrar kullanılan bu teslimatı kontrol eder; farklı tekrar sayısıyla üretilmiş çıktıda bu kontrolün başarısız olması beklenir.

NumPy, SciPy ve Matplotlib mevcut sanal ortamdan kullanılır. Bu işlem için EuroLLM ağırlıkları veya GPU gerekmez. İlerleme her 500 tekrarda terminale yazdırılır. Herhangi bir geçersiz eşleştirme veya sayısal yakınsama problemi varsa hata alınır; sessizce eksik örneklemle devam edilmez.

## 21. Bu çalışmanın sınırları ve sonraki bilimsel işler

Ölçülen değişken morfem veya ek sayısı değildir. Noktalama tokenlarının katkısını içerir. Sonuç aynı model ailesi, bu veri seti, mevcut etiketler ve puanlama protokolü için geçerlidir. Sözcük tanımı, uzunluk kontrolü, marjın fonksiyonel biçimi, ridge cezası ve yaklaşık kümeler yorumun parçasıdır.

P-değerleri üretilmedi; çoklu keşifsel karşılaştırmalar için düzeltilmemiş %95 aralıklar sunuldu. “Hiç p-değeri yok, o hâlde çoklu karşılaştırma sorunu yok” denemez. Alternatif modellere bakıp yalnızca en uygun sonucu seçmek yine yanıltıcı olabilir; bu nedenle ana ve alternatif sonuçlar birlikte raporlandı.

Bağımsız uzmanlarla kör etiket incelemesi ve gerçek üretim şablonlarının doğrulanması yapılmadı. Bunlar ayrı bilimsel işlerdir. Bu ek analiz onları tamamlamış sayılmaz. Makale yazımı da burada otomatik olarak gerçekleştirilmedi.

## 22. Yöntem kaynakları ve ilgili belgeler

- [SciPy bootstrap belgesi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html): yeniden örnekleme ve yüzdelik aralık yaklaşımının genel tanımı. Bu projede ağırlıklar özel kategori/küme tasarımına göre elle üretilir; `scipy.stats.bootstrap` doğrudan çağrılmaz.
- [SciPy minimize belgesi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html): `minimize`, gradyan ve Hessian arayüzü.
- [Önceki ayrıntılı deney raporu](/Users/omerfarukozturk/Desktop/turkce-quantization/ProjeMarkdown/ISTATISTIKSEL_ANALIZ_VE_DENEY_RAPORU.md).
- [Bu ek analizin kısa sonuç raporu](/Users/omerfarukozturk/Desktop/turkce-quantization/results/fragmentation_analysis/fragmentation_report.md).
