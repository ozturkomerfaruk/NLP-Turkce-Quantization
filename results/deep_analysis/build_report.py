"""Render the Turkish research narrative from verified numeric artifacts."""
import json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
R=json.loads((OUT/'extended_statistics.json').read_text())
S=json.loads((OUT/'recomputed/main_statistics.json').read_text())
N=['fp16','q8','q4'];C=list(R['category'])
CN=dict(subject_agreement='Özne–yüklem uyumu',argument_structure_transitive='Argüman yapısı / durum eki seçimi',nominalization='Adlaştırma',irregular_forms='Düzensiz biçimler')
parts=[]
def add(s):parts.append(s.strip()+'\n')
def f(x,d=3):return f'{x:.{d}f}'.replace('.',',')
def pv(x):return f'{x:.8g}'.replace('.',',')
def interval(x):return '['+f(x[0])+'; '+f(x[1])+']'
def table(headers,rows):
 add('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows))
def link(p,label=None):return f'[{label or p}]({ROOT/p})'
def fig(name,caption):add(f'![{caption}]({OUT}/figures/{name}.png)\n\n{caption}')
add('''# EuroLLM-1.7B Türkçe nicemleme deneyi: ayrıntılı istatistiksel analiz ve yeniden üretim raporu

**Rapor tarihi:** 28 Ekim 2026, Türkiye saati. **Kapsam:** aynı 1.000 TurBLiMP minimal çiftinin FP16, Q8 ve Q4 ile karşılaştırılması. **Ana değerlendirme:** token başına ortalama log-olasılık. **Analiz türü:** mevcut ana deneyin bağımsız denetimi, model puanlamasının ayrı klasörde tekrarı, eşleştirilmiş istatistik ve ön dilbilimsel inceleme.

Bu belge önceki durum notlarını değiştirmeyen yeni bir araştırma raporudur. Kullanıcının sağladığı Aşama 5 metni bir taslak olarak ele alındı; rakamları dosyalardan yeniden hesaplandı. İnsan uzman tarafından onaylanmamış dilbilimsel değerlendirmeler açıkça ön inceleme olarak belirtilmiştir. “Doğru” sözcüğü, aksi belirtilmedikçe veri setindeki iyi etiketli cümlenin daha yüksek puan alması anlamındadır; evrensel dilbilgisel doğruluğun ölçüldüğü iddiası değildir.

## 1. Araştırma sorusu ve en önemli sonuç

Bu çalışma, EuroLLM-1.7B’nin yerel FP16 sürümünü aynı modelin 8 ve 4 bit nicemlenmiş sürümleriyle karşılaştırıyor. Soru, ağırlıkların daha az bellek kaplayacak biçimde temsil edilmesinin Türkçe minimal çiftlerdeki tercihleri ve kaynak tüketimini nasıl değiştirdiğidir. Üç farklı model ailesi, eğitim yöntemi veya veri seti karşılaştırılmıyor.

Ana benchmark sonucunda FP16 912, Q8 915, Q4 884 doğru karar verdi. Q8’in üç ek doğru kararı istatistiksel üstünlük göstermiyor. Q4’ün 28 net doğru kaybı, öncelikli iki FP16 karşılaştırmasına uygulanan Holm düzeltmesinden sonra da anlamlı. Ancak bu çıkarımın örneklerin bağımsızlığına ilişkin varsayımları var; benzer cümle kümeleri, kategori bileşimi ve etiket geçerliliği aşağıda ayrıca incelendi.

Önemli ayrım: bu sabit 1.000 çiftteki doğruluk oranları kesin sayımlardır. Güven aralıkları, bu örneklemin benzer bir örnek üretim sürecinin temsili kabul edilmesi durumundaki örnekleme belirsizliğine ilişkindir. Aralıklar başka model ailelerine, tüm Türkçeye veya farklı uygulama görevlerine otomatik olarak taşınamaz.''')
table(['Ölçüt','FP16','Q8','Q4'],[
 ['Doğru / toplam']+[f"{R['accuracy'][n]['correct']}/1000" for n in N],
 ['Doğruluk (%)']+[f(R['accuracy'][n]['accuracy_percent'],1) for n in N],
 ['Doğruluk için %95 Clopper–Pearson GA (%)']+[interval(R['accuracy'][n]['ci95_percent']) for n in N],
 ['Özgün puanlama süresi (sn)']+[f(R['resources'][n]['seconds'],2) for n in N],
 ['Özgün MLX tepe bellek (GB)']+[f(R['resources'][n]['mlx_peak_gb']) for n in N],
 ['Ağırlık dosyaları (GB)']+[f(R['resources'][n]['weights_gb']) for n in N]])
add('''“GA” güven aralığı, “yp” yüzde puan anlamındadır. GB, 10⁹ bayt olarak hesaplandı. Yukarıdaki kaynak değerleri özgün deney kayıtlarıdır; bu oturumdaki tekrar ölçümleri ayrıca verilir. Doğruluk için ayrı ayrı hesaplanan aralıkların üst üste gelmesi, eşleştirilmiş fark testinin yerine geçmez.''')
fig('01_accuracy','Şekil 1. Ana yöntem doğrulukları; çubuklar %95 kesin binom aralıklarını gösterir. Bu aralıklar model farkı aralıkları değildir.')
add('''## 2. Hangi dosya ne yapıyor?

Kod satırlarını tekrar etmek yerine deneyin işleyişini ve sorumluluklarını aşağıda açıklıyorum. Ana deney kodu değiştirilmedi; ek araştırma hesapları ayrı dosyada tutuldu.''')
table(['Dosya','Deneydeki görev'],[
 [link('scripts/prepare_dataset.py'),'Dört ham CSV’yi okur; boş/eşit cümleleri ve birebir tekrar çiftleri ayıklar; sabit rastgele tohumla her kategoriden 250 çift seçer.'],
 [link('scripts/finalize_pilot_dataset.py'),'Ana veri hash’ini doğrular; ana verinin içinden kategori başına 5 pilot çift seçer; veri ve kaynak hash’lerini manifest dosyasına yazar.'],
 [link('scripts/score_pilot.py'),'20 çiftlik pilotta BOS bağlamını, token hedeflerini ve ortalama/toplam log-olasılık hesabını uygular. Ana veri setinden bağımsız bir test kümesi oluşturmaz.'],
 [link('scripts/score_main.py'),'Aynı puanlama kuralıyla 50 çiftlik 20 parça çalıştırır. Her cümleyi tek başına puanlar; sonuçları ve süre/bellek kayıtlarını yazar; tamamlanmış parçaları atlar.'],
 [link('scripts/summarize_main.py'),'Üç modelin tam kayıtlarını, kimlikleri ve token eşleşmelerini kontrol eder; sayımları ve kaynak toplamlarını çıkarır.'],
 [link('scripts/analyze_main.py'),'Kesin McNemar, iki testlik Holm, muhafazakâr eşleştirilmiş fark aralıkları, kategori ve puanlama duyarlılığı üretir; 20 inceleme çifti seçer.'],
 [link('tests/test_analyze_main.py'),'Gerçek kayıtların yüklenmesini ve sıfırdan büyük küçük puan farklarının pilot protokolüne uygun kabul edilmesini sınar.'],
 [link('results/deep_analysis/analyze_extended.py'),'Bu rapor için kaynak yeniden seçimini, tokenları, sayısal özdeşlikleri ve SciPy karşılaştırmasını denetler; bootstrap, kümeler, etiket/uzunluk/kategori duyarlılıkları, grafikler ve inceleme tablolarını üretir.'],
 [link('results/deep_analysis/rerun_models.py'),'Var olan score_main uygulamasını kullanır; yalnızca çıktı kökünü ayrı doğrulama klasörüne yönlendirir.'],
 [link('results/deep_analysis/build_report.py'),'Doğrulanmış JSON ve CSV’lerden bu ayrıntılı Markdown raporunu üretir.']])
add('''## 3. Veri nereden geliyor ve gerçekten 1.000 farklı çift var mı?

Ham kaynak sürümü manifest içinde `297de13fb7a0ce524fe32e8b175c6b5255d66960` olarak sabitlenmiş. Bu oturumda yeni bir sürüm indirilmedi. Yerel ham dosyaların hash’leri manifest ile karşılaştırıldı. Ham CSV’lerden temizleme ve seçme işlemi bellekte yeniden yürütüldü; aynı kategori sırası, `random.Random(42)` ve aynı örnekleme yöntemi kullanıldığında elde edilen kayıtlar ana JSONL ile içerik ve sıra bakımından birebir eşleşti.

Bu, eldeki ham dosyalardan aynı örneklemin üretilebildiğini doğrular. Yerel ham dosyaların uzak depodaki sürümle birebir eşitliği bu oturumda yeniden indirme yoluyla doğrulanmadı; kaynak sürüm bilgisi ve yerel manifest üzerinden izleniyor. Hash, kaynağın anlamsal doğruluğunu değil dosya bütünlüğünü sınar.''')
table(['Kategori','Ham satır','Birebir tekrar','Geçerli benzersiz havuz','Seçilen','Havuzdan örnekleme oranı'],[[CN[c],v['original_rows'],v['duplicates_within_category'],v['unique_available'],250,'%'+f(250/v['unique_available']*100,2)] for c,v in R['audit']['raw_stats'].items()])
add(f'''Ham verilerde toplam {sum(v['duplicates_within_category'] for v in R['audit']['raw_stats'].values())} kategori içi birebir tekrar ayıklandı. İncelenen dört dosyada boş/eşit cümle nedeniyle çıkarılan satır ve kategoriler arası birebir çift tekrarı yok. Seçilmiş örneklemde 1.000 benzersiz çift kimliği, 1.000 benzersiz iyi cümle ve 1.000 benzersiz kötü cümle var. Ancak birebir benzersizlik, dilsel şablonların bağımsız olduğu anlamına gelmiyor.

Ana veri SHA-256: `{R['dataset_sha256']}`.

Dört kategori eşit büyüklükte olduğundan genel mikro doğruluk ile bu dört kategorinin basit makro ortalaması aynı. Bu ağırlıklandırma gerçek hayattaki Türkçe cümlelerin dağılımı değildir. Veri havuzlarından yaklaşık dörtte bir örnek seçilmesine rağmen ana binom aralıklarında sonlu anakütle düzeltmesi uygulanmadı. Çünkü hedef tüm yerel sonlu havuzun doğruluğunu tasarım temelli tahmin etmek olarak kurulmamış; bağımsız örnek varsayımıyla benchmark belirsizliği raporlanıyor. Tüm ham havuzun doğruluğu, henüz puanlanmamış çiftler hakkında sonuç uydurularak bulunamaz.''')
add('''## 4. Model ve puanlama protokolünün teknik anlamı

`models/eurollm-fp16`, `models/eurollm-q8` ve `models/eurollm-q4` klasörleri aynı mimarinin yerel koşullarını temsil ediyor. Yapılandırmada 24 katman, 2.048 gizli boyut, 16 attention head, 8 key/value head ve 4.096 bağlam sınırı var. Q8/Q4 için nicemleme grup boyutu 64 ve mod `affine`. Ağırlık dosyası başlıkları FP16’da F16, nicemlenmiş dosyalarda U32 paketli ağırlıklar ve F16 yardımcı tensörler gösteriyor. Dolayısıyla Q4’te her şeyin 4 bit olduğu veya tüm hesapların 4 bit yürütüldüğü söylenemez.

Yapılandırmada `torch_dtype=bfloat16` yazması, yerel ağırlıkların gerçekten BF16 olduğu anlamına gelmiyor: dosya başlığı FP16 ağırlıkların F16 olduğunu gösteriyor. Raporun koşul adı dosya içeriğine dayanıyor. Bu inceleme özgün dönüştürme işlemini yeniden yapmadı; eldeki dönüştürülmüş ağırlıkları çalıştırdı. Aynı kaynaktan türetildikleri proje kayıtlarıyla destekleniyor; her tensörün matematiksel türetim geçmişi için ayrı dönüşüm manifesti yararlı olur.

`score_main.py` her cümleyi özel token eklemeden tokenlaştırır. Tek BOS tokenı yalnızca başlangıç bağlamıdır; EOS eklenmez ve puanlanmaz. Modelin bir önceki konumdan tahmin ettiği tüm cümle tokenları hedef alınır. Çıktı logits değerleri float32’ye çevrilir, log-softmax uygulanır ve gerçekleşen tokenların log-olasılıkları seçilir. Bu değerlerin toplamı toplam puan; token sayısına bölünmesi ortalama puandır. Float32 log-softmax kullanılması modelin tamamını FP32 yapmaz.

Bir çiftte iyi cümlenin ortalama puanı eksi kötü cümlenin ortalama puanı `mean_logp_diff` olarak saklanır. Fark sıfırdan büyükse doğru, sıfırsa eşit, negatifse yanlış kabul edilir. Tolerans tam sıfırdır. Üç modelde de tam eşitlik gözlenmedi. Aynı metnin her modelde aynı tokenlara ayrılması gerektiği ayrıca denetlendi.

Bu yöntem cümle üretimi, çoktan seçmeli cevap üretme, insan kabul edilebilirlik puanı veya sonraki-token doğruluğu ölçmüyor. İki cümleden hangisine daha yüksek dil modeli puanı verildiğini ölçüyor. Kritik bölge bilgisi veri içinde bulunsa da ana puan tüm cümleyi kapsıyor; etkilerin yalnızca belirli bir ekin log-olasılığından geldiği, kaydedilmemiş token puanları incelenmeden ileri sürülemez.''')
add('''## 5. Bu oturumda yapılan doğrulamalar

Projenin iki mevcut birim testi çalıştırıldı ve geçti. Python dosyalarının sözdizimsel derleme kontrolü de geçti. Python/MLX projesinde bu kontrol bir native uygulama derlemesi değildir; model yürütmesinin doğrulaması ayrıca yapılmıştır.

Ana sonuçlardan 3.000 çift-model kaydı ve bunların 6.000 cümle kaydı kontrol edildi. Her modelin tokenizer’ı yerel dosyalardan yüklenerek her cümle yeniden tokenlaştırıldı. Token sayısı ve token ID hash’i kayıtlarla eşleşti. Her parçanın 50 çift içerdiği, veri/yöntem/model kimliği, yapılandırma hash’i, puanların sonluluğu, toplam puan farkı, ortalama puan farkı ve karar işareti doğrulandı. Metriklerdeki toplam token sayısı da cümle kayıtlarından yeniden hesaplandı.''')
add(f'''Toplam puanın token sayısına bölünmesi ile saklanmış float32 ortalama arasındaki en büyük mutlak fark {R['audit']['max_mean_rounding_residual']:.3g}; bu fark float32 yuvarlamasıyla uyumlu ve uygulanan 10⁻⁶ sınırının altında. Özgün istatistik betiği ayrı çıktı klasörüne çalıştırıldı; ürettiği `main_statistics.json` eski dosyayla yapılandırılmış içerik bakımından tamamen eşleşti.

Özgün kodun kendi Clopper–Pearson hesapları SciPy beta dağılımı ters fonksiyonuyla 20, 250 ve 1.000 örnek büyüklüklerinde, sınır ve iç noktalarda, iki alfa düzeyinde karşılaştırıldı. En büyük fark {R['audit']['max_cp_error_vs_scipy']:.3g}. Kesin McNemar değerleri, sıfır uyumsuz çift dahil seçilmiş sınır durumlarında SciPy `binomtest` ile doğrulandı. Bu doğrulama yeni bir teori kanıtı değil; uygulamaların beklenen sayısal sonuçları verdiğine ilişkin bağımsız hesap kontrolüdür.''')
add('''## 6. Genel doğruluk, hata oranı ve etki büyüklüğü

FP16 88 hata, Q8 85 hata, Q4 116 hata verdi. Q8’in doğruluk artışı +0,3 yp; Q4’ün doğruluk kaybı −2,8 yp. Yüzde puan ile göreli yüzde aynı değildir. Q4’ün FP16’ya göre göreli doğruluk azalması yaklaşık %3,07; hata sayısındaki göreli artış ise 28/88 = %31,82. Hata oranı başlangıçta küçük olduğundan bu ikinci oran daha büyük görünür. Her ikisi de aynı 28 net farkı farklı paydalarda anlatır.

Q8’in göreli hata azalması 3/88 = %3,41’dir. Bu tanımlayıcı azalma, Q8’in genel olarak FP16’dan daha yetenekli olduğunu göstermiyor. Aynı örneklemde daha az hataya sahip olması, nicemlemenin bir düzenlileştirme yararı yarattığını veya modelin sistematik olarak iyileştiğini kanıtlamaz.''')
add('''## 7. Neden eşleştirilmiş McNemar testi?

Her çift üç modelin tamamına verildiği için gözlemler model grupları arasında bağımsız değil. Bir çiftin her iki modelce doğru/yanlış bulunması ortak zorluğu yansıtır. İki bağımsız oran z-testi veya eşleşmeyi bozan bağımsız bootstrap bu yapıyı dikkate almaz.

McNemar testi yalnızca iki modelin ayrıştığı çiftlere bakar. `down`, FP16 doğru iken diğer modelin yanlış olduğu; `up`, FP16 yanlış iken diğer modelin doğru olduğu sayıdır. Eşit başarı olasılığı sıfır hipotezinde, bu uyumsuz çiftlerin iki yöne dağılması 0,5 olasılıklı binom deneyine karşılık gelir. Küçük uyumsuzluk sayısında asimptotik ki-kare yaklaşımı yerine iki yönlü kesin binom testi kullanıldı.

Q8’de yalnızca 3 uyumsuz çift var; üçünün de iyileşme yönünde olması p=0,25 verir. Bu kadar az ayrışmayla üstünlük iddiası güçlü biçimde sınanamaz. Q4’te 56 uyumsuz çiftin 42’si bozulma, 14’ü iyileşmedir. Net 28 kayıp buradan gelir; “Q4 28 çiftte hata yaptı” ifadesi eksiktir, çünkü toplam 116 hata ve 56 değişen karar vardır.''')
table(['Karşılaştırma','İkisi doğru','İkisi yanlış','Doğru→yanlış','Yanlış→doğru','Fark (yp)','Ham p','Holm p'],[[n.upper()+' − FP16',v['both_correct'],v['both_wrong'],v['down'],v['up'],f(v['delta_pp']),pv(v['p']),pv(v['p_holm'])] for n,v in R['primary'].items()])
add('''Holm düzeltmesi ana ailede iki karşılaştırmaya uygulanmıştır: Q8–FP16 ve Q4–FP16. Küçük p değeri ikiyle çarpılır; sıralama ve monotonluk korunarak diğer değer düzeltilir. İki karşılaştırmanın öncelikli sayılması mevcut analiz kodunun tanımıdır; burada zaman damgalı bir ön kayıt belgesi görülmediği için kesin bir preregistration iddiası yapılmıyor.

Ana test ailesinin dışında sonradan yapılan keşifsel analizler ayrı işaretlenmiştir. Her tabloya yerel Holm uygulamak, bütün araştırmadaki tüm olası analizleri kapsayan tek bir hata denetimi sağlamaz. Özellikle birçok duyarlılık senaryosundan en küçük p değerini seçmek uygun değildir.''')
fig('03_transitions','Şekil 3. FP16’ya göre iyileşen ve bozulan kararlar. Q4’te tek yönlü bir bozulma değil, iki yönlü hareket ve negatif net etki vardır.')
add('''## 8. Güven aralıkları ne söylüyor?

Model doğruluğunun tek başına aralığı Clopper–Pearson yöntemiyle hesaplandı. Model farkında ise eşleşme korunuyor: bütün örneklemdeki bozulma ve iyileşme olay oranlarının her biri için %97,5 kesin binom aralığı kuruluyor. Bonferroni birleşimi bu iki olay oranı için en az %95 ortak kapsama sağlıyor; üst-alt sınırlar uygun biçimde çıkarılarak fark aralığı elde ediliyor.

Bu yöntem muhafazakârdır; aralıklar gereğinden geniş olabilir. “En az %95” ifadesi bağımsız çiftler ve ilgili örnekleme modeli altında geçerlidir; şablon bağımlılığı varsa matematiksel kapsama güvencesi doğrudan korunmaz. Ayrıca bu aralıklar her model karşılaştırması içindir; iki model karşılaştırmasının tümü için ayrıca eşzamanlı %95 aralık diye sunulmuyor. P değerlerinde Holm kullanılması güven aralıklarını kendiliğinden Holm-düzeltilmiş yapmaz.''')
table(['Karşılaştırma','Fark (yp)','Muhafazakâr %95 eşleştirilmiş GA','Kategori tabakalı bootstrap %95 GA','Küme bootstrap %95 GA'],[[n.upper()+' − FP16',f(R['primary'][n]['delta_pp']),interval(R['primary'][n]['ci_pp']),interval(R['stratified_bootstrap'][n]['delta_ci_pp']),interval(R['cluster_sensitivity']['pair_weighted_cluster_bootstrap'][n]['delta_ci_pp'])] for n in ['q8','q4']])
add('''Q8 aralığının sıfırı içermesi “eşitlik kanıtlandı” demek değildir. Eşdeğerlik veya aşağı kalmama için kabul edilebilir doğruluk kaybı sınırı araştırma öncesinde belirlenmeli, uygun test ve güç tasarımı yapılmalıdır. Sonuca baktıktan sonra ±1 yp gibi bir sınır seçerek eşdeğerlik ilan edilmedi.

Bootstrap, gözlenmiş çiftleri yeniden seçer; Q8’de gözlenmiş bozulma hiç olmadığı için yeniden örneklemeler de bozulma üretemez. Bu nedenle Q8 bootstrap alt sınırının sıfır olması, gelecekte bozulma riskinin sıfır olduğuna ilişkin kanıt değildir. Seyrek ayrışma durumunda ana kesin/muhafazakâr aralık daha temkinli yorum sağlar. Başarı farkı testinin p değeri ile muhafazakâr aralığın sıfırı içerip içermemesi bazı alt gruplarda birebir örtüşmeyebilir; çünkü bunlar aynı testin matematiksel tersleri değildir.''')
fig('06_uncertainty','Şekil 6. Aynı ana fark için üç belirsizlik yaklaşımı. Bootstrap aralıkları keşifsel sağlamlık kontrolüdür; ana testi değiştirmez.')
add('''## 9. Üç modelin ortak hata yapısı ve Q8–Q4 karşılaştırması

Aşağıdaki üç haneli desen FP16, Q8, Q4 sırasındadır; 1 doğru, 0 yanlış demektir. Bu tablo toplam hata sayısının arkasındaki örtüşmeyi açıklar.''')
table(['Desen','Çift sayısı','Anlam'],[[k,v,{'111':'Üçü de doğru','000':'Üçü de yanlış','110':'FP16/Q8 doğru, Q4 yanlış','001':'Yalnız Q4 doğru','010':'Yalnız Q8 doğru','011':'Q8/Q4 doğru, FP16 yanlış'}[k]] for k,v in sorted(R['patterns'].items())])
add('''870 çiftte üç model de doğru, 72 çiftte üçü de yanlış. Bütün modellerin yanlış bulduğu çiftleri otomatik “kötü etiket” saymak yanlış olur; bunlar gerçek ortak model zayıflığı da olabilir. Etiket incelemesi model oyundan bağımsız gerekçeler gerektirir.

Q8, bu örneklemde FP16’nın doğru bulduğu hiçbir çifti kaybetmiyor. Q4’ün iyileştirdiği 14 çiftin 13’ünde Q8 hâlâ yanlış; bu yüzden nicemleme yalnızca FP16 hatalarını büyüten bir mekanizma gibi düşünülemez. İki nicemleme düzeyi farklı yerel karar hareketleri doğuruyor.''')
v=R['all_pairwise_exploratory']['q8_to_q4']
add(f'''Keşifsel Q4−Q8 karşılaştırmasında {v['down']} bozulma, {v['up']} iyileşme; net fark {f(v['delta_pp'])} yp; eşleştirilmiş aralık {interval(v['ci_pp'])} yp. Üç ikili karşılaştırmayı birlikte içeren ayrı Holm ailesinde p={pv(v['p_holm'])}. Bu karşılaştırma ana iki testlik aileye gizlice eklenmedi; ayrı keşifsel sonuçtur.''')
add('''## 10. Kategori sonuçları ve etkilerin dağılımı

Her kategoride 250 çift var; bir karar 0,4 yp eder. Kategorilerin farklı başlangıç zorlukları ve tavan etkileri vardır. Örneğin düzensiz biçimlerde FP16 zaten 248/250 doğrudur; iyileşme için yalnızca iki hata vardır. Adlaştırma ve özne–yüklem kategorilerinde daha fazla hata ve karar hareketi vardır.''')
table(['Kategori','FP16 (%)','Q8 (%)','Q4 (%)','Q8 fark (yp)','Q4 fark (yp)','Q4 ↓/↑'],[[CN[c]]+[f(v['accuracy'][n],1) for n in N]+[f(v['comparisons'][n]['delta_pp'],1) for n in ['q8','q4']]+[str(v['comparisons']['q4']['down'])+'/'+str(v['comparisons']['q4']['up'])] for c,v in R['category'].items()])
fig('02_categories','Şekil 2. Eşit ağırlıklı dört kategori. Kategori isimleri veri dosyalarının sınıflandırmasını temsil eder.')
add('''Q4’ün 28 net kaybının 18’i özne–yüklem, 4’ü argüman yapısı, 7’si adlaştırma kategorisinden geliyor; düzensiz biçimler bir doğru kazanımla kaybı azaltıyor. Özne–yüklem katkısı net kaybın yaklaşık %64,3’üdür. Bu bir katkı ayrıştırmasıdır; kategori etkilerinin birbirinden istatistiksel olarak farklı olduğuna ilişkin etkileşim testi değildir.

Keşifsel olarak dört kategori × iki FP16 karşılaştırması, sekiz testlik ayrı Holm ailesiyle incelendi. Sonuçlar aşağıdadır. Aynı tabloda kategori fark aralıkları verilmiştir; bu aralıklar sekiz kategori testinin tamamı için eşzamanlı aralık değildir.''')
table(['Kategori','Model','Fark için GA (yp)','Ham p','8 testlik Holm p'],[[CN[c],n.upper(),interval(v['comparisons'][n]['ci_pp']),pv(v['comparisons'][n]['p']),pv(v['comparisons'][n]['p_holm'])] for c,v in R['category'].items() for n in ['q8','q4']])
add('''Özne–yüklem Q4 karşılaştırması bu keşifsel düzeltmede de eşiğin altında kalıyor. Yine de kategori düzeyinde nicemlemenin belirli bir dilbilgisel yetiyi nedensel olarak bozduğu sonucu tek başına buradan çıkarılamaz. Aynı şablonların tekrarları, etiketler, tokenizasyon ve yalnızca bir model ailesi incelenmiş olması açıklamanın kapsamını sınırlar. Bir kategoride p<0,05, diğerinde p>0,05 bulunması doğrudan “bu kategorilerdeki etkiler anlamlı biçimde farklıdır” anlamına gelmez.''')
add('''## 11. Puan farkları: karar değişiminin büyüklüğü ve yeri

Bir modelin tercihinin işareti kadar iyi ve kötü cümleyi ne kadar ayırdığı da incelendi. Puan farkının birimi token başına doğal log-olasılıktır. Bu değer bir sınıflandırıcı güven olasılığı veya kalibre edilmiş doğruluk tahmini değildir. “Sınırda” ifadesi burada yalnızca puan farkının sıfıra yakın olması anlamına gelir.''')
table(['Model','Fark ortalaması','Fark medyanı','FP16’ya göre değişim ort.','Değişimin MAE değeri','Değişimin RMSE değeri','Pearson r','Spearman ρ'],[[n.upper(),f(v['margin']['mean'],6),f(v['margin']['median'],6),f(v['change']['mean'],6),f(v['mae'],6),f(v['rmse'],6),f(v['pearson'],6),f(v['spearman'],6)] for n,v in R['margins'].items()])
add('''Q8’de ortalama değişim neredeyse sıfır; tek tek çiftlerdeki küçük pozitif ve negatif hareketler birbirini dengeliyor. Q4’te mutlak değişim çok daha büyük. Ortalama kaymanın küçük olması puanların tek tek korunduğu anlamına gelmez; bu nedenle MAE ve RMSE de verildi. Pearson doğrusal ilişkiyi, Spearman sıralama korunmasını ölçüyor. Yüksek korelasyon, sıfıra yakın çiftlerde kararın değişmeyeceğini garanti etmez.''')
table(['FP16 |fark| çeyreği','Çift','Alt sınır','Üst sınır','Q8 değişim','Q4 değişim','Q4 değişim oranı'],[[i+1,250,f(S['margin_diagnostics']['q4']['flip_rate_by_fp16_abs_margin_quartile'][i]['fp16_abs_margin_min'],6),f(S['margin_diagnostics']['q4']['flip_rate_by_fp16_abs_margin_quartile'][i]['fp16_abs_margin_max'],6),R['margin_quartiles']['q8'][i]['flips'],R['margin_quartiles']['q4'][i]['flips'],'%'+f(R['margin_quartiles']['q4'][i]['flips']/2.5,1)] for i in range(4)])
add('''Q4’ün 56 karar değişiminin 46’sı en küçük mutlak FP16 farkına sahip çeyrekte: %82,14. İlk çeyrekte değişim oranı %18,4, kalan 750 çiftte %1,33; oran yaklaşık 13,8 kattır. Bu keşifsel ilişki nicemleme hatasının rastgele dağılmadığını düşündürür. Ancak karar değişimi zaten puanın sıfır sınırını geçmesini gerektirdiği için bu ilişkinin kısmen matematiksel/geometrik olması beklenir; bağımsız bir nedensel mekanizma kanıtı gibi sunulmamalıdır.''')
fig('04_margins','Şekil 4. Solda FP16/Q4 farkları; sağda FP16 mutlak fark çeyreklerine göre Q4 karar değişimleri.')
add('''## 12. Ortalama mı toplam log-olasılık mı?

Ana yöntem pilotta kullanılmış olan ortalama log-olasılıktır. Alternatif analiz saklanan toplam log-olasılıklardan yeniden karar üretir; modelin tekrar yüklenmesine gerek yoktur. Toplam puan uzun cümlede daha fazla negatif terim toplar. Ortalama puan bu uzunluk etkisini azaltmayı amaçlar, fakat eklerin farklı tokenlara bölünmesi nedeniyle sıralamayı yine değiştirebilir. İki yöntem farklı ölçüm tercihleridir; daha yüksek doğruluk veren yöntemi sonradan ana yöntem yapmak uygun değildir.''')
table(['Model','Ortalama doğruluk (%)','Toplam doğruluk (%)','Yöntemler arası farklı karar'],[[n.upper(),f(R['accuracy'][n]['accuracy_percent'],1),f(R['total_scoring']['accuracy'][n],1),R['total_scoring']['disagreements'][n]] for n in N])
table(['Toplam logp karşılaştırması','Fark (yp)','↓/↑','GA (yp)','Holm p'],[[n.upper()+' − FP16',f(v['delta_pp']),f"{v['down']}/{v['up']}",interval(v['ci_pp']),pv(v['p_holm'])] for n,v in R['total_scoring']['comparisons'].items()])
add('''Alternatif puanlamada Q8 artışı yine küçük ve anlamlı değil; Q4 kaybı −3,9 yp ile devam ediyor. Böylece temel yön yalnızca ortalama seçimine bağlı görünmüyor. Fakat doğruluk düzeyleri ve fark büyüklükleri yönteme duyarlı; makale yalnızca tek tablo verip bu duyarlılığı gizlememeli.

Her iki yöntemde de aynı tokenizer ve cümleler kullanıldığı için farklar tokenlaştırma koşulunun değiştirilmesinden değil, mevcut puanların toplama/normalleştirme biçiminden kaynaklanıyor. Eşit token sayılı çiftlerde toplamı aynı pozitif sayıya bölmek sıralamayı değiştirmez; kontrol sonucunda bu 626 çiftte hiçbir modelde yöntemler arası karar ayrılığı yok.''')
add('''## 13. Token uzunluğunun rolü

İyi cümleler ortalama 13,415, kötü cümleler 13,464 token. İyi taraf 6–28, kötü taraf 6–29 token aralığında. Dolayısıyla 4.096 bağlam sınırına yaklaşılmıyor; kesme/truncation gerekmiyor. Üç modelin tamamı 26.879 cümle tokenı puanlıyor. BOS bağlam tokenı bu puanlanan token toplamına dahil değil.''')
table(['Uzunluk grubu','n','FP16 (%)','Q8 (%)','Q4 (%)','Q4−FP16 (yp)','FP16 yöntem değişimi','Q4 yöntem değişimi'],[[{'equal':'Eşit token sayısı','good_longer':'İyi cümle daha uzun','good_shorter':'İyi cümle daha kısa'}[k],v['n']]+[f(v['accuracy'][n],2) for n in N]+[f(v['comparisons']['q4']['delta_pp']),v['mean_total_disagreements']['fp16'],v['mean_total_disagreements']['q4']] for k,v in R['length']['groups'].items()])
add('''İyi cümle daha kısa olduğunda ortalama puan doğruluğunun belirgin düşük olması dikkat çekici. Ancak bu gruplar kategorilerden ve dilsel dönüşüm türlerinden bağımsız oluşturulmuş deney koşulları değil. Bu tablo “kısalık hataya neden olur” sonucunu desteklemez. Daha ayrıntılı nedensel yorum için kategori/şablon/ek türünü kontrol eden önceden tasarlanmış bir çalışma gerekir. Eşit uzunluk grubunda Q4 farkının yaklaşık −2,88 yp olması ise tüm kaybın yalnızca farklı token sayılarından kaynaklandığı açıklamasını zayıflatıyor.''')
add('''## 14. Sıfıra yakın fark ve karar eşiği duyarlılığı

Ana eşik değiştirilmedi. Aşağıdaki tablo, pozitif farkın alternatif eşikten büyük olmasını istemenin etkisini gösterir. Burada eşik altındaki pozitifler doğru sayılmıyor; bu bir “çekimser kalma” sisteminin başarısı değil. `|fark|≤eşik` sütunları karar sınırının çevresindeki çiftleri sayar.''')
table(['Eşik','FP16 doğru','Q8 doğru','Q4 doğru','FP16 yakın','Q8 yakın','Q4 yakın'],[[k]+[v['correct'][n] for n in N]+[v['near_zero'][n] for n in N] for k,v in R['tolerance_sensitivity'].items()])
add('''10⁻⁶ ve 10⁻⁴ düzeyinde sonuçlar değişmiyor. 10⁻³ düzeyinde Q8’de iki pozitif karar kayboluyor. Bu, Q8’in küçük kazancının birkaç sınır örneğine dayandığını gösterir. 10⁻² artık float32 yuvarlama toleransı gibi yorumlanamayacak bir değerlendirme tercihidir; o sonucu ana puanlama yerine koymadık.''')
add('''## 15. Benzer cümleler ve bağımsızlık varsayımı

Birebir tekrarlar temizlenmiş olsa da “Ancak o bu projeye çok emek verdi” ile “Şimdi o bu projeye çok emek verdi” gibi ortak çekirdeği olan çiftler var. Böyle örneklerin hataları ilişkili olabilir. Aynı dilsel örüntüyü tekrar tekrar ölçmek, 1.000 tamamen bağımsız dilsel durum ölçmekle aynı değildir.

Keşifsel bir küme tanımı uygulandı: çiftin iyi ve kötü tarafının başında aynı sözcük varsa ve bu sözcük açıkça listelenmiş bağlaç/zarf kümesindeyse kaldırıldı; sonrasında iki cümlenin normalize edilmiş tam içeriği aynı olan çiftler, kategori içinde aynı kümeye alındı. Örneğin ve/ama/ancak/çünkü ile başlayan yakın varyantlar böyle birleşebiliyor. İşlem sadece başta ve iki tarafta ortak sözcüklere uygulanıyor; gövdedeki sözcükler veya hedef ekler silinmiyor.

Bu kural sonuçlar görüldükten sonra tasarlanan bir sağlamlık kontrolüdür. Gerçek üretim şablonu kimliklerini geri kazanmaz; anlamlı zaman/odak değişikliklerini de bir araya getirebilir. Sözlük değişimli diğer şablon akrabalıklarını kaçırır. Dolayısıyla burada bulunan küme sayısı “gerçek bağımsız örnek sayısı” değildir.''')
table(['Kategori','Çift','Yaklaşık küme','Tek üyeli küme','En büyük küme'],[[CN[c],250,v['clusters'],v['singleton_clusters'],v['max_size']] for c,v in R['cluster_sensitivity']['per_category'].items()])
add(f'''Toplam {R['cluster_sensitivity']['n_clusters']} yaklaşık küme oluştu. Küme listesi `template_clusters.csv` içinde her üyenin kimliğiyle veriliyor. Kategori başına kümeler yerine koyarak seçildi; seçilen kümedeki bütün çiftler birlikte taşındı. Her bootstrap örneklemesinde kategori doğruluğu seçilmiş toplam doğru / seçilmiş toplam çift olarak hesaplandı; dört kategori yine eşit ağırlıkla birleştirildi. 20.000 tekrar, tohum {R['seed']+1}. Bu yöntem çift ağırlıklı hedefe ilişkin küme duyarlılığını inceler.

Ek olarak her kümeyi kategori içinde eşit ağırlıklı sayınca doğruluklar FP16 %{f(R['cluster_sensitivity']['equal_cluster_accuracy_percent'][0],3)}, Q8 %{f(R['cluster_sensitivity']['equal_cluster_accuracy_percent'][1],3)}, Q4 %{f(R['cluster_sensitivity']['equal_cluster_accuracy_percent'][2],3)} oluyor. Q8 farkı {f(R['cluster_sensitivity']['equal_cluster_delta_pp'][1])} yp, Q4 farkı {f(R['cluster_sensitivity']['equal_cluster_delta_pp'][2])} yp. Bu yeniden ağırlıklandırma farklı bir hedef büyüklüğü tanımlar; özgün benchmark puanının “düzeltilmiş gerçek değeri” değildir.

Küme bootstrap aralığında Q4 farkının sıfırın altında kalması bu özel kümelendirmeye karşı yönün korunduğunu gösterir. Daha geniş şablon bağımlılıklarını çözmüş sayılmaz. Kaynak üretim şablonu kimlikleri varsa gerçek kümelerle yeniden analiz, yoksa bağımsız yeni cümle örneklemi en güçlü sonraki kontrol olur.''')
add('''## 16. Pilot örtüşmesi ve kategori çıkarma duyarlılığı

Pilot 20 çift, ana 1.000’in içinden seçildi. Bu nedenle pilot doğrulaması bağımsız dış test seti değildir. Puanlama yönteminin pilot incelemesine dayanması, pilotta görülen örneklere kısmi uyum riski doğurabilir; bunun boyutunu abartmadan görmek için bu 20 çift çıkarıldı. Ayrıca her kategori sırayla dışarıda bırakılarak genel kaybın tek bir kategoriye ne ölçüde dayandığı incelendi.''')
keys=['exclude_pilot20']+['leave_out_'+c for c in C]
table(['Senaryo','n','Q8 fark (yp)','Q8 Holm p','Q4 fark (yp)','Q4 Holm p'],[[('Pilot 20 çıkarıldı' if k=='exclude_pilot20' else CN[k.replace('leave_out_','')]+' çıkarıldı'),R['sensitivity'][k]['n'],f(R['sensitivity'][k]['comparisons']['q8']['delta_pp']),pv(R['sensitivity'][k]['comparisons']['q8']['p_holm']),f(R['sensitivity'][k]['comparisons']['q4']['delta_pp']),pv(R['sensitivity'][k]['comparisons']['q4']['p_holm'])] for k in keys])
add('''Pilot çıkarıldığında Q4 sonucu ana sonuca yakın kalıyor. Özne–yüklem çıkarıldığında Q4 farkı −1,33 yp ve iki karşılaştırmalı Holm p≈0,174; bu alt örneklemde anlamlılık eşiği aşılmıyor. Bu, “Q4’ün diğer kategorilerde etkisi yok” anlamına gelmez: hem etki büyüklüğü hem örnek sayısı ve uyumsuzluk sayısı değişir. Yine de genel bulgunun özne–yüklem kategorisinden güçlü destek aldığını gösterir. Düzensiz biçimler çıkarıldığında kayıp büyüyor; bu kategorinin yüksek doğruluğu genel puanı yukarı taşımıştı.

Her senaryonun Holm düzeltmesi yalnızca o senaryodaki iki karşılaştırmaya uygulanmıştır. Senaryolar sonradan seçilmiş keşifsel kontrollerdir; bağımsız doğrulama deneyleri değillerdir.''')
add('''## 17. Dilbilimsel inceleme nasıl yapıldı?

Özgün `manual_review_20.csv` dosyası 12 Q4 bozulması, 5 Q4 iyileşmesi ve 3 Q8 iyileşmesinden oluşacak biçimde sabit tohumla seçilmiş. Seçim çift tekrarlarını önlüyor. Bu 20 örnek tüm benchmark’tan rastgele seçilmiş temsilî bir örneklem değil; özellikle karar değiştiren çiftleri inceliyor. Dolayısıyla buradaki şüpheli etiket oranından bütün veri setinin etiket hata oranı hesaplanamaz.

Bu raporda inceleme, en az bir nicemlenmiş modelin FP16’ya göre karar değiştirdiği 58 benzersiz çiftin tamamına genişletildi. Q4’te 56, Q8’de 3 değişim var; bir çift ortak olduğundan birleşim 58. Her çiftte iyi/kötü cümle, hedef ipucu, yüklem ve model yönü birlikte okunarak kısa gerekçe yazıldı. Bu işlem model destekli ön dilbilimsel değerlendirmedir; kör, bağımsız iki insan uzmanın anotasyonu değildir. İnsanlar arası uyum veya Cohen κ hesaplanamaz.

Kullanılan durumlar: etiket ayrımı destekleniyor; bağlama duyarlı; iki cümle de kabul edilebilir görünüyor; uzman incelemesi gerekli. “İki cümle” durumu kesin hatalı etiket ilanı değildir. Anlam farkı ile dilbilgisizlik ayrımı özellikle korunmuştur. Standart Türkçedeki kişi uyumu daha açık görünürken adlaştırma tamamlayıcıları olay/olgu ve söylem bağlamına daha duyarlıdır.''')
table(['Ön değerlendirme','20 seçilmiş çift','58 değişen çift'],[[k,R['review']['selected20_status_counts'].get(k,0),R['review']['status_counts'].get(k,0)] for k in R['review']['status_counts']])
add('''Önemli örnekler: “yatırım kararlarını hızla almasına … şaşırdı” ve “… aldığına … şaşırdı” olay/olgu odaklarıyla okunabilir. “Toplantının ertelenmesine … tepki gösterdi” ile “… ertelendiğine … tepki gösterdi” de benzer biçimde açık bir dilbilgisizlik karşıtlığı kurmayabilir. “Doğruyu söylediğine … inanıyor” ile “doğruyu söylemesine … inanıyor” için farklı anlamsal okuma ihtimali vardır. Buna karşılık “uygulanmasını destekledi” / “uygulandığını destekledi” çiftinde ikinci biçimin kabulü daha özel bir iddia/kanıt bağlamı gerektirebilir; bu nedenle kesin iki-geçerli etiketi verilmedi.

Kullanıcının taslağındaki 0962 örneği kısaltılmıştı; gerçek kayıtta cümle “Ve şirketin…” diye başlıyor. Aşağıdaki alıntılar mevcut CSV’deki cümleleri aynen korur. Bu küçük ayrıntı, şablon ilişkisini ve tekrar üretilebilirliği izlemek açısından önemlidir.''')
add('''## 18. Seçilmiş 20 çiftin tek tek gerekçeli incelemesi

Aşağıdaki puan farkları FP16 / Q8 / Q4 sırasındadır. Pozitif değer veri setinin iyi etiketli cümlesini destekler. Bu bölüm ana sonuçları düzeltmez; tartışmalı değerlendirmeyi görünür kılar.''')
rev20=list(csv.DictReader((OUT/'manual_review_20_annotated.csv').open(encoding='utf-8-sig')))
for i,r in enumerate(rev20,1):
 add(f'''### 18.{i}. `{r['pair_id']}` — {r['review_stratum']}

**İyi etiketli:** {r['sentence_good']}

**Kötü etiketli:** {r['sentence_bad']}

**Puan farkları:** {f(float(r['fp16_margin']),6)} / {f(float(r['q8_margin']),6)} / {f(float(r['q4_margin']),6)}.

**Ön değerlendirme:** `{r['linguistic_validity']}`. {r['manual_review_note']}''')
add('''## 19. Etiket duyarlılığı: sonuç tartışmalı örneklerden mi doğuyor?

Orijinal etiketler ve ana benchmark sayımları değiştirilmedi. Üç farklı dışlama senaryosu kullanıldı. Birincisi yalnızca incelenmiş 58 çift içindeki iki-geçerli olarak işaretlenen örnekleri çıkarıyor; bu sonuç değişimine göre seçilmiş bir alt küme olduğundan seçilim yanlılığı taşıyor. İkincisi aynı yüklem temelli taramayı 1.000 çiftin tamamına uyguluyor: adlaştırma kategorisinde `şaşırdı`, `tepki gösterdi`, `inanıyor` kritik bölgesi olan bütün çiftler dışarıda bırakılıyor. Bu tarama modelin değişip değişmediğine bakmıyor; yine de tüm etiket sorunlarının eksiksiz taraması değil. Üçüncüsü adlaştırma kategorisini tamamen çıkarıyor ve bu nedenle araştırmanın kapsamını değiştiriyor.''')
keys=['exclude_reviewed_ambiguous','exclude_uniform_predicate_screen','exclude_all_nominalization'];labels=['İncelenen 8 tartışmalı çift çıkarıldı','Tüm veride aynı yüklemlerle taranan 17 çift çıkarıldı','Adlaştırmanın 250 çifti çıkarıldı']
table(['Senaryo','n','FP16 (%)','Q8 (%)','Q4 (%)','Q4 fark (yp)','Q4 GA (yp)','Q4 Holm p'],[[label,R['sensitivity'][k]['n']]+[f(R['sensitivity'][k]['accuracy'][n],3) for n in N]+[f(R['sensitivity'][k]['comparisons']['q4']['delta_pp']),interval(R['sensitivity'][k]['comparisons']['q4']['ci_pp']),pv(R['sensitivity'][k]['comparisons']['q4']['p_holm'])] for k,label in zip(keys,labels)])
add('''Q4’ün negatif yönü bu üç senaryoda korunuyor. Bu, burada saptanan adlaştırma şüphelerinin bütün sonucu tek başına açıklamadığını gösterir. Veri setinin kalanının hatasız olduğunu veya tüm olası etiket düzeltmelerinde p değerinin korunacağını göstermez.

Ek bir sınır hesabında, aynı 17 taranmış çiftin etiketlerinin herhangi bir alt kümede tamamen ters çevrilebileceği varsayıldı; n=1.000 sabit tutuldu. İki model aynı karar verdiyse ters çevirme aralarındaki farkı değiştirmez; ayrıştılarsa katkının işareti tersine döner. Q4−FP16 farkı bu koşullu en kötü/en iyi senaryolarda −3,2 ile −1,8 yp arasında; Q8−FP16 +0,1 ile +0,3 yp arasında kalıyor. Bunlar güven aralığı veya p değeri değil, yalnızca belirtilen 17 çift için deterministik etiket sınırlarıdır. Her iki cümlenin kabul edilebilir olması gerçekte etiketi ters çevirmekten farklıdır; ters çevirme yalnızca matematiksel stres testidir.''')
add('''## 20. Kaynak tüketimi: hangi ölçüm neyi kapsıyor?

Özgün `score_main.py`, model yükleme ve ısınma sonrasında her 50 çiftlik parçayı zamanlar. Puanlama döngüsü tokenlaştırma, model hesabı, log-olasılık çıkarımı, kayıt nesnelerinin hazırlanması ve çiftler arası MLX cache temizliği maliyetlerini içerir. Dosya yazımı, model yükleme, ısınma ve parçalar arası beklemeler bu süreye dahil değildir. Bu yüzden değerler saf GPU kernel süresi veya kullanıcı açısından baştan sona toplam çalışma süresi değildir.

MLX allocator tepe belleği, süreç RSS tepe belleği ve diskteki model dosyası boyutu ayrı ölçümlerdir. Apple birleşik bellek sisteminde bunları birbirine eklemek veya birini toplam sistem RAM tüketimi diye sunmak yanlış olur. RSS süreç ömrü boyunca bir yüksek su seviyesi olduğundan parça ölçümü değildir; yeniden başlayan süreçler aynı geçmişi paylaşmaz. Metriklerdeki yükleme/ısınma süreleri de aynı süreçte birden fazla parçaya tekrar yazılır; 20 parçanın hepsindeki yükleme değerlerini toplamak yanlış sonuç verir.''')
table(['Ölçüt']+[n.upper() for n in N],[[label]+[f(R['resources'][n][k],d) for n in N] for k,label,d in [('seconds','Puanlama toplamı (sn)',3),('pairs_per_second','Çift/sn',3),('tokens_per_second','Puanlanan token/sn',3),('mlx_peak_gb','MLX tepe (GB)',6),('rss_peak_gb','Süreç RSS tepe (GB)',6),('weights_gb','Safetensors dosya toplamı (GB)',6)]])
table(['FP16’ya göre','Q8','Q4'],[[label]+[f(R['resources'][n][k],3) for n in ['q8','q4']] for k,label in [('time_saving_percent','Süre azalması (%)'),('speedup','Hız çarpanı'),('mlx_saving_percent','MLX tepe azalması (%)'),('weight_saving_percent','Ağırlık dosyası küçülmesi (%)')]])
add('''Süre azalması ile hız artışı farklı oranlardır: Q4’ün %22,62 daha kısa sürmesi 1,292× hız çarpanına, yani yaklaşık %29,24 daha yüksek çift/sn değerine karşılık gelir. Q8 ile Q4 arasındaki özgün süre farkı yalnızca yaklaşık 2,68 saniye, Q8’e göre %2,74’tür; bu farkı tek yürütmeden evrensel hız üstünlüğü olarak sunmak doğru değildir.

RSS FP16’da Q8’den küçük gözlenmişken MLX allocator FP16’da daha büyüktür. Bu çelişki gibi görünse de farklı bellek kapsamları ve süreç koşulları nedeniyle mümkündür. “Q4 toplam cihaz belleğini %63,8 azaltır” yerine “bu deneyde MLX allocator tepe ölçümü %63,8 daha düşüktür” denmelidir. Swap, bellek baskısı, güç ve enerji tüketimi için bu kayıtlarda eşzamanlı ölçüm yoktur.''')
fig('05_resources','Şekil 5. Özgün deneyde süre, MLX tepe ve ağırlık boyutu. Bunlar farklı fiziksel büyüklüklerdir; ortak bir bellek metriği gibi birleştirilmemelidir.')
add('''### 20.1. Parçalar arası süre değişkenliği

Her model için 20 parça bulunması, 20 bağımsız tam deney tekrarı bulunduğu anlamına gelmez. Parçalar farklı cümleleri içerir ve kategori sırasına göre bloklanmıştır. Isı, işletim sistemi yükü ve önbellek durumu da zaman içinde değişebilir. Bu nedenle aşağıdaki standart sapmalar çalışma içi tanımlayıcı değişkenliktir; modellerin hız farkına ilişkin güven aralığı değildir.''')
table(['Model','Parça ort. (sn)','Parça SS (sn)','En kısa','Medyan','En uzun'],[[n.upper()]+[f(R['resources'][n]['chunk_seconds'][k],4) for k in ['mean','sd','min','median','max']] for n in N])
add('''## 21. Bu oturumdaki tam model tekrarı

Model tekrarı mevcut `score_main.py` puanlama fonksiyonunu kullanır. Yeni sarmalayıcı sadece çıktı klasörünü `results/deep_analysis/rerun/` olarak ayarlar; tamamlanmış özgün sonuçlar atlanmış gibi görünmesin diye ayrı klasör kullanılmıştır. FP16, Q8 ve Q4 ayrı süreçlerde, sırayla, her biri 20 parça ve toplam 1.000 çift olacak biçimde çalıştırıldı. Parça arası bekleme sıfırlandı; bu bekleme zaten puanlama süresine dahil değildi. Bu yürütme, yeni eğitim veya yeni nicemleme değildir.

Tekrar çalışması sırasında analiz ve diğer masaüstü işleri de yürüdüğünden bu, kontrollü performans benchmark’ı değildir. Doğruluk ve kayıt tekrarlanabilirliğini denetlemek için kullanılır. Süreyi yeni resmi ana sonuç olarak eski tabloya karıştırmıyoruz.''')
repeat=OUT/'rerun_comparison.json'
if repeat.exists():
 rr=json.loads(repeat.read_text())
 table(['Model','Yeniden doğru','Kararı değişen çift','Tam eşleşen kayıt','En büyük ort. fark değişimi','Özgün süre (sn)','Tekrar süre (sn)','Tekrar MLX tepe (GB)'],[[n.upper(),v['correct'],v['decision_changes'],v['identical_records'],f(v['max_abs_margin_difference'],9),f(R['resources'][n]['seconds'],3),f(v['score_seconds'],3),f(v['mlx_peak_gb'],6)] for n,v in rr['models'].items()])
 add('Tekrar kontrolünün ayrıntıları '+link('results/deep_analysis/rerun_comparison.json')+' dosyasında bulunuyor. Aynı modelin aynı 1.000 çiftte yeniden çalıştırılması örneklem büyüklüğünü 2.000’e çıkarmaz; aynı deterministik kararların tekrar ölçümüdür. Ayrıca kaynak süresindeki değişim, tek ölçümün makine koşullarına bağımlı olduğunu gösterir.')
else:
 add('**Tekrar ölçümü henüz rapora bağlanmadı; bu bölüm nihai doğrulama dosyası üretildiğinde doldurulmalıdır.**')
add('''## 22. Kodda hata var mı; değişiklik gerekiyor mu?

Mevcut kayıtlarda doğruluk, eşleştirme veya ana istatistik sonucunu değiştirmeyi gerektiren doğrulanmış bir hesap hatası bulunmadı. Bu, kodun her olası bozuk girdiye dayanıklı olduğunun kanıtı değildir. Kullanıcının kod sorunlarını birlikte değerlendirme isteği nedeniyle mevcut deney betikleri değiştirilmedi. Aşağıdaki noktalar gözlenen sonuçları düzeltmekten çok araştırma güvenilirliğini ve veri korumasını güçlendirme önerileridir.''')
table(['Öncelik','Dosya','Bulgu ve önerilen yaklaşım'],[
 ['Yüksek: anotasyon kaybı',link('scripts/analyze_main.py'),'main() her çalışmada manual_review_20.csv üretirken inceleme alanlarını boş yazar. İnsan notları bu dosyaya eklenirse sonraki çalışmada silinebilir. Ayrı anotasyon dosyası veya pair_id ile mevcut notları koruyan birleştirme gerekir. Bu oturumda yeni çıktılar ayrı tutuldu.'],
 ['Orta: kayıt doğrulama',link('scripts/analyze_main.py'),'load_data ortalama fark ve token eşleşmesini kontrol ediyor; toplam puan özdeşliği, toplam/token=ortalama, yerel tokenizer yeniden hesabı ve model config hash’i gibi ek kontroller bu raporda ayrı yapıldı. Mevcut kayıtlar geçti; ana koda eklenmesi gelecekteki bozulmaları erken yakalar.'],
 ['Orta: devam etme',link('scripts/score_main.py'),'check_saved_chunk aynı veri/model adı/yönteme odaklı. Aynı klasörde ağırlıklar değiştirilirse eski ve yeni parçaların karışmasını engellemek için model ağırlık hash’i ve ortam sürümü doğrulaması güçlendirilebilir. Bu oturumda böyle bir karışma saptanmadı.'],
 ['Orta: ölçüm kaybı',link('scripts/score_main.py'),'Puan dosyası yazıldıktan sonra metrik yazımı başarısız olursa eksik metrik için uyarı veriliyor; geçmiş süreyi puanlardan geri üretmek mümkün değil. Yeni koşuda ayrı ölçüm gerekir. Mevcut 60 özgün metrik dosyası tam.'],
 ['Orta: eksik belleğin sıfır sayılması',link('scripts/analyze_main.py'),'Kaynak özetinde None değerini 0 ile değiştiren ifade var. Gelecekte ölçüm yoksa “ölçülmedi” yazılmalı. Bu deneyde bellek değerleri mevcut; rapordaki sayılar bu eksiklikten etkilenmedi.'],
 ['Araştırma tasarımı',link('scripts/prepare_dataset.py'),'Tam tekrar temizliği şablon bağımlılığını çözmez. Kaynak şablon ID’lerinin korunması ve şablon düzeyinde örnekleme değerlendirilebilir; mevcut ana veri seti sonradan değiştirilmemeli.'],
 ['Kaynak izlenebilirliği',link('models/eurollm-fp16/config.json'),'Yapılandırmadaki dtype alanı yerel ağırlık başlığıyla aynı semantiği taşımıyor. Dönüşüm komutu, sürüm, kaynak/ağırlık hash’leri ve gerçek tensör türleri ayrı manifestte belgelenebilir.']])
add('''Bunların hiçbiri mevcut %91,2 / %91,5 / %88,4 sayımlarını otomatik olarak geçersiz kılmıyor. Ana öncelik puanlama kodunu değiştirmek değil, anotasyonları korumak ve yöntemin yorum sınırlarını doğru belgelemek. Sözcük bazlı özel kurallar ekleyip mevcut örneklerde başarıyı artırmak bu araştırmanın nicemleme karşılaştırmasını bozar; yapılmadı.''')
add('''## 23. Yapılmayan analizler ve nedenleri

**Enerji, güç ve swap:** Eşzamanlı telemetri yok; model ağırlığından enerji tasarrufu hesaplanamaz.

**Token bazlı nedensel hata açıklaması:** Kaydedilmiş sonuçlar cümle toplamları ve ortalamalarıdır. Hangi ekin hangi attention katmanında bozulduğunu bunlardan çıkarmak mümkün değil. Böyle bir soru için ek token/katman kayıtları gerekir.

**İnsanlar arası dilbilimsel uyum:** İki bağımsız insan anotatör yok; κ veya insan doğrulama oranı uydurulmadı. 58 değişen çiftin ön incelemesi bütün 1.000 çift için kör dilbilimsel denetim değildir.

**Kontrollü hız anlamlılık testi:** Bağımsız, rastgele sıralı çok sayıda tam çalışma yok. Parçaları sahte tekrar sayarak çok küçük p değerleri üretilmedi. İki yürütme de aynı donanımın bütün kullanım durumlarını temsil etmez.

**Kapsayıcı bütün hipotez taraması:** Mümkün her alt gruba test uygulamak bilimsel kapsamı artırmak yerine yanlış pozitif ve sonradan hikâye kurma riskini artırabilir. Bu yüzden ana sonuçla ilişkili ölçüm, örnekleme, etiket ve kaynak duyarlılıkları seçildi; keşifsel statüleri korundu.

**Sonradan güç analiziyle kanıt üretme:** Gözlenen p değerini başka biçimde yeniden söyleyen “observed power” raporlanmadı. Gelecekteki deney için önceden tanımlı kabul edilebilir kayıp, beklenen uyumsuzluk ve şablon bağımlılığı kullanılarak örneklem tasarlanmalı.

**Bütün Türkçe veya başka modeller hakkında genelleme:** Dört kategori ve tek model ailesi, TurBLiMP’in tamamı veya genel Türkçe yetkinliği değildir. Üretim kalitesi, olgusal doğruluk, muhakeme ve güvenlik bu deneyde ölçülmedi.''')
add('''## 24. Makale için kullanılabilecek bulgu metni

EuroLLM-1.7B’nin FP16, 8 bit ve 4 bit koşulları, dört TurBLiMP kategorisinden eşit sayıda seçilmiş toplam 1.000 minimal çift üzerinde, BOS bağlamlı ve EOS içermeyen token başına ortalama log-olasılık yöntemiyle değerlendirildi. FP16, Q8 ve Q4 doğrulukları sırasıyla %91,2, %91,5 ve %88,4 oldu. Aynı çiftler üzerindeki eşleştirilmiş karşılaştırmada Q8’in FP16’ya göre +0,3 yüzde puan farkı anlamlı değildi (iki yönlü kesin McNemar, Holm p=0,25). Q4’ün −2,8 yüzde puan farkı anlamlıydı (Holm p≈0,000469; muhafazakâr eşleştirilmiş %95 güven aralığı −5,151 ile −0,416 yüzde puan). Q4’te 42 doğru→yanlış ve 14 yanlış→doğru geçiş gözlendi.

Q4 kaybının en büyük tanımlayıcı katkısı özne–yüklem uyumu kategorisinden geldi. Alternatif toplam log-olasılık puanlaması, pilot örneklerin dışlanması, belirtilen adlaştırma etiketlerinin dışlanması ve keşifsel cümle kümesi analizlerinde genel negatif yön korundu. Buna karşılık özne–yüklem kategorisi dışlandığında genel fark küçüldü ve anlamlılık eşiğini geçmedi. Bu bulgular kategori bileşiminin ve puanlama yönteminin raporlanmasının gerekli olduğunu gösteriyor.

Özgün yürütmede Q4, FP16’ya kıyasla yaklaşık %63,8 daha düşük MLX allocator tepe ölçümü ve %22,6 daha kısa puanlama süresi verdi. Bu değerler kontrollü çoklu performans deneyinin sonuçları değil, belirli donanım ve yazılım koşullarındaki ölçümlerdir. Dilbilimsel ön incelemede bazı adlaştırma çiftlerinde iki cümlenin de bağlama göre kabul edilebilir olabileceği belirlendi; benchmark etiketleri değiştirilmeden ayrı duyarlılık analizleri raporlandı. Sonuçlar seçilen örneklem, model ve ölçüm protokolüyle sınırlıdır.''')
add('''## 25. Yeniden çalıştırma ve çıktı envanteri

Proje kökünde kullanılan temel komutlar aşağıdadır. Model komutları GPU erişimi olan oturum gerektirir. Üç model komutunu sırayla, ayrı süreçlerde çalıştırmak gerekir. Mevcut tamamlanmış tekrar klasöründe aynı komutlar yeni puanlama yapmadan kayıtları tanıyıp çıkar; yeni gerçek tekrar istenirse yeni bir çıktı kökü seçilmelidir. Bu rapordaki ilk tekrar, boş ayrı klasörle başladı.''')
add('''```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q scripts tests results/deep_analysis
.venv/bin/python results/deep_analysis/analyze_extended.py
.venv/bin/python results/deep_analysis/rerun_models.py --model fp16 --max-chunks 20 --pause-seconds 0
.venv/bin/python results/deep_analysis/rerun_models.py --model q8 --max-chunks 20 --pause-seconds 0
.venv/bin/python results/deep_analysis/rerun_models.py --model q4 --max-chunks 20 --pause-seconds 0
.venv/bin/python results/deep_analysis/verify_rerun.py
.venv/bin/python results/deep_analysis/build_report.py
```''')
add(f"Ek analiz ortamı: Python {R['versions']['python']}, NumPy {R['versions']['numpy']}, SciPy {R['versions']['scipy']}, Matplotlib {R['versions']['matplotlib']}. SciPy ve Matplotlib bu oturumda proje sanal ortamına kuruldu. Bu oturumdaki donanım okuması Apple M2, 16 GiB RAM (17.179.869.184 bayt) ve macOS 26.4.1 doğruladı; bunlar sistem ölçümü olup GPU/CPU sıcaklığı ya da güç ölçümü değildir. Ayrıntılar `environment.json` içinde. Özgün model metrikleri Python 3.12.15, MLX 0.32.4, MLX-LM 0.32.2 ve batch size 1 bildiriyor. Bootstrap {R['bootstrap_replicates']:,} tekrar, çift örnekleme tohumu {R['seed']}; küme örnekleme tohumu bir fazlası. Sonuçlar bu tohumlarla yeniden üretilebilir; başka tohumla Monte Carlo uçları küçük ölçüde değişebilir.")
table(['Çıktı','İçerik'],[[link('results/deep_analysis/'+p),label] for p,label in [
 ('extended_statistics.json','Bağımsız sayısal denetimler ve tüm ek analiz özetleri'),('recomputed/main_statistics.json','Özgün analiz kodunun ayrı klasörde yeniden ürettiği, eski sonuçla eşleşen çıktı'),('pair_audit.csv','1.000 çift için kararlar, farklar, uzunluklar ve tarama bayrakları'),('manual_review_20_annotated.csv','Özgün seçilmiş 20 çiftin gerekçeli ön değerlendirmesi'),('linguistic_review_58.csv','Karar değiştiren 58 benzersiz çiftin tamamına ait ön değerlendirme'),('predicate_screen_all_pairs.csv','Tüm veri üzerinde aynı yüklem kuralıyla taranan 17 çift'),('template_clusters.csv','Keşifsel yaklaşık cümle kümeleri ve üye kimlikleri'),('rerun_comparison.json','Yeni model çalışmasının eski kayıtlarla karşılaştırması'),('rerun/','Üç modelin yeni 60 puan dosyası ve 60 metrik dosyası'),('figures/','Altı şeklin PNG ve ölçeklenebilir SVG sürümleri'),('provenance_sha256.json','Kaynak ve analiz dosyalarının yeniden üretim için hash envanteri')]])
add('''## 26. Yöntem kaynakları ve sonraki bilimsel adım

Veri setinin kapsamı ve özgün çalışması: [Başar ve arkadaşları, TurBLiMP, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.834/). Bu raporun sayısal bulguları makaleden alınmadı; yerel deney kayıtlarından hesaplandı. Özgün benchmark 16 olguyu içerirken burada yalnızca dört kategori seçilmiş olması, yayın karşılaştırmasında açıkça belirtilmelidir.

Kesin binom hesabının bağımsız uygulama referansı: [SciPy binomtest belgesi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html). Clopper–Pearson sınırları ek analizde beta dağılımı kantilleriyle hesaplandı; mevcut betiğin sayısal kök bulma uygulamasıyla karşılaştırıldı.

Bu veriyle yapılabilecek temel istatistiksel hesaplar, karşılaştırmalar ve başlıca duyarlılık kontrolleri yürütüldü. Yayın öncesindeki en önemli açık iş, tartışmalı çiftlerin bağımsız Türkçe/dilbilim uzmanları tarafından kör incelenmesi ve gerçek kaynak şablonlarının temin edilmesidir. Kaynak performansına ilişkin güçlü bir iddia hedefleniyorsa, sabit ortamda rastgele model sırasıyla tekrarlı ayrı bir ölçüm tasarımı gerekir. Bu açık işler, mevcut hesapların eksik bırakılmasıyla değil eldeki verinin kanıt sınırlarıyla ilgilidir.''')
path=ROOT/'ProjeMarkdown/ISTATISTIKSEL_ANALIZ_VE_DENEY_RAPORU.md'
path.write_text('\n'.join(parts),encoding='utf-8')
print(path);print('words',len(path.read_text().split()),'lines',len(path.read_text().splitlines()))
