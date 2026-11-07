from pathlib import Path
import json,csv,hashlib,shutil,random,math,zipfile
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT=Path('/Users/omerfarukozturk/Desktop/turkce-quantization');OUT=ROOT/'outputs/socio_submission';PACK=OUT/'replication_package';PACK.mkdir(exist_ok=True)
# Independent checks of the saved pair-level observations.
models={}
for name in ['fp16','q8','q4']:
 rows=[json.loads(l) for p in sorted((ROOT/'results/main'/name).glob('chunk_*.jsonl')) for l in p.read_text().splitlines() if l]
 assert len(rows)==1000 and len({r['pair_id'] for r in rows})==1000
 models[name]={r['pair_id']:r for r in rows}
 assert sum(r['good']['mean_logp']>r['bad']['mean_logp'] for r in rows)=={'fp16':912,'q8':915,'q4':884}[name]
 for r in rows:
  assert (r['good']['mean_logp']>r['bad']['mean_logp'])==r['correct']
  for side in ['good','bad']:
   assert abs(r[side]['mean_logp']-r[side]['total_logp']/r[side]['n_tokens'])<5e-7
checks={}
for name in ['q8','q4']:
 assert set(models[name])==set(models['fp16'])
 d=u=0
 for k,a in models['fp16'].items():
  b=models[name][k]
  d+=a['correct'] and not b['correct'];u+=not a['correct'] and b['correct']
  for side in ['good','bad']:assert a[side]['token_ids_sha256']==b[side]['token_ids_sha256']
 p=min(1.0,2*sum(math.comb(d+u,k) for k in range(min(d,u)+1))/2**(d+u))
 checks[name]={'degradations':d,'improvements':u,'raw_p':p,'delta_pp':100*(u-d)/1000}
assert checks['q4']['degradations']==42 and checks['q4']['improvements']==14
assert checks['q8']['degradations']==0 and checks['q8']['improvements']==3
assert abs(2*checks['q4']['raw_p']-.00046889172183839367)<1e-12
sample=ROOT/'data/processed/turblimp_1000.jsonl'
assert hashlib.sha256(sample.read_bytes()).hexdigest()=='bcb2c7f25475f8290d1511d742e1e9f8f515fd29df11dd3e3f64f2f53bcb4835'
checks['sample_sha256']=hashlib.sha256(sample.read_bytes()).hexdigest()
(OUT/'work/numeric_verification.json').write_text(json.dumps(checks,indent=2))
# Package existing research materials locally, without uploading them or claiming a DOI.
folders=['scripts','tests','data','results/main','results/analysis','results/fragmentation_analysis','results/deep_analysis']
for folder in folders:
 for p in (ROOT/folder).rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix not in ['.py','.json','.jsonl','.csv','.txt','.png','.svg']:continue
  if p.name in ['build_report.py','final_audit.py']:continue # require historical reports or full model weights
  dest=PACK/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
for name in ['fp16','q8','q4']:
 for p in (ROOT/'models'/('eurollm-'+name)).glob('*.json'):
  dest=PACK/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
(PACK/'README.md').write_text('''# EuroLLM Turkish linguistic measurement replication materials

Local preparation date: 6 November 2026. This package has not been uploaded and has no DOI.

## Scope
The package contains the original selected TurBLiMP sample, source CSVs, saved pair-level model scores, statistical outputs, analysis code, and model configuration files. It contains no model weights. It supports reanalysis of the reported benchmark experiment; it is not evidence of a completed social-science coding study or independent human annotation.

## Provenance and attribution
TurBLiMP: Başar, Padovani, Jumelet, and Bisazza (2025), https://doi.org/10.18653/v1/2025.emnlp-main.834 . Repository: https://github.com/ezgibasar/TurBLiMP . Selected source revision: 297de13fb7a0ce524fe32e8b175c6b5255d66960. The repository README states CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The source CSVs are retained; the JSONL sample is a deduplicated, sampled, reformatted derivative. Credit the original authors and indicate those modifications when redistributing.

EuroLLM: Martins et al. (2024), https://doi.org/10.48550/arXiv.2409.16235 . Model card: https://huggingface.co/utter-project/EuroLLM-1.7B . Model weights are excluded. MLX documentation: https://ml-explore.github.io/mlx/ . Do not infer a complete conversion recipe from the configuration files; upstream model revision and conversion commands were incompletely archived.

The author's own code has no newly assigned license in this preparation. Decide the code license before public deposit. Existing upstream rights remain unchanged.

## Reanalysis without model inference
From the extracted package root, run:

    python3 scripts/analyze_main.py

This reads frozen scores and regenerates results/analysis outputs. It does not rerun the model. Preserve an untouched copy of the archive if comparing regenerated and saved outputs. Main analysis uses Python's standard library. The original inference used Python 3.12.15 and the additional analyses report NumPy 2.5.4, SciPy 1.18.1, and Matplotlib 3.11.2; environment metadata are under results/deep_analysis.

To recompute exploratory fragmentation estimates, use a compatible scientific-Python environment and run:

    python3 scripts/analyze_fragmentation.py --bootstrap 2000

The extended forensic audit script requires the original weight files for tensor inspection and size calculations. It is retained for transparency, but cannot be fully rerun from this weights-excluded package. The historical provenance inventory may name files outside the package; PACKAGE_SHA256.json is the inventory of files actually supplied here.

## Submission steps still needed
Check third-party attribution and the intended code license, deposit the materials in a persistent repository, obtain a reviewer-accessible link, and update the manuscript's Data and Code Availability statement. Do not describe this local archive as publicly available until the deposit is complete.
''',encoding='utf-8')
# Blinded full-sample linguistic review forms. No annotations or model outputs supplied to reviewers.
REV=OUT/'human_validation_forms';REV.mkdir(exist_ok=True)
rows=[json.loads(l) for l in sample.read_text().splitlines()]
print('dataset fields',list(rows[0]))
rng=random.Random(20261106);rng.shuffle(rows)
forms=[];key=[]
for i,r in enumerate(rows,1):
 good=r.get('sentence_good',r.get('good',r.get('sentence_grammatical')))
 bad=r.get('sentence_bad',r.get('bad',r.get('sentence_ungrammatical')))
 if good is None or bad is None:raise ValueError(r.keys())
 flip=rng.choice([False,True]);a,b=(bad,good) if flip else (good,bad)
 rid=f'L{i:04d}'
 forms.append({'review_id':rid,'sentence_A':a,'sentence_B':b,'preferred_A_B_BOTH_NEITHER_UNCLEAR':'','context_required_Y_N':'','comment':''})
 key.append({'review_id':rid,'pair_id':r['pair_id'],'category':r['category'],'benchmark_preferred':'B' if flip else 'A'})
def write_csv(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write_csv(REV/'reviewer_1_blinded.csv',forms);write_csv(REV/'reviewer_2_blinded.csv',forms);write_csv(REV/'coordinator_key_do_not_share_with_reviewers.csv',key)
(REV/'README_TR.txt').write_text('''Bu dosyalar tamamlanmış insan değerlendirmesi değildir. İki değerlendirici formu bilerek boştur.

İki bağımsız Türkçe uzmanına yalnız reviewer_1_blinded.csv veya reviewer_2_blinded.csv dosyasını verin. Koordinatör anahtarını, model tahminlerini ve önceki AI yorumlarını göstermeyin. Aynı kişinin formu iki kez doldurması bağımsız değerlendirme değildir. Çift sırası ve A/B yönü seed 20261106 ile rastgeleleştirildi; tüm 1000 çift dahildir.

Görev: bağlamsız standart Türkçede A mı B mi daha kabul edilebilir? BOTH: ikisi de kabul edilebilir; NEITHER: ikisi de kabul edilemez; UNCLEAR: yeterli karar verilemiyor. Bağlam gerekiyorsa context_required_Y_N sütununa Y yazın. Karar gerekçesini comment alanına ekleyin. Değerlendiriciler ilk turda birbirlerinin kararlarını görmemeli.

Önce ayrı bir eğitim malzemesiyle yönergeyi netleştirin; ana değerlendirmede protokolü sonuca göre değiştirmeyin. Değerlendirici niteliklerini, karar tarihlerini, ilk bağımsız kararları, uzlaşma sürecini ve kurumun gerektirdiği etik/izin değerlendirmesini kaydedin. Kategori bazında ham uyumu ve uygun şans-düzeltilmiş uyum ölçüsünü belirsizliğiyle raporlayın; uzlaşılmış etiketler üzerindeki model sonuçlarını ayrı duyarlılık analizi olarak hesaplayın.

Bu işlem mevcut dilsel benchmarkı denetler. Sosyal bilimsel haber kodlama deneyinin yerine geçmez.
''',encoding='utf-8')
# Author-facing readiness and experimental extension document.
n=Document();n.add_paragraph('Socioinformatics için düzenleme ve gönderim notları','Title')
def h(t):n.add_paragraph(t,'Heading 1')
def p(t):n.add_paragraph(t)
h('1. Hazırlanan sürüm')
p('Ana makale, mevcut deney korunarak yöntem ve veri makalesi hedefiyle yeniden düzenlendi. Başlık nicemleme ile Türkçe dilsel ölçümlerin güvenilirliğine odaklanıyor. Sosyal bilimsel ölçüm geçerliliği, girişe, literatüre, araştırma sorularına, ölçüm çerçevesine ve tartışmaya bağlandı. Sosyal bilimsel kodlama deneyi yapılmış gibi sunulmadı.')
checksdoc=json.loads((OUT/'work/document_checks.json').read_text())
p(f"Ana metin yaklaşık {checksdoc['body_words_excluding_abstract_references_captions_tables']:,} kelime; özet {checksdoc['abstract_words']} kelime; altı anahtar kelime ve {checksdoc['references']} kaynak bulunuyor. Altı tablonun sayısal içeriği, denklemler ve iki şekil korunmuştur. A4, 2,5 cm kenar boşluğu, 11 punto gövde metni, 1,5 satır aralığı ve iki yana yaslama uygulanmıştır. Kaynakça ve tablo hücrelerinde okunabilirlik için özel hizalama kullanılmıştır.")
p('Ana dosyadan yazar adı ve belge yazar metaverileri çıkarıldı. Ayrı başlık sayfasına Ankara Üniversitesi, Yapay Zekâ ve Veri Mühendisliği yüksek lisans öğrencisi bilgisi ve omerfaruko@ankara.edu.tr e-postası eklendi. Kurum adının İngilizce program karşılığı gönderim öncesinde resmî kullanımla karşılaştırılmalıdır.')
h('2. Gönderimden önce tamamlanması gerekenler')
for text in [
'ORCID numaranızı başlık sayfasına yazın. ORCID, araştırmacıya ait kalıcı kimlik numarasıdır; kurum veya e-posta yerine geçmez.',
'Finansman ve çıkar çatışması alanlarını gerçek durumunuza göre tamamlayın. Destek veya çıkar çatışması olmadığı henüz teyit edilmediğinden sizin adınıza “yoktur” beyanı yazılmadı. CRediT katkı rollerini doğrulayın.',
'AI Use Declaration bölümüne önceki çalışmalarda kullandığınız AI araçlarını ve bilinen model/sürüm bilgilerini ekleyin. Codex ile bu düzenleme tarihi belirtildi; bilinemeyen sürümler uydurulmadı.',
'Kurumunuzun bu ikincil benchmark analizi için etik kurul onayı veya muafiyet gerektirip gerektirmediğini teyit edin. Ana metin insan katılımcı toplanmadığını açıklıyor; resmî muafiyet kararı varmış gibi sunmuyor.',
'Yerel veri/kod paketini kalıcı bir arşive koyup erişilebilir bağlantı veya DOI alın; ana metindeki Data and Code Availability bölümünü güncelleyin. Methods and Data Article türünde veri ve kod paylaşımı zorunludur. Bu hazırlıkta dışarıya yükleme yapılmadı.',
'Kaynakların WoS indeks kontrolünü sizin son talimatınız doğrultusunda siz yapacaksınız. DOI ve bilimsel uygunluk, WoS kaydıyla aynı şey değildir. Model, benchmark ve yazılımın özgün kaynaklarını koruyun.',
'Derginin telif/lisans ve katkı formunu gerçek bilgilerle tamamlayıp imzalayın; aynı anda başka dergide değerlendirmede olmama ve yazar onayı gibi kişisel beyanlar otomatik verilmedi.']:p(text)
p('Kapsam bakımından kalan önemli risk: Bu çalışma hâlâ Türkçe dilbilgisi benchmarkına dayanıyor; gerçek bir sosyal bilimsel değişkeni ölçen ikinci deney yapılmadı. Yöntemsel çerçeve güçlendirildi, fakat bu işlem dergi kapsamına kabul garantisi sağlamaz. Bu nedenle dosya “eksiksiz ve kesin gönderime hazır” olarak değerlendirilmemelidir.')
h('3. Ek sosyal bilim deneyi için uygulanabilir protokol')
p('Amaç: Nicemlemenin Türkçe haberlerde eylemi gerçekleştiren aktörün kodlanmasını ve aktör kategorilerinin tahmin edilen oranlarını değiştirip değiştirmediğini değerlendirmek. “Kim yaptı?” ile ahlaki/hukuki sorumluluk birbirine karıştırılmamalıdır. Sorumluluk çerçevesi ayrıca incelenecekse ayrı tanım ve etiket gerekir. Bu protokol uygulanmamış bir araştırma tasarımıdır; sonuç değildir.')
p('Örnekleme: Belirlenmiş bir konu, dönem ve kaynak listesinden haber paragrafları seçin. Örnekleme birimini ve dışlama kurallarını sonuçlar görülmeden sabitleyin. Aynı haberin yeniden yayımlanmış sürümlerini ve aynı olay etrafındaki bağımlılığı belirleyin. Pilot/geliştirme malzemesini ana değerlendirmeden ayırın. Ana örneklem büyüklüğünü pilotta gözlenen uyumsuzluk oranı, önemli sayılacak değişim ve hedeflenen güven aralığına göre gerekçelendirin; gelişigüzel bir sayı yeterli değildir.')
p('Kodlama yönergesi: Hedef eylemi ve aktörü tanımlayın. Etiketler önerisi: kamu kurumu/kamu görevlisi; özel kuruluş; sivil toplum/örgüt; birey; birden çok aktör; belirlenemiyor. Metnin göstermediği niyet veya sorumluluk çıkarılmamalıdır. Alıntı yapan kişi ile eylemi yapan kişi ayrılmalıdır. Etiketlerin korpusa uygunluğu geliştirme aşamasında sınanmalı; ana değerlendirmede sabitlenmelidir.')
p('İnsan referansı: İki bağımsız değerlendirici model çıktısını görmeden kodlama yapmalı. Önce bağımsız kararlar saklanmalı; ardından uyuşmazlıklar gerekçeli olarak uzlaştırılmalı. İlk tur uyumu, kategori dağılımları ve belirsiz örnekler raporlanmalı. İzin ve etik değerlendirme veri toplamadan önce yapılmalı. Mevcut makaledeki AI destekli yorumlar insan referansı olarak kullanılmamalıdır.')
p('Model uygulaması: EuroLLM-1.7B bir temel modeldir. Sabit bir istem ve aday etiket puanlama yöntemi yalnız geliştirme verisinde pilotlanmalıdır. Kısa etiketleri doğrudan karşılaştırmanın token uzunluğu etkisi dikkate alınmalıdır. FP16, Q8 ve Q4 aynı örnek, aday etiket sırası, bağlam ve karar kuralını kullanmalıdır. İnce ayar gerekiyorsa bu yeni müdahale açıkça tanımlanır; mevcut deneyin aynısıymış gibi sunulmaz.')
p('Birincil çıktılar: İnsan referansına göre doğruluk ve sınıf bazında hata; FP16 ile Q8/Q4 arasındaki eşleştirilmiş karar değişimleri; her aktör kategorisinin tahmin edilen oranındaki fark. Oran farkları, hedef korpusun örnekleme tasarımına uygun biçimde hesaplanmalı. Belirsizlik için aynı haber/olay kümesini birlikte tutan bir yeniden örnekleme tasarımı kullanılabilir. Ana karşılaştırma ailesi, kabul edilebilir kayıp varsa tolerans ve dışlama kuralları önceden sabitlenmelidir.')
p('Raporlama: Sonuçlar olumlu da olumsuz da olsa mevcut dilbilgisi deneyiyle birlikte sunulur. Minimal çift doğruluğu ile haber kodlama hatası arasında nedensellik kendiliğinden kurulmaz. İkinci deney tamamlanırsa makalenin başlığı, özeti, yöntemi ve sonuçları yeni kapsamı yansıtacak biçimde tekrar düzenlenmelidir.')
h('4. Mevcut benchmark için bağımsız dilsel kontrol')
p('Paket içindeki human_validation_forms klasörü, tüm 1.000 çift için iki boş ve körlenmiş değerlendirici formu içerir. Cümle yönü ve sırası rastgeleleştirilmiştir. Koordinatör anahtarı ayrı tutulmuştur; değerlendiricilere verilmemelidir. Formlarda model kararları ve AI yorumları bulunmaz. Bu formlar doldurulmuş veri veya gerçekleştirilmiş bir değerlendirme değildir.')
p('İki bağımsız uzmanın değerlendirmesinden sonra kategori bazında etiket uyuşmazlıkları ve model sonuçlarının insan etiketlerine duyarlılığı hesaplanabilir. Bu çalışma dilbilgisi benchmarkını güçlendirir; sosyal bilimsel uygulama deneyinin yerine geçmez.')
h('5. Kaynak ve kanıt notları')
p('Yeni kuramsal dayanaklar: Adcock ve Collier (2001), ölçüm geçerliliğinin kavram ve gözlem düzeyleri; Grimmer ve Stewart (2013), otomatik metin analizinde probleme özgü doğrulama; Gilardi ve arkadaşları (2023), insan anotasyonlarıyla karşılaştırılan dil modeli tabanlı metin kodlama. Bu kaynaklar kendi nicemleme sonuçlarınızın kanıtı olarak kullanılmadı.')
p('Adcock ve Collier: https://doi.org/10.1017/S0003055401003100\nGrimmer ve Stewart: https://doi.org/10.1093/pan/mps028\nGilardi ve arkadaşları: https://doi.org/10.1073/pnas.2305016120')
p('Deney kontrolü: 3.000 çift–yapılandırma kaydında kimlikler, token imzaları ve doğruluk değerleri yeniden kontrol edildi. Doğru sayıları 912/915/884; Q4 değişimleri 42 kötüleşme ve 14 iyileşme; Q8 değişimleri 0 kötüleşme ve 3 iyileşme olarak doğrulandı. McNemar p değeri ve örneklem SHA-256 özeti yeniden hesaplandı. Bu kontrol yeni model çalıştırması veya bağımsız replikasyon değildir.')
p('Dergi kuralları: https://dergipark.org.tr/en/pub/socio/writing-rules\nDergi kapsamı: https://dergipark.org.tr/en/pub/socio/aim-and-scope\nKontrol tarihi: 6 Kasım 2026.')
for s in n.sections:s.page_width=Cm(21);s.page_height=Cm(29.7);s.top_margin=s.bottom_margin=s.left_margin=s.right_margin=Cm(2.5)
for name in ['Normal','Title','Heading 1']:
 st=n.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0);st.font.size=Pt(11 if name=='Normal' else 15 if name=='Title' else 12)
 st.paragraph_format.line_spacing=1.15;st.paragraph_format.space_after=Pt(7)
 if name!='Normal':st.font.bold=True;st.paragraph_format.keep_with_next=True
for pp in n.paragraphs:pp.paragraph_format.widow_control=True
n.core_properties.author='';n.core_properties.last_modified_by='';n.save(OUT/'03_Gonderim_Notlari_ve_Ek_Deney_Protokolu.docx')
# Integrity manifest is for actual packaged files; historical manifests remain labeled as historical in README.
inventory={str(p.relative_to(PACK)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(PACK.rglob('*')) if p.is_file()}
(PACK/'PACKAGE_SHA256.json').write_text(json.dumps(inventory,indent=2))
print('Verified primary data; prepared',len(inventory),'replication files and',len(forms),'blank review rows per reviewer.')
