from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from copy import deepcopy
from lxml import etree
import re, json, hashlib, shutil, zipfile, math

ROOT=Path('/Users/omerfarukozturk/Desktop/turkce-quantization')
OLD=ROOT/'outputs/socio_submission'
OUT=ROOT/'outputs/Socioinformatics_Final_2026_11_07'
OUT.mkdir(parents=True,exist_ok=True)
TITLE='Weight Quantization and the Stability of Turkish Grammatical Judgments in EuroLLM'
REPO='https://github.com/ozturkomerfaruk/NLP-Turkce-Quantization'
COMMIT='fc3493a63c1a58606fbb4e83d1887f2083547504'
src=OLD/'01_Weight_Quantization_and_Turkish_Linguistic_Measurement_Anonymized.docx'
d=Document(src)
ps=list(d.paragraphs)
replacements={
0:TITLE,
2:'''Language models used in Turkish text research should be evaluated for linguistic stability as well as computational cost. This study examines how weight quantization changes EuroLLM-1.7B's grammatical preferences on 1,000 deduplicated TurBLiMP minimal pairs across four categories. The same sentences are scored with a local FP16 reference and MLX affine 8-bit and 4-bit configurations. Mean token log probability defines the primary decision rule. FP16, Q8, and Q4 achieve accuracies of 91.2%, 91.5%, and 88.4%, respectively. The Q8 difference of +0.3 percentage points is not statistically significant and does not establish equivalence. Q4 loses 2.8 percentage points (Holm-adjusted p = .000469), with 42 correct-to-incorrect and 14 incorrect-to-correct transitions. Subject agreement accounts for 18 of the 28 net lost decisions. Summed scoring increases the Q4 deficit to 3.9 percentage points. Exploratory analyses do not show a monotonic increase in reversals with subword fragmentation. Measured peak MLX allocator memory is 63.8% lower for Q4. For computational social research, the contribution is a reproducible diagnostic of a linguistic component: aggregate accuracy, decision stability, category-specific errors, and scoring choices must be assessed together. The findings establish an accuracy–memory trade-off within the tested setting; their implications for substantive text coding require application-specific evidence.''',
5:'''Computational social research often turns textual material into observations. The credibility of the resulting analysis depends partly on the language-processing instruments used to produce those observations. Grimmer and Stewart (2013) emphasize that automated text analysis requires extensive, problem-specific validation. A practical question arises when a researcher replaces a model with a version that uses less memory: which linguistic decisions remain stable, and which change? This study addresses that question for Turkish through a controlled comparison of weight precision.''',
6:'''The methodological problem is instrument substitution. Lower computational requirements may make a model easier to run locally, but do not establish that it produces interchangeable observations. Agreement with the original model and agreement with an external reference are separate properties. We examine both through a minimal-pair task, in which the model ranks a benchmark-preferred sentence against a minimally different alternative. The measured outcome is grammatical preference on those contrasts. It is a linguistic diagnostic relevant to the preparation of text-analysis methods, rather than a direct measurement of a social concept.''',
9:'''We evaluate three precision configurations of EuroLLM-1.7B (Martins et al., 2024) on the same Turkish minimal pairs from TurBLiMP (Başar et al., 2025). The overarching methodological question is how the choice of weight precision changes the linguistic observations obtained from a fixed Turkish benchmark. RQ1 asks how 8-bit and 4-bit quantization change accuracy and individual decisions relative to FP16. RQ2 asks whether the conclusions remain stable across grammatical categories, sentence-scoring conventions, and sensitivity analyses for related or disputed pairs. RQ3 asks what accuracy–memory trade-off is observed under the measured local workload. An exploratory analysis examines associations between subword fragmentation and Q4 reversals.''',
10:'''The contribution is a documented procedure for evaluating a linguistic component before substituting one model configuration for another. It combines paired decisions, benchmark agreement, grammatical categories, scoring sensitivity, and workload-specific resource measurements. Its relevance to computational social science lies in making the construction and stability of textual observations inspectable. Application validity remains a separate question: the present benchmark cannot establish the quality of sentiment, stance, or other substantive codes.''',
20:'''Gilardi et al. (2023) assess language-model annotation against human judgments in application-specific text tasks. Their approach illustrates why evidence should correspond to the outcome being claimed. The present evaluation instead uses a small base model, sentence likelihoods, and grammatical contrasts. The two settings address different levels of validation. A grammatical diagnostic can reveal sensitivity to a computational change, while an application-level evaluation determines whether the outputs support the intended substantive analysis. Keeping these levels separate allows linguistic evidence to inform research design without overstating its reach.''',
28:'''Three relationships define the evaluation. Configuration stability concerns whether changing weight precision changes the preferred sentence for a fixed pair. Benchmark agreement concerns whether that preference matches the supplied grammatical label. Application validity concerns whether a model output measures the construct of interest in a research corpus (Adcock & Collier, 2001). The first two are observed in this experiment. The third sets the boundary of interpretation: grammaticality scores are not operational definitions of substantive social variables.''',
31:'''These distinctions make the methodological question testable with the existing paired design. A replacement can preserve aggregate accuracy while changing individual observations, or reduce accuracy mainly within a particular grammatical category. Either pattern would be obscured by reporting a single overall score. The comparison therefore evaluates the distribution of changes as well as their net direction. This approach is useful when linguistic decisions form part of a wider research pipeline, even though the benchmark itself does not measure the final substantive construct.''',
44:'''The reference is designated FP16 on the basis of the F16 tensors in the local weight file. The BF16 field in the source configuration describes a different aspect of the model. The quantized files contain packed U32 weights and F16 auxiliary tensors. The Q4 designation therefore refers to weight representation and does not imply that all arithmetic was performed at four bits. Project documentation records the Hugging Face download reference as refs/pr/5 and describes separate Q8 and Q4 conversions from FP16. That reference is mutable; the exact resolved upstream commit and complete conversion commands were not archived, which limits reconstruction of the weight-preparation stage.''',
65:'''Interpretation follows the estimand and measurement scope. The paired accuracy contrast describes agreement with the benchmark, whereas the reversal count describes stability relative to FP16. A nonsignificant contrast does not establish equivalence, and category-specific losses do not identify demographic effects. Resource findings refer to the recorded workload and the memory measurement used. Together, these rules prevent accuracy, stability, and computational cost from becoming interchangeable indicators of overall model suitability.''',
68:'''The saved experiment contains 3,000 pair–configuration records and 6,000 sentence scores. Consistency checks covered identifiers, categories, token counts and digests, and score arithmetic. The maximum absolute discrepancy between the stored mean and the sum divided by token count was 4.24 × 10⁻⁷. Reanalysis of the saved scores reproduced the reported main accuracy and paired-test results. These checks assess record integrity and computational consistency; they do not add new test items.''',
127:'8.2. Implications for computational social research',
128:'''Instrument substitution should be assessed at the level of the observations it changes. The observed Q4 loss is not distributed uniformly: some decisions improve, others deteriorate, and subject agreement contributes most of the net decline. An aggregate score alone would conceal this pattern. For researchers building Turkish text-analysis pipelines, category-specific diagnostics can identify which linguistic distinctions need scrutiny when precision changes. The results do not demonstrate a corresponding error rate in an untested application, but they show why nominally similar model configurations should be evaluated rather than assumed interchangeable.''',
130:'''Benchmark labels introduce another source of uncertainty. The nominalization examples show that a change toward the preferred label may depend on the reading assigned to a sentence. The AI-assisted review is diagnostic and purposively selected, so it cannot establish a corpus-wide labeling-error rate. Accordingly, configuration stability can be measured directly from paired decisions, while claims about linguistic correctness remain conditional on label validity. This distinction is particularly relevant when a convenient benchmark is used to justify the selection of an instrument for a different corpus.''',
131:'8.3. Resource requirements and local research workflows',
132:'''Memory is an operational constraint in local research workflows. The Q4 configuration used less measured allocator memory, while Q8 combined a smaller memory footprint with an accuracy close to FP16 in this sample. These results provide workload-specific information for choosing which configurations to examine further. They do not measure institutional access, adoption, monetary expenditure, or energy consumption. The relevant methodological trade-off is therefore between a recorded resource reduction and a recorded change in grammatical decisions, rather than an inferred social benefit.''',
134:'8.4. Reporting a precision comparison in text research',
135:'''A usable precision comparison should identify its unit of observation, reference labels, numerical configuration, and decision rule. Reporting the same items across configurations makes individual changes traceable. Retaining continuous margins explains how a stable distribution can contain vulnerable decisions near the boundary. Disaggregated results expose category concentration, and sensitivity analyses indicate how scoring or sample composition affects the conclusion. The replication materials retain these components so that readers can inspect the path from sentence scores to reported findings.''',
136:'''For computational social research, this procedure contributes to transparent instrument evaluation. The 2.8-percentage-point Q4 decline is an estimate for this benchmark and scoring rule; it is not a transferable correction factor for other text variables. Researchers can use the diagnostic to document a model-selection decision and its limits. Whether a selected configuration is suitable for a substantive research question must be judged against that question's own evidence requirements, as emphasized in the validation literature (Grimmer & Stewart, 2013).''',
141:'''The pilot overlaps the main sample, the study was not preregistered, and multiplicity was not controlled across all exploratory analyses. The replication package supports reanalysis of the saved scores, but incomplete model-conversion provenance limits reconstruction from the upstream weights. Decision reproducibility on the same inputs is distinct from replication on new items. Timing comparisons were not based on repeated full runs with randomized configuration order.''',
142:'''The empirical scope is a Turkish grammatical benchmark. The study does not test a substantive social-science variable or estimate how quantization changes a downstream research conclusion. Its methodological contribution is confined to the evaluation of a linguistic component used in preparing text-analysis instruments. This boundary should be retained when using the findings to motivate broader applications.''',
145:'''The exploratory analysis did not support an increase in decision changes with greater fragmentation. The inverse conditional association depended on the definition of length. Alongside a 63.8% reduction in measured peak MLX allocator memory, these findings quantify an accuracy–memory trade-off for the tested model, sample, and scoring procedure. The scope of this finding is bounded by benchmark-label uncertainty, related templates, incomplete conversion provenance, and the limited performance-measurement design.''',
146:'''For computational research using Turkish texts, model configuration and measurement procedure should be reported together. Overall accuracy, item-level stability, grammatical categories, and resource scope each contribute different evidence about a potential substitution. The present comparison makes that evidence inspectable through a fixed sample and saved scores. It supports transparent evaluation of a Turkish linguistic component while reserving claims about substantive research validity for the relevant application.''',
149:'''The original TurBLiMP benchmark is available at https://github.com/ezgibasar/TurBLiMP (Başar et al., 2025). This experiment uses source revision 297de13fb7a0ce524fe32e8b175c6b5255d66960 and the sample digest reported in Section 4. The supplementary replication package provides the selected sample, source CSVs, saved scores, analysis code, model configuration files, figure assets, and a SHA-256 inventory. It supports reanalysis without model inference and excludes model weights. The author's public project repository contains the core experiment code and saved results; its identifying URL is provided separately to the editor to preserve anonymous review. The supplementary package additionally supplies the selected data and exploratory analyses. The recorded download reference and incomplete conversion history constrain reconstruction from the original weights.''',
151:'''OpenAI Codex assisted with Python code development, analysis checks, literature searching, English-language revision, and document preparation in this project. The manuscript revision of 7 November 2026 used Codex with GPT-6.1; exact model identifiers for earlier assistance were not retained. Preliminary linguistic comments on selected sentence pairs were AI-assisted and are treated as exploratory rather than independent human judgments. Experimental predictions are the likelihood scores produced by the EuroLLM configurations described in Sections 5 and 6; Codex outputs are not additional benchmark observations. The sole human author is responsible for the research, submitted text, references, and interpretation.''',
153:'''This study analyzes existing benchmark sentences and model likelihood scores. It did not recruit human participants, collect participant responses, or analyze identifiable personal records. Participant consent is therefore not applicable to this secondary benchmark analysis. No ethics-committee approval or formal exemption was obtained. The study reports no human-annotation experiment.''',
}
for i,t in replacements.items():ps[i].text=t
# Restore coherent heading levels while retaining the journal's numbered headings.
for p in d.paragraphs:
 if re.match(r'^\d+\.\d+\.',p.text):p.style=d.styles['Heading 3']
 elif re.match(r'^\d+\. ',p.text):p.style=d.styles['Heading 2']
for style in d.styles:
 if style.type==1:
  if style.name.startswith('Heading') or style.name=='Title':
   style.font.color.rgb=RGBColor(0,0,0)
   pp=style.element.find(qn('w:pPr'))
   if pp is not None:
    for e in list(pp):
     if e.tag==qn('w:pBdr'):pp.remove(e)
for p in d.paragraphs:
 if p.style.name.startswith('Heading') or p.style.name=='Title':
  for r in p.runs:r.font.color.rgb=RGBColor(0,0,0)
d.core_properties.author='';d.core_properties.last_modified_by='';d.core_properties.title=TITLE
main=OUT/'01_Socioinformatics_Quantization_Turkish_Grammatical_Judgments_Anonymous.docx'
d.save(main)

def base_doc():
 x=Document()
 s=x.sections[0];s.page_width=Cm(21);s.page_height=Cm(29.7)
 s.top_margin=s.bottom_margin=s.left_margin=s.right_margin=Cm(2.5)
 for name in ['Normal','Title','Heading 1','Heading 2']:
  st=x.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
  st.font.size=Pt(11 if name=='Normal' else 14 if name=='Title' else 12)
  st.paragraph_format.space_after=Pt(6)
  pp=st.element.find(qn('w:pPr'))
  if pp is not None:
   for e in list(pp):
    if e.tag==qn('w:pBdr'):pp.remove(e)
 x.styles['Normal'].paragraph_format.line_spacing=1.15
 return x
t=base_doc();t.add_paragraph(TITLE,'Title')
for line in ['Article type: Methods and Data Article','Author and corresponding author: Ömer Faruk Öztürk','Affiliation: Master’s student, Artificial Intelligence and Data Engineering, Ankara University, Ankara, Türkiye','Email: omerfaruko@ankara.edu.tr','ORCID: https://orcid.org/0009-0005-6176-0251']:t.add_paragraph(line)
for h,body in [
 ('Author contributions','Ömer Faruk Öztürk: Conceptualization, Methodology, Software, Investigation, Data curation, Formal analysis, Validation, Visualization, Writing – original draft, Writing – review and editing. The author conducted the project with the AI assistance disclosed in the manuscript and assumes responsibility for all contributions.'),
 ('Funding','This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.'),
 ('Conflict of interest','The author declares no conflicts of interest.'),
 ('Data and code repository',f'{REPO}\nRepository revision inspected on 7 November 2026: {COMMIT}. The repository contains core code and saved results; the supplementary replication package also provides the selected sample and exploratory analyses. No DOI has been assigned. The identifying repository URL is kept on this title page for anonymous review.')]:
 t.add_paragraph(h,'Heading 2');t.add_paragraph(body)
t.core_properties.author='Ömer Faruk Öztürk';t.core_properties.title=TITLE
t.save(OUT/'02_Socioinformatics_Title_Page_Omer_Faruk_Ozturk.docx')

# Independent numerical verification from frozen records, without touching experiment outputs.
check={}
records={}
for model in ['fp16','q8','q4']:
 rr=[json.loads(line) for p in sorted((ROOT/'results/main'/model).glob('chunk_*.jsonl')) for line in p.read_text().splitlines() if line.strip()]
 assert len(rr)==1000 and len({r['pair_id'] for r in rr})==1000
 records[model]={r['pair_id']:r for r in rr}
 check[model]={'pairs':len(rr),'correct':sum(r['correct'] for r in rr)}
for model in ['q8','q4']:
 a=records['fp16'];b=records[model];down=up=0
 for k,r in a.items():
  v=b[k];down+=r['correct'] and not v['correct'];up+=not r['correct'] and v['correct']
  for side in ['good','bad']:assert r[side]['token_ids_sha256']==v[side]['token_ids_sha256']
 n=down+up;p=min(1,2*sum(math.comb(n,i) for i in range(min(down,up)+1))/2**n)
 check[model].update(down=int(down),up=int(up),exact_p=p)
assert [check[k]['correct'] for k in ['fp16','q8','q4']]==[912,915,884]
assert check['q4']['down']==42 and check['q4']['up']==14
assert len(d.tables)==6 and len(d.inline_shapes)==2
old=Document(src)
assert [[c.text for r in z.rows for c in r.cells] for z in d.tables]==[[c.text for r in z.rows for c in r.cells] for z in old.tables]
ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
assert [etree.tostring(e) for e in d.element.xpath('//m:oMath')]==[etree.tostring(e) for e in old.element.xpath('//m:oMath')]
full='\n'.join(p.text for p in d.paragraphs)
assert not any(w in full for w in ['news sentence','actor category','responsibility-attribution','ozturkomerfaruk','Ömer Faruk','omerfaruko@'])
start=full.index('1. Introduction');end=full.index('11. Research Transparency')
body='\n'.join(p.text for p in d.paragraphs if p.text and not p.text.startswith(('Table ','Figure ','Source:')))
word_count=len(re.findall(r'\S+',body[body.index('1. Introduction'):body.index('11. Research Transparency')]))
abstract_words=len(ps[2].text.split())
assert 4000<=word_count<=7000 and 150<=abstract_words<=250
check.update(main_text_words_approx=word_count,abstract_words=abstract_words,tables=6,figures=2)
(OUT/'verification.json').write_text(json.dumps(check,indent=2))

# Prepare a clean supplement without unused prospective experiments or forms.
pack=OUT/'replication_package'
if pack.exists():shutil.rmtree(pack)
shutil.copytree(OLD/'replication_package',pack)
shutil.copytree(OLD/'figures',pack/'figures',dirs_exist_ok=True)
readme=(pack/'README.md').read_text()
readme=readme.replace('# EuroLLM Turkish linguistic measurement replication materials','# EuroLLM Turkish grammatical judgments replication materials')
readme=readme.replace('Local preparation date: 6 November 2026. This package has not been uploaded and has no DOI.',f'Package preparation date: 7 November 2026. Public core project: {REPO} . Inspected revision: {COMMIT}. The supplementary package additionally contains the selected sample and exploratory analyses; it has no DOI.')
readme=readme.replace('it is not evidence of a completed social-science coding study or independent human annotation.','the evaluation consists of the completed minimal-pair experiment and analyses of its saved scores.')
readme=readme[:readme.index('## Submission steps still needed')]+'''## Use with the manuscript
Provide this ZIP as supplementary replication material. The public repository provides core experiment code and saved results; the selected JSONL sample and exploratory fragmentation files are supplied here. No new inference, news-text experiment, or independent human-labeling study is included. Keep this package with the submitted manuscript version. Code reuse remains subject to the author's licensing decision and upstream data attribution requirements.
'''
(pack/'README.md').write_text(readme)
inventory={str(p.relative_to(pack)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(pack.rglob('*')) if p.is_file() and p.name!='PACKAGE_SHA256.json' and '__pycache__' not in p.parts}
(pack/'PACKAGE_SHA256.json').write_text(json.dumps(inventory,indent=2))
zpath=OUT/'04_Socioinformatics_Replication_Materials.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(pack.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(Path('replication_package')/p.relative_to(pack)))
with zipfile.ZipFile(zpath) as z:assert z.testzip() is None

n=base_doc();n.add_paragraph('Socioinformatics gönderim notları','Title')
for h,body in [
 ('Hazırlanan makale',f'Çalışma yalnızca tamamlanan 1.000 cümle çiftlik FP16–Q8–Q4 deneyine dayanır. Haber deneyi önerisi ve boş değerlendirici formları bu gönderim paketinden çıkarılmıştır. Ana metin yaklaşık {word_count:,} kelime, özet {abstract_words} kelimedir. Altı tablo, iki şekil ve deney sonuçları korunmuştur. Makale türü önerisi Methods and Data Article olarak bırakılmıştır.'),
 ('Kullanılacak dosyalar','01 ile başlayan dosya anonim ana makaledir. 02 ile başlayan dosya yazar bilgilerini ve beyanları içerir. 04 ile başlayan ZIP veri ve kod ekidir. Bu not dosyası sizin içindir; makalenin bir parçası değildir. Yüksek çözünürlüklü şekiller figures klasöründe ayrıca verilmiştir.'),
 ('Tamamlanan bilgiler','Tek yazar Ömer Faruk Öztürk; Ankara Üniversitesi, Yapay Zekâ ve Veri Mühendisliği yüksek lisans öğrencisi; omerfaruko@ankara.edu.tr; ORCID 0009-0005-6176-0251. Maddi destek ve çıkar çatışması bulunmadığı beyanları eklendi. Öğrencilik ve bölüm ilişkisi tek başına ayrıca bir çıkar çatışması varmış gibi sunulmadı.'),
 ('GitHub ve veri paylaşımı',f'Kamuya açık depo kontrol edildi: {REPO} . Depoda çekirdek deney kodları ve sonuçlar var. Seçilmiş JSONL veri ve keşifsel parçalanma analizleri yerel ek pakette ayrıca sağlandı. Kör değerlendirmede kimliği açığa çıkarmamak için kişisel depo bağlantısı başlık sayfasında tutuldu. Gönderimde ZIP ekini de yükleyin; yalnızca GitHub bağlantısı bu ekin bütün içeriğini sağlamıyor.'),
 ('Son yazar kontrolleri','Kaynakların Web of Science durumunu istediğiniz gibi siz kontrol edeceksiniz; WoS doğrulaması yapılmış kabul edilmemelidir. Dergi sistemindeki telif/lisans ve katkı formlarını okuyup gerçek durumunuza göre doldurun. Ana metindeki etik açıklama, insan katılımcı bulunmayan ikincil benchmark analizini tarif eder; resmî muafiyet belgesi değildir. Önceki AI araçlarının sürümleri kayıtlarda yoksa tahmin ederek eklemeyin.'),
 ('Kapsam ve değerlendirme','Metin, Türkçe metin araştırmalarında kullanılan dilsel bir bileşenin değerlendirilmesine yöntemsel katkı olarak çerçevelendi. Yeni bir sosyal bilim veri seti veya uygulama sonucu iddia edilmedi. Derginin kapsam değerlendirmesi editöre aittir. Bu not ek deney yapmanız gerektiği anlamına gelmez.'),
]:n.add_paragraph(h,'Heading 2');n.add_paragraph(body)
n.add_paragraph('Kurallar: https://dergipark.org.tr/en/pub/socio/writing-rules\nKapsam: https://dergipark.org.tr/en/pub/socio/aim-and-scope\nKontrol tarihi: 7 Kasım 2026')
n.save(OUT/'03_Socioinformatics_Gonderim_Notlari.docx')
shutil.copytree(OLD/'figures',OUT/'figures',dirs_exist_ok=True)
print(json.dumps(check,indent=2))
print('Output directory:',OUT)
