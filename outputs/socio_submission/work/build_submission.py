from pathlib import Path
from copy import deepcopy
from io import BytesIO
from datetime import datetime, timezone
import csv,json,re,hashlib,shutil,zipfile,random
from docx import Document
from docx.shared import Cm,Pt,RGBColor,Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image
ROOT=Path('/Users/omerfarukozturk/Desktop/turkce-quantization')
OUT=ROOT/'outputs/socio_submission'
SRC=Path('/Users/omerfarukozturk/Desktop/Effects_of_Weight_Quantization_on_Turkish_Morphosyntactic_Performance_in_EuroLLM.docx')
TITLE='Weight Quantization and the Reliability of Turkish Linguistic Measurements in EuroLLM'
doc=Document(SRC)
changes=[]
def find(prefix):
 return next(p for p in doc.paragraphs if p.text.startswith(prefix))
def replace(prefix,text):
 p=find(prefix);changes.append({'before':p.text,'after':text});p.text=text;return p
def before(anchor,text,style='Normal'):
 return anchor.insert_paragraph_before(text,style)
def remove(p):p._element.getparent().remove(p._element)
def insert_block(anchor,items):
 for style,text in items:before(anchor,text,style)
replace('Effects of Weight Quantization',TITLE)
remove(find('Ömer Faruk Öztürk'))
replace('Weight quantization reduces', 'Computational research on Turkish texts requires transparent evidence about the stability of its linguistic instruments. This study audits how weight quantization changes grammatical preferences in EuroLLM-1.7B, treating minimal-pair performance as a bounded diagnostic rather than a validated measure of social meaning. The local FP16 reference is compared with MLX affine 8-bit and 4-bit configurations on 1,000 deduplicated TurBLiMP pairs across four categories. Mean token log probability defines the primary score. FP16, Q8, and Q4 achieve accuracies of 91.2%, 91.5%, and 88.4%. The Q8 difference of +0.3 percentage points is not significant and does not establish equivalence. Q4 loses 2.8 percentage points (Holm-adjusted p = .000469), with 42 correct-to-incorrect and 14 incorrect-to-correct transitions. Subject agreement accounts for 18 of the 28 net lost decisions. Summed scoring increases the Q4 deficit to 3.9 percentage points, showing sensitivity to operationalization. Exploratory analyses do not support monotonically increasing reversals with subword fragmentation. Measured peak MLX allocator memory is 63.8% lower for Q4. A measurement-oriented audit links these findings to requirements for validating local text-analysis instruments: explicit constructs, paired comparisons, disaggregated errors, label review, and task-specific validation. The experiment establishes a model-specific accuracy–memory trade-off; it does not establish the validity of downstream social-science coding, equitable access, or equivalence across precision settings.')
replace('Keywords:', 'Keywords: weight quantization, Turkish, measurement validity, computational social science, morphosyntax, reproducibility.')
intro=find('Running language models on local hardware')
insert_block(intro,[('Normal','When a language model is used to turn texts into research observations, the stability of that instrument matters alongside its computational cost. A model may help identify linguistic patterns without supplying valid measurements of the social concepts a researcher intends to study. Grimmer and Stewart (2013) emphasize that automated text analysis requires extensive, problem-specific validation. This concern motivates the present question: what evidence should researchers require before treating a memory-reduced language model as a dependable linguistic component of a Turkish text-analysis workflow?'),('Normal','The empirical contribution is an audit of grammatical preference under changes in weight precision. It addresses one part of instrument evaluation: whether the same fixed texts receive the same linguistic decisions, how those decisions relate to benchmark labels, and how the answer depends on scoring and sample construction. This is relevant to the preparation of computational research instruments, but it is not a direct evaluation of sentiment, political stance, responsibility attribution, or any other social-science variable.')])
replace('We evaluate three precision configurations', 'We evaluate three precision configurations of EuroLLM-1.7B (Martins et al., 2024) on the same Turkish minimal pairs from TurBLiMP (Başar et al., 2025). Three empirical questions organize the audit. RQ1 asks how 8-bit and 4-bit quantization change benchmark accuracy and individual decisions relative to FP16. RQ2 asks whether the conclusions are stable across grammatical categories, sentence-scoring conventions, and sensitivity analyses for related or disputed pairs. RQ3 asks what accuracy–memory trade-off is observed under the measured local workload. An exploratory analysis examines associations between subword fragmentation and Q4 reversals. These questions delimit the evidence that this experiment can contribute to evaluating a linguistic instrument.')
insert_block(find('2. Related Work'),[('Normal','The methodological contribution is a reporting procedure that keeps numerical stability, benchmark agreement, and application validity separate. This distinction prevents a small aggregate accuracy difference from being interpreted as evidence that a compressed model can replace its reference in any Turkish research task. The discussion translates the audit into requirements for a subsequent study of actual research coding; it does not present that subsequent study as completed.')])
insert_block(find('3. Theoretical Framework'),[('Heading 2','2.4. Validation in computational text research'),('Normal','Adcock and Collier (2001) distinguish the conceptual and observational levels involved in measurement and emphasize the contextual specificity of validity claims. This provides a useful starting point for assessing language-model outputs: the meaning of a target concept must be specified before a score can be interpreted as its indicator. For this study, the operational target is preference for the benchmark-designated grammatical alternative, not general communicative competence or comprehension of a social claim.'),('Normal','Gilardi et al. (2023) evaluate ChatGPT on annotation tasks involving tweets and news articles, including relevance, stance, topics, and frames. Their study illustrates an application-level evaluation in which model labels are assessed against human annotations. The present task uses sentence likelihoods from a smaller base model and a grammatical benchmark. Findings from either setting cannot establish validity in the other without additional evidence. A successful linguistic audit is therefore one source of diagnostic information, rather than a substitute for a validated coding scheme.')])
replace('3. Theoretical Framework','3. Measurement Framework')
insert_block(find('4. Data and Task'),[('Heading 2','3.3. Levels of evidence and permissible inference'),('Normal','We organize the audit around three distinct relationships. First, configuration stability concerns whether changing weight precision changes the decision for a fixed pair. Second, benchmark agreement concerns whether that decision matches the supplied label. Third, application validity concerns whether an output measures the construct of interest in the intended corpus. The first relationship can be evaluated without assuming that every benchmark label is correct. The second depends on label quality. The third requires evidence that is absent from the present grammaticality experiment.'),('Normal','This distinction matters because identical errors can produce perfect stability. Conversely, a reversal can improve agreement with the benchmark. We therefore report both directions of change and retain the underlying continuous margins. A model that reproduces the reference is not necessarily a valid instrument, and a model that differs from it is not necessarily worse on every item. These statements follow from the definitions of stability and accuracy; they do not assume that FP16 provides ground-truth language judgments.'),('Normal','The audit also distinguishes the measurement rule from the computational configuration. The rule specifies how token probabilities become a sentence score and how two scores become a binary decision. The configuration specifies the model weights, tokenizer, numerical representation, and runtime. Holding one fixed while varying the other makes their contributions observable. When both scoring conventions and weight precision change, their effects should not be collapsed into an unspecified change in model quality.'),('Normal','A possible application is the coding of who performs an action in a news sentence. Person agreement and argument structure may be relevant to interpreting that sentence, but grammatical preference does not itself identify an actor or establish responsibility attribution. Connecting the diagnostic to that application would require an explicit actor-coding definition, a representative corpus, independent human judgments, and direct evaluation of the resulting codes. This example motivates a validation requirement; no actor-coding outcomes are inferred from the benchmark.')])
# Add an explicit, reproducible audit procedure without claiming an unperformed new study.
insert_block(find('6. Experiments'),[('Heading 2','5.6. Audit procedure and interpretation rules'),('Normal','The audit treats the pair identifier as the join key across precision settings and verifies that the same token sequences were scored. It retains continuous scores, binary decisions, and benchmark labels so that a reader can separate reference agreement from label agreement. Overall accuracy is accompanied by transition counts, category-specific results, and normalization sensitivity. The resource measurements are attached to the exact scoring workload rather than generalized to all deployments.'),('Normal','Interpretation follows the available evidence. A nonsignificant superiority test is not treated as proof of equivalence. A category-specific loss is not treated as evidence of harm to a demographic group. A memory reduction is not treated as evidence of improved access, lower energy consumption, or lower institutional expenditure. These restrictions define the use of the audit in research planning: it identifies decisions requiring further validation and records the limits of the evidence rather than certifying a model for an untested application.')])
replace('8. Discussion','8. Discussion')
insert_block(find('The Q4 configuration reduced'),[('Heading 2','8.1. Stability of the linguistic instrument')])
anchor=find('9. Limitations')
insert_block(anchor,[('Heading 2','8.2. Implications for computational social research'),('Normal','The results expose a practical problem in instrument substitution. A researcher comparing only overall accuracy might regard the observed Q4 decline as a uniform small loss. The transition table and category breakdown show a more selective pattern: some decisions improve, others deteriorate, and subject agreement contributes most of the net loss. If a future coding task relies on particular constructions, its relevant error distribution may differ from the equally weighted benchmark average. This is a reason to validate on the intended task, not evidence that the same pattern will necessarily occur there.'),('Normal','The normalization analysis also locates part of the measurement uncertainty outside the weight representation. On the same sentences, mean and summed log probability yield different accuracy levels and different estimates of the Q4 deficit. Neither normalization can be selected solely because it gives a favorable comparison. A research workflow should specify its scoring rule before the main evaluation, retain an interpretable alternative as a sensitivity check, and explain what construct the resulting score is intended to measure. Here, both rules identify a negative Q4 contrast, while disagreeing about its size.'),('Normal','Benchmark labels introduce a further source of uncertainty. The nominalization examples show why a change toward the preferred label cannot automatically be called a linguistic improvement. Context-dependent alternatives require judgments about intended readings. Reviewing only changed pairs is useful for diagnosing reversals but cannot estimate the prevalence of questionable labels in the full sample. A future validation should therefore combine a probability sample from all categories with a separately identified diagnostic sample of reversals. Those samples answer different questions and should not be pooled without accounting for their selection.'),('Heading 2','8.3. Resource requirements and the limits of access claims'),('Normal','The measured reduction in allocator memory identifies a computational opportunity under one workload. It does not establish that a particular institution or user can adopt the model, because adoption also depends on available equipment, task requirements, maintenance, expertise, and other constraints not observed here. Likewise, the experiment does not estimate a monetary benefit or an environmental saving. The appropriate conclusion is narrower: a configuration with substantially lower measured allocator memory produced less reliable agreement with the benchmark, and the distribution of that loss deserves scrutiny before application.'),('Normal','The Q8 result offers a useful comparison but not a deployment recommendation. Its accuracy is close to FP16 in this sample and its weight file and allocator memory are smaller. An equivalence or noninferiority claim would require an application-specific acceptable loss and a suitable design. Choosing such a tolerance after observing the present result would change the inferential question. Researchers can use the reported intervals to plan a future evaluation, but cannot retrospectively convert the superiority test into a prespecified equivalence test.'),('Heading 2','8.4. A validation workflow for subsequent applications'),('Normal','The proposed workflow begins by defining the social-science construct and its coding unit. Researchers then select texts from the intended setting, develop a coding manual, and obtain independent human labels before comparing model configurations. The development material must be separated from the final evaluation, including any prompt selection or adaptation required for a base model. A precision comparison should keep those decisions fixed. At the final stage, researchers should examine not only individual coding errors but also changes in the aggregate quantity they intend to report, such as the proportion of texts assigned to an actor category.'),('Normal','This workflow connects the current diagnostic to application validity without assuming a numerical transfer from one to the other. For example, a 2.8-percentage-point decline in minimal-pair accuracy does not imply a 2.8-point bias in an estimated responsibility-attribution rate. The relationship depends on the new labels, corpus, and coding procedure. The present contribution is the documented linguistic audit and its interpretation rules; direct validation of a social-science coding task remains an empirical next step.')])
# Strengthen limitation and conclusion; retain original numerical findings.
insert_block(find('10. Conclusion'),[('Normal','No social-science corpus was coded and no downstream research estimate was compared across precision settings. The methodological relevance of the audit therefore rests on its role in evaluating a linguistic component before application. It cannot establish the reliability or validity of a completed social-science measurement instrument. The proposed application workflow is a research design recommendation, not a second experiment or an externally validated protocol.')])
insert_block(find('Research Transparency'),[('Normal','For computational research using Turkish texts, the methodological lesson is to document the measurement procedure alongside the model configuration and to evaluate both before substitution. The reported audit makes disagreement, benchmark uncertainty, and resource scope visible. Applying this procedure to an actual social-science coding task is necessary before drawing conclusions about the stability of substantive research findings.')])
# Replace generic transparency with separate journal-required headings.
replace('Research Transparency','11. Research Transparency')
replace('Data and code availability.', 'The original TurBLiMP benchmark is available at https://github.com/ezgibasar/TurBLiMP (Başar et al., 2025). The experiment uses revision 297de13fb7a0ce524fe32e8b175c6b5255d66960 and the sample digest reported in Section 4. A local replication package accompanies this manuscript preparation and includes the selected data, saved scores, analysis scripts, model configuration files, and an integrity inventory. It does not contain model weights. A persistent public repository link has not yet been assigned. The incomplete archive of model conversion commands limits reconstruction of the weight-conversion stage.')
before(find('The original TurBLiMP benchmark'),'11.1. Data and Code Availability','Heading 2')
replace('Use of artificial intelligence.', 'AI tools assisted with earlier manuscript preparation and preliminary linguistic review, as reported in the experimental records. OpenAI Codex was used during the revision of 6 November 2026 for literature checking, manuscript restructuring, document preparation, and consistency checks against saved results. The exact model identifier for this revision and the tool names and versions used in earlier stages are not documented in the supplied records. AI-assisted linguistic comments are exploratory and are not independent human annotations. The tested EuroLLM configurations are described in Sections 5 and 6. Responsibility for the submitted manuscript, source accuracy, and interpretation remains with the human author.')
before(find('AI tools assisted'),'11.2. AI Use Declaration','Heading 2')
insert_block(find('References'),[('Heading 2','11.3. Ethics and consent'),('Normal','The reported experiment uses existing benchmark sentences and model-generated likelihood scores. It did not recruit human participants, collect participant responses, or analyze identifiable personal records. No participant consent was collected and no ethics-committee decision is reported. The study is presented as a secondary benchmark analysis; any formal institutional exemption or approval requirement must be determined under the applicable institutional procedure before submission.')])
# Add sourced social-methodology references and normalize reference text to APA sentence case.
newrefs=[
'Adcock, R., & Collier, D. (2001). Measurement validity: A shared standard for qualitative and quantitative research. American Political Science Review, 95(3), 529–546. https://doi.org/10.1017/S0003055401003100',
'Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. Proceedings of the National Academy of Sciences, 120(30), Article e2305016120. https://doi.org/10.1073/pnas.2305016120',
'Grimmer, J., & Stewart, B. M. (2013). Text as data: The promise and pitfalls of automatic content analysis methods for political texts. Political Analysis, 21(3), 267–297. https://doi.org/10.1093/pan/mps028']
refs_idx=doc.paragraphs.index(find('References')) if False else next(i for i,p in enumerate(doc.paragraphs) if p.text=='References')
refs=[p.text for p in doc.paragraphs[refs_idx+1:] if p.text.strip()]+newrefs
for p in list(doc.paragraphs[refs_idx+1:]):remove(p)
fix={
'LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale':'LLM.int8(): 8-bit matrix multiplication for transformers at scale',
'Bootstrap Methods: Another Look at the Jackknife':'Bootstrap methods: Another look at the jackknife',
'A Simple Sequentially Rejective Multiple Test Procedure':'A simple sequentially rejective multiple test procedure',
'TurBLiMP: A Turkish Benchmark of Linguistic Minimal Pairs':'TurBLiMP: A Turkish benchmark of linguistic minimal pairs',
'A Morphology-Aware Evaluation of Turkish Syntax in Large Language Models':'A morphology-aware evaluation of Turkish syntax in large language models',
'GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers':'GPTQ: Accurate post-training quantization for generative pre-trained transformers',
'AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration':'AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration',
'How Does Quantization Affect Multilingual LLMs?':'How does quantization affect multilingual LLMs?',
'EuroLLM: Multilingual Language Models for Europe':'EuroLLM: Multilingual language models for Europe',
'BLiMP: The Benchmark of Linguistic Minimal Pairs for English':'BLiMP: The benchmark of linguistic minimal pairs for English',
'SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models':'SmoothQuant: Accurate and efficient post-training quantization for large language models',
'(n.d).':'(n.d.).',
'EuroLLM-1.7B Model Card. Hugging Face.':'EuroLLM-1.7B [Model card]. Hugging Face. Retrieved November 6, 2026, from',
'MLX 0.32.4 documentation.':'MLX documentation (Version 0.32.4). Retrieved November 6, 2026, from',
'https://arxiv.org/abs/2208.07339':'https://doi.org/10.48550/arXiv.2208.07339',
'https://arxiv.org/abs/2210.17323':'https://doi.org/10.48550/arXiv.2210.17323'}
for t in sorted(refs,key=lambda x:x.casefold().replace('ş','s').replace('ı','i')):
 for a,b in fix.items():t=t.replace(a,b)
 doc.add_paragraph(t)
# Formatting and anonymization.
def format_doc(d,main=False):
 for s in d.sections:
  s.page_width=Cm(21);s.page_height=Cm(29.7)
  s.top_margin=s.bottom_margin=s.left_margin=s.right_margin=Cm(2.5)
  for part in [s.header,s.footer]:
   for p in part.paragraphs:p.clear()
 for name in ['Normal','Title','Heading 1','Heading 2','Heading 3','Caption']:
  
  if name not in d.styles:d.styles.add_style(name,1)
  st=d.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
  st.font.size=Pt(11 if name in ['Normal','Caption'] else 13 if name=='Heading 1' else 12 if name.startswith('Heading') else 16)
  pf=st.paragraph_format;pf.line_spacing=1.5;pf.space_after=Pt(6);pf.space_before=Pt(0)
  if name.startswith('Heading'):pf.keep_with_next=True;pf.space_before=Pt(10);st.font.bold=True
  if name=='Title':st.font.bold=True;pf.space_after=Pt(12)
  if name=='Normal':pf.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
  if name=='Caption':pf.line_spacing=1.15;pf.keep_with_next=True
 refs=False
 for p in d.paragraphs:
  if p.text=='References':refs=True
  p.paragraph_format.widow_control=True
  p.paragraph_format.page_break_before=False
  for r in p.runs:
   r.font.name='Times New Roman';r.font.color.rgb=RGBColor(0,0,0);r.font.size=None
  if p.style.name=='Title':p.alignment=WD_ALIGN_PARAGRAPH.CENTER
  elif p.style.name=='Normal':
   p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY;p.paragraph_format.line_spacing=1.5
   p.paragraph_format.keep_with_next=False
  elif p.style.name.startswith('Heading'):p.alignment=WD_ALIGN_PARAGRAPH.LEFT
  if refs and p.text!='References':
   p.alignment=WD_ALIGN_PARAGRAPH.LEFT;p.paragraph_format.left_indent=Cm(1.27);p.paragraph_format.first_line_indent=Cm(-1.27)
   p.paragraph_format.keep_together=True
  if p._p.xpath('.//w:drawing'):
   p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
 for tab in d.tables:
  tab.autofit=False
  n=len(tab.columns)
  widths={4:[5.8,3.4,3.4,3.4],5:[6,2.5,2.5,2.5,2.5],6:[4.3,2.4,2.3,2.3,2.5,2.2]}.get(n,[16/n]*n)
  # The primary statistics table needs more width for its intervals.
  if n==6 and 'Model' in tab.cell(0,0).text:widths=[1.6,2.0,4.1,1.4,4.7,2.2]
  for col,w in zip(tab.columns,widths):col.width=Cm(w)
  pr=tab._tbl.tblPr
  borders=OxmlElement('w:tblBorders')
  for edge in ['top','left','bottom','right','insideH','insideV']:
   el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
  old=pr.find(qn('w:tblBorders'))
  if old is not None:pr.remove(old)
  pr.append(borders)
  for ri,row in enumerate(tab.rows):
   trpr=row._tr.get_or_add_trPr();nosplit=OxmlElement('w:cantSplit');trpr.append(nosplit)
   if ri==0:trpr.append(OxmlElement('w:tblHeader'))
   for ci,cell in enumerate(row.cells):
    cell.width=Cm(widths[ci]);cell.vertical_alignment=1
    cp=cell._tc.get_or_add_tcPr();mar=OxmlElement('w:tcMar')
    for edge in ['top','left','bottom','right']:
     e=OxmlElement('w:'+edge);e.set(qn('w:w'),'75');e.set(qn('w:type'),'dxa');mar.append(e)
    cp.append(mar)
    if ri==0:
     sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EAEAEA');cp.append(sh)
    for p in cell.paragraphs:
     p.paragraph_format.line_spacing=1.1;p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2);p.paragraph_format.keep_with_next=False
     p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
     for r in p.runs:r.font.name='Times New Roman';r.font.size=Pt(10);r.font.bold=(ri==0);r.font.color.rgb=RGBColor(0,0,0)
 d.core_properties.author='' if main else 'Ömer Faruk Öztürk'
 d.core_properties.last_modified_by='';d.core_properties.comments='';d.core_properties.keywords=''
 d.core_properties.title=TITLE if main else d.paragraphs[0].text
 # Strip author-bearing revision identifiers and comments, if any.
 for el in d._element.iter():
  for k in list(el.attrib):
   if k.endswith('}author') or 'rsid' in k:del el.attrib[k]
format_doc(doc,True)
# Italicize APA journal/volume portions without altering reference text.
ref_started=False
journals=['American Political Science Review','Proceedings of the National Academy of Sciences','Political Analysis','Biometrika','Psychometrika','The Annals of Statistics','Scandinavian Journal of Statistics','Transactions of the Association for Computational Linguistics','Veri Bilimi']
for p in doc.paragraphs:
 if p.text=='References':ref_started=True;continue
 if ref_started:
  t=p.text
  for j in journals:
   m=re.search(re.escape(j)+r', \d+',t)
   if m:
    p.clear();p.add_run(t[:m.start()]);p.add_run(t[m.start():m.end()]).italic=True;p.add_run(t[m.end():]);break
# Extract figure assets without resampling; effective dpi > 300.
for i,shape in enumerate(doc.inline_shapes,1):
 part=doc.part.related_parts[shape._inline.graphic.graphicData.pic.blipFill.blip.embed]
 (OUT/'figures'/f'Figure_{i}.png').write_bytes(part.blob)
 p=shape._inline.getparent().getparent()
 im=Image.open(BytesIO(part.blob));assert im.width/(shape.width/914400)>=300
# Caption paragraphs from source sometimes split labels and source notes: keep them together, not the entire next section.
for i,p in enumerate(doc.paragraphs):
 if p.style.name=='Caption' or p.text.startswith(('Table ','Figure ','Source:')):
  p.paragraph_format.keep_with_next=True
# Figure source is after graphic: permit following body to flow.
for p in doc.paragraphs:
 if p.text.startswith('Source:'):p.paragraph_format.keep_with_next=False;p.paragraph_format.line_spacing=1.0;p.paragraph_format.space_after=Pt(8)
MAIN=OUT/'01_Weight_Quantization_and_Turkish_Linguistic_Measurement_Anonymized.docx'
doc.save(MAIN)
# Title page: unknown declarations visibly await author input rather than being fabricated.
t=Document();t.add_paragraph(TITLE,'Title');t.add_paragraph('Title page','Heading 1')
for label,value in [('Author','Ömer Faruk Öztürk'),('Affiliation','Master’s student, Artificial Intelligence and Data Engineering, Ankara University, Ankara, Türkiye'),('Corresponding author','Ömer Faruk Öztürk'),('Email','omerfaruko@ankara.edu.tr'),('ORCID','[Author to provide ORCID]'),('Proposed article type','Methods and Data Article')]:
 p=t.add_paragraph();p.add_run(label+': ').bold=True;p.add_run(value)
t.add_paragraph('Author contributions','Heading 1')
t.add_paragraph('Proposed CRediT statement for author confirmation: Ömer Faruk Öztürk — Conceptualization, Methodology, Investigation, Formal analysis, Data curation, Writing – original draft, Writing – review and editing. Confirm each role and add Software, Validation, or Visualization only where personally applicable.')
t.add_paragraph('Funding','Heading 1');t.add_paragraph('[Author to state funding source and grant number, or confirm that no specific funding was received.]')
t.add_paragraph('Conflict of interest','Heading 1');t.add_paragraph('[Author to disclose relevant interests, or confirm that there are no conflicts of interest.]')
t.add_paragraph('Acknowledgements','Heading 1');t.add_paragraph('[Author to identify any contributors who do not meet authorship criteria, or remove this section.]')
format_doc(t);t.save(OUT/'02_Title_Page_and_Author_Declarations.docx')
# Local verification: no invented results; compare tables and equations before/after.
original=Document(SRC)
assert [[ [c.text for c in r.cells] for r in tab.rows] for tab in original.tables]==[[ [c.text for c in r.cells] for r in tab.rows] for tab in doc.tables]
from lxml import etree
assert [etree.tostring(x) for x in original._element.xpath('.//m:oMath')]==[etree.tostring(x) for x in doc._element.xpath('.//m:oMath')]
body=[];active=False
for p in doc.paragraphs:
 if p.text.startswith('1. Introduction'):active=True
 if p.text.startswith('11. Research Transparency'):active=False
 if active and p.style.name!='Caption' and not p.text.startswith(('Table ','Figure ','Source:')):body.append(p.text)
wc=len(' '.join(body).split());abstract=len(find('Computational research on Turkish').text.split())
assert 150<=abstract<=250,(abstract)
assert 4000<=wc<=7000,wc
assert not any('Ömer' in p.text or 'omerfaruko@' in p.text for p in doc.paragraphs)
(OUT/'work/changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
(OUT/'work/document_checks.json').write_text(json.dumps({'body_words_excluding_abstract_references_captions_tables':wc,'abstract_words':abstract,'references':len(refs),'tables_preserved':len(doc.tables),'equations_preserved':len(doc._element.xpath('.//m:oMath')),'figures':len(doc.inline_shapes)},indent=2))
print('Created main manuscript and title page',wc,'body words;',abstract,'abstract words;',len(refs),'references')
