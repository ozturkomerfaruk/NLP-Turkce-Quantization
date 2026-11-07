from pathlib import Path
from docx import Document
import re,json,shutil,hashlib
OUT=Path('outputs/socio_submission');path=OUT/'01_Weight_Quantization_and_Turkish_Linguistic_Measurement_Anonymized.docx';d=Document(path)
for p in d.paragraphs:
 t=p.text
 if t.startswith('Başar, E., & Bisazza'):
  t=t.replace('Proceedings of SIGTURK 2026 (pp. 95–102).','In K. Oflazer, A. Köksal, & O. Varol (Eds.), Proceedings of the Second Workshop Natural Language Processing for Turkic Languages (SIGTURK 2026) (pp. 95–102). Association for Computational Linguistics.')
 elif t.startswith('Başar, E., Padovani'):
  t=t.replace('Proceedings of EMNLP 2025 (pp. 16495–16510).','In C. Christodoulopoulos, T. Chakraborty, C. Rose, & V. Peng (Eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (pp. 16495–16510). Association for Computational Linguistics.')
 elif t.startswith('Marchisio, K.'):
  t=t.replace('Findings of the Association for Computational Linguistics: EMNLP 2024 (pp. 15928–15947).','In Y. Al-Onaizan, M. Bansal, & Y.-N. Chen (Eds.), Findings of the Association for Computational Linguistics: EMNLP 2024 (pp. 15928–15947). Association for Computational Linguistics.')
 elif t.startswith('Frantar, E.'):
  t=t.replace('GPTQ: Accurate post-training quantization for generative pre-trained transformers. International Conference on Learning Representations. https://doi.org/10.48550/arXiv.2210.17323','OPTQ: Accurate quantization for generative pre-trained transformers. The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=tcbBPnfwxS')
 elif t.startswith('Dettmers, T.'):
  t=t.replace('https://doi.org/10.48550/arXiv.2208.07339','https://doi.org/10.52202/068431-2198')
 elif t.startswith('Bayram, C.'):
  t=t.replace('Nicemleme Dengelemesi: Doğruluk, Hız ve Bellek Kullanımı Üzerine Bir Değerlendirme','nicemleme dengelemesi: Doğruluk, hız ve bellek kullanımı üzerine bir değerlendirme')
 if t!=p.text:p.text=t
# Italicize publication titles (and journal volume where applicable).
ref=False
journals=['American Political Science Review','Proceedings of the National Academy of Sciences','Political Analysis','Biometrika','Psychometrika','The Annals of Statistics','Scandinavian Journal of Statistics','Transactions of the Association for Computational Linguistics','Veri Bilimi','Advances in Neural Information Processing Systems','Proceedings of Machine Learning and Systems','Proceedings of Machine Learning Research']
venues=['Proceedings of the Second Workshop Natural Language Processing for Turkic Languages (SIGTURK 2026)','Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing','Findings of the Association for Computational Linguistics: EMNLP 2024','The Eleventh International Conference on Learning Representations']
for p in d.paragraphs:
 if p.text=='References':ref=True;continue
 if not ref:continue
 t=p.text;span=None
 for j in journals:
  m=re.search(re.escape(j)+r', \d+',t)
  if m:span=(m.start(),m.end());break
 if span is None:
  for j in venues:
   if j in t:span=(t.index(j),t.index(j)+len(j));break
 if span:
  a,b=span;p.clear();p.add_run(t[:a]);p.add_run(t[a:b]).italic=True;p.add_run(t[b:])
d.save(path)
# Restore archived original result outputs after the successful reanalysis check.
root=Path('.');pack=OUT/'replication_package'
a=json.loads((root/'results/analysis/main_statistics.json').read_text());b=json.loads((pack/'results/analysis/main_statistics.json').read_text())
for model in ['fp16','q8','q4']:
 for k in ['safetensors_bytes','safetensors_gb_decimal']:b['resource_summary'][model][k]=a['resource_summary'][model][k]
assert a==b
for p in (root/'results/analysis').glob('*'):
 if p.is_file():shutil.copy2(p,pack/'results/analysis'/p.name)
p=pack/'README.md';s=p.read_text();s=s.replace('It does not rerun the model.','It does not rerun the model. Without the weight files, this command reports null for the weight-file size fields; the other regenerated main-statistics values were checked against the original and matched. The archive retains the original measured size fields in the saved outputs.')
p.write_text(s)
# Remove transient interpreter cache from the package only.
for p in pack.rglob('__pycache__'):shutil.rmtree(p)
inventory={str(p.relative_to(pack)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(pack.rglob('*')) if p.is_file() and p.name!='PACKAGE_SHA256.json'}
(pack/'PACKAGE_SHA256.json').write_text(json.dumps(inventory,indent=2))
print('Final reference metadata and replication archive checks completed.')
