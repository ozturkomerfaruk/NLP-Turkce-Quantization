"""Check final artifacts and record provenance without changing original files."""
import json,csv,re,sys,hashlib,subprocess,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from score_main import sha256
from scipy.stats import binomtest
R=json.loads((OUT/'extended_statistics.json').read_text())
S=json.loads((ROOT/'results/analysis/main_statistics.json').read_text())
repeat=json.loads((OUT/'rerun_comparison.json').read_text())
assert all(v['n']==1000 and v['decision_changes']==0 for v in repeat['models'].values())
assert sum(R['patterns'].values())==1000
for n in ['q8','q4']:
 v=R['primary'][n];old=S['primary_comparisons'][n]
 assert abs(v['p']-binomtest(min(v['down'],v['up']),v['down']+v['up']).pvalue)<1e-14
 assert abs(v['p_holm']-old['mcnemar_p_holm'])<1e-14
 assert abs(v['delta_pp']-100*old['delta_accuracy'])<1e-14
 assert max(abs(x-100*y) for x,y in zip(v['ci_pp'],old['paired_ci95_conservative']))<1e-8
for name,count in [('manual_review_20_annotated.csv',20),('linguistic_review_58.csv',58),('predicate_screen_all_pairs.csv',17),('pair_audit.csv',1000),('template_clusters.csv',R['cluster_sensitivity']['n_clusters'])]:
 rows=list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
 assert len(rows)==count,(name,len(rows))
 if 'review' in name:assert all(r['manual_review_note'] and r['linguistic_validity'] for r in rows)
assert len(list((OUT/'figures').glob('*.png')))==6
assert len(list((OUT/'figures').glob('*.svg')))==6
# Freeze original and research artifact identities, including actual weight files.
files=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+list((ROOT/'data').rglob('*'))+list((ROOT/'results/main').rglob('*'))+list((ROOT/'results/analysis').rglob('*'))+list((ROOT/'results/pilot').rglob('*'))
files+=list((OUT).rglob('*'))+[ROOT/'ProjeMarkdown/ISTATISTIKSEL_ANALIZ_VE_DENEY_RAPORU.md']
for n in ['fp16','q8','q4']:
 files+=list((ROOT/'models'/('eurollm-'+n)).glob('*.json'))+list((ROOT/'models'/('eurollm-'+n)).glob('*.safetensors'))
files=sorted(set(p for p in files if p.is_file() and '__pycache__' not in p.parts and p.name not in ['provenance_sha256.json','final_audit.log']))
provenance={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in files}
(OUT/'provenance_sha256.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
report=(ROOT/'ProjeMarkdown/ISTATISTIKSEL_ANALIZ_VE_DENEY_RAPORU.md').read_text()
assert 'henüz rapora bağlanmadı' not in report
for target in re.findall(r'\]\((/[^)]+)\)',report):assert Path(target).exists(),target
print('PASS: all numeric cross-checks, repeated model decisions, review row counts, 12 figures and report links')
print('SHA256 entries:',len(provenance));print('Report words:',len(report.split()))
