"""Validate complete new GPU runs using original validators and compare every record."""
import sys,json,math,contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import score_main as scoring
import summarize_main as summary
import analyze_main as analysis
from scipy.stats import binomtest
summary.OUT=OUT/'rerun'
with (OUT/'rerun_validation.log').open('w') as log,contextlib.redirect_stdout(log):summary.main()
analysis.RESULTS=OUT/'rerun'
source,ids,cats,scores,metrics,data_sha=analysis.load_data()
report={'dataset_sha256':data_sha,'models':{}}
for name in ['fp16','q8','q4']:
 old=[row for i in range(20) for row in scoring.read_jsonl(ROOT/'results/main'/name/f'chunk_{i:02d}.jsonl')]
 new=[scores[name][p] for p in ids]
 for a,b in zip(old,new):
  assert a['pair_id']==b['pair_id']
  for side in ['good','bad']:
   assert a[side]['token_ids_sha256']==b[side]['token_ids_sha256']
   assert a[side]['n_tokens']==b[side]['n_tokens']
  assert b['total_logp_diff']==b['good']['total_logp']-b['bad']['total_logp']
 for met in metrics[name]:
  assert met['model_config_sha256']==scoring.sha256(ROOT/'models'/scoring.MODELS[name]/'config.json')
 report['models'][name]=dict(n=len(new),correct=sum(r['correct'] for r in new),decision_changes=sum(a['correct']!=b['correct'] for a,b in zip(old,new)),identical_records=sum(a==b for a,b in zip(old,new)),max_abs_margin_difference=max(abs(a['mean_logp_diff']-b['mean_logp_diff']) for a,b in zip(old,new)),max_abs_total_difference=max(abs(a['total_logp_diff']-b['total_logp_diff']) for a,b in zip(old,new)),score_seconds=sum(m['score_seconds_chunk'] for m in metrics[name]),mlx_peak_gb=max(m['mlx_allocator_peak_gb_decimal_chunk'] for m in metrics[name]),versions={k:sorted(set(str(m[k]) for m in metrics[name])) for k in ['python_version','mlx_version','mlx_lm_version']},timestamp_min=min(m['timestamp_utc'] for m in metrics[name]),timestamp_max=max(m['timestamp_utc'] for m in metrics[name]))
report['comparisons']={}
for n in ['q8','q4']:
 c=analysis.comparison(ids,scores,n);c.pop('down_ids');c.pop('up_ids');report['comparisons'][n]=c
adj=analysis.holm_adjust({n:v['mcnemar_exact_p_unadjusted'] for n,v in report['comparisons'].items()})
for n,v in report['comparisons'].items():v['p_holm']=adj[n]
(OUT/'rerun_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
