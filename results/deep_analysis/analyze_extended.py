"""Reproducible research audit; never mutates original scoring code/data/results.
Run from project root: .venv/bin/python results/deep_analysis/analyze_extended.py
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/turkce-quantization-mpl')
import sys, json, csv, math, struct, hashlib, platform, contextlib, io
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import scipy
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_main as old
import prepare_dataset as prep
from score_main import sha256, read_jsonl, token_hash
NAMES=['fp16','q8','q4']; SEED=20261028; B=20000
OUT.mkdir(exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)

def dump(name,data):
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
 if not rows:return
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cp(k,n,alpha=.05):
 return [float(stats.beta.ppf(alpha/2,k,n-k+1)) if k else 0.,float(stats.beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.]
def paired(a,b):
 d=int(np.sum(a & ~b)); u=int(np.sum(~a & b)); n=len(a)
 dl,dh=cp(d,n,.025);ul,uh=cp(u,n,.025)
 return dict(n=n,reference_correct=int(a.sum()),other_correct=int(b.sum()),both_correct=int((a&b).sum()),both_wrong=int((~a&~b).sum()),down=d,up=u,delta_pp=100*(u-d)/n,ci_pp=[100*(ul-dh),100*(uh-dl)],p=float(stats.binomtest(min(d,u),d+u,.5).pvalue) if d+u else 1.)
def desc(x):
 x=np.asarray(x,float)
 return dict(n=len(x),mean=float(x.mean()),sd=float(x.std(ddof=1)) if len(x)>1 else 0.,min=float(x.min()),q25=float(np.quantile(x,.25)),median=float(np.median(x)),q75=float(np.quantile(x,.75)),max=float(x.max()))
def holm(results):
 adjusted=old.holm_adjust({k:v['p'] for k,v in results.items()})
 for k,v in results.items(): v['p_holm']=adjusted[k]
 return results
source,ids,cats,scores,metrics,data_sha=old.load_data()
source_by={r['pair_id']:r for r in source}
C=np.array([[scores[n][p]['correct'] for n in NAMES] for p in ids],bool)
M=np.array([[scores[n][p]['mean_logp_diff'] for n in NAMES] for p in ids])
T=np.array([[scores[n][p]['total_logp_diff'] for n in NAMES] for p in ids])
L=np.array([[scores['fp16'][p][s]['n_tokens'] for s in ['good','bad']] for p in ids])
cat_array=np.array([r['category'] for r in source]); groups=[np.where(cat_array==c)[0] for c in cats]
manifest=json.loads((ROOT/'data/processed/manifest.json').read_text())
# Independently reconstruct selection without writing dataset or fetching anything.
import random
rng=random.Random(42);seen=set();selected=[];raw_stats={}
for cat,filename in prep.CATEGORIES.items():
 path=ROOT/'data/raw'/filename
 assert sha256(path)==manifest['raw_file_hashes'][filename]
 pool,st=prep.read_unique_pairs(path,cat,seen); selected+=rng.sample(pool,250);raw_stats[cat]=st
assert selected==source
# Verify all six thousand saved sentence records against tokenizer and numeric identities.
from transformers import AutoTokenizer
checks=0;max_mean_residual=0.
for name in NAMES:
 modeldir=ROOT/'models'/('eurollm-'+name)
 tok=AutoTokenizer.from_pretrained(str(modeldir),local_files_only=True)
 for p in ids:
  row=scores[name][p]
  for side in ['good','bad']:
   v=row[side];tokens=tok.encode(source_by[p]['sentence_'+side],add_special_tokens=False)
   assert token_hash(tokens)==v['token_ids_sha256'] and len(tokens)==v['n_tokens']
   assert math.isfinite(v['total_logp']) and math.isfinite(v['mean_logp'])
   residual=abs(v['mean_logp']-v['total_logp']/v['n_tokens'])
   max_mean_residual=max(max_mean_residual,residual); assert residual<1e-6
   checks+=1
  assert row['total_logp_diff']==row['good']['total_logp']-row['bad']['total_logp']
 for idx,met in enumerate(metrics[name]):
  assert met['model_config_sha256']==sha256(modeldir/'config.json')
  assert met['pairs_in_chunk']==50
  assert met['scored_tokens_chunk']==sum(scores[name][p][s]['n_tokens'] for p in ids[idx*50:(idx+1)*50] for s in ['good','bad'])
  assert met['score_seconds_chunk']>0
# Independent numerical verification of original exact intervals and p-values.
max_cp_error=0.
for n in [20,250,1000]:
 for k in sorted(set([0,1,2,3,14,42,n//2,n-1,n])):
  if k>n:continue
  for alpha in [.05,.025]:
   max_cp_error=max(max_cp_error,max(abs(np.array(cp(k,n,alpha))-old.clopper_pearson(k,n,alpha))))
assert max_cp_error<1e-9
for d,u in [(0,0),(0,3),(42,14),(1,1),(10,25),(250,250)]:
 expected=stats.binomtest(min(d,u),d+u,.5).pvalue if d+u else 1
 assert abs(old.mcnemar_exact(d,u)-expected)<1e-12
old.OUTPUT=OUT/'recomputed';old.OUTPUT.mkdir(exist_ok=True)
with (OUT/'recomputed/run.log').open('w') as log, contextlib.redirect_stdout(log):old.main()
original=json.loads((ROOT/'results/analysis/main_statistics.json').read_text())
recomputed=json.loads((OUT/'recomputed/main_statistics.json').read_text())
assert original==recomputed
R=dict(seed=SEED,bootstrap_replicates=B,dataset_sha256=data_sha,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),audit=dict(raw_stats=raw_stats,exact_dataset_reconstruction=True,sentence_records_verified=checks,max_mean_rounding_residual=max_mean_residual,max_cp_error_vs_scipy=max_cp_error,original_statistics_exact_match=True,exact_sentence_pair_duplicates=1000-len(set((r['sentence_good'],r['sentence_bad']) for r in source)),unique_good_sentences=len(set(r['sentence_good'] for r in source)),unique_bad_sentences=len(set(r['sentence_bad'] for r in source))))
R['accuracy']={n:dict(correct=int(C[:,j].sum()),accuracy_percent=float(C[:,j].mean()*100),ci95_percent=[v*100 for v in cp(int(C[:,j].sum()),1000)],ties=int((M[:,j]==0).sum())) for j,n in enumerate(NAMES)}
R['primary']=holm({n:paired(C[:,0],C[:,j]) for j,n in enumerate(NAMES) if j})
R['all_pairwise_exploratory']=holm({NAMES[a]+'_to_'+NAMES[b]:paired(C[:,a],C[:,b]) for a,b in [(0,1),(0,2),(1,2)]})
R['category']={c:dict(n=250,accuracy={n:float(C[g,j].mean()*100) for j,n in enumerate(NAMES)},comparisons={n:paired(C[g,0],C[g,j]) for j,n in enumerate(NAMES) if j}) for c,g in zip(cats,groups)}
cat_tests={c+'__'+n:R['category'][c]['comparisons'][n] for c in cats for n in ['q8','q4']};holm(cat_tests)
R['patterns']={''.join(map(str,map(int,k))):v for k,v in Counter(map(tuple,C)).items()}
R['total_scoring']=dict(accuracy={n:float((T[:,j]>0).mean()*100) for j,n in enumerate(NAMES)},disagreements={n:int(((T[:,j]>0)!=C[:,j]).sum()) for j,n in enumerate(NAMES)},comparisons=holm({n:paired(T[:,0]>0,T[:,j]>0) for j,n in enumerate(NAMES) if j}))
R['length']=dict(good=desc(L[:,0]),bad=desc(L[:,1]),good_minus_bad=desc(L[:,0]-L[:,1]),equal=int((L[:,0]==L[:,1]).sum()),groups={})
for label,mask in [('equal',L[:,0]==L[:,1]),('good_longer',L[:,0]>L[:,1]),('good_shorter',L[:,0]<L[:,1])]:
 R['length']['groups'][label]=dict(n=int(mask.sum()),accuracy={n:float(C[mask,j].mean()*100) for j,n in enumerate(NAMES)},mean_total_disagreements={n:int(((T[mask,j]>0)!=C[mask,j]).sum()) for j,n in enumerate(NAMES)},comparisons={n:paired(C[mask,0],C[mask,j]) for j,n in enumerate(NAMES) if j})
R['margins']={}
for j,n in enumerate(NAMES):
 e=M[:,j]-M[:,0]
 R['margins'][n]=dict(margin=desc(M[:,j]),change=desc(e),mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),pearson=float(stats.pearsonr(M[:,0],M[:,j]).statistic),spearman=float(stats.spearmanr(M[:,0],M[:,j]).statistic),abs_change_quantiles=np.quantile(abs(e),[.5,.9,.95,.99]).tolist())
R['margin_quartiles']={}
order=np.argsort(abs(M[:,0]),kind='stable')
for j,n in [(1,'q8'),(2,'q4')]:
 R['margin_quartiles'][n]=[dict(quartile=i+1,n=250,flips=int((C[g,0]!=C[g,j]).sum()),min_abs_margin=float(abs(M[g,0]).min()),max_abs_margin=float(abs(M[g,0]).max())) for i,g in enumerate(np.array_split(order,4))]
R['tolerance_sensitivity']={str(t):dict(correct={n:int((M[:,j]>t).sum()) for j,n in enumerate(NAMES)},near_zero={n:int((abs(M[:,j])<=t).sum()) for j,n in enumerate(NAMES)}) for t in [0,1e-6,1e-4,.001,.01]}
# Paired, category-stratified resampling. Never resample models independently.
rng=np.random.default_rng(SEED)
boot=np.zeros((B,3))
for g in groups:
 for start in range(0,B,500):
  ix=rng.choice(g,size=(min(500,B-start),len(g)),replace=True)
  boot[start:start+len(ix)]+=C[ix].mean(axis=1)/4
R['stratified_bootstrap']={n:dict(accuracy_ci_percent=(np.quantile(boot[:,j],[.025,.975])*100).tolist(),delta_ci_pp=(np.quantile(boot[:,j]-boot[:,0],[.025,.975])*100).tolist()) for j,n in enumerate(NAMES)}
# Exploratory family proxy: strip only shared leading discourse/time adverbs, then exact string identity.
PREFIXES=set('ama ve evet çünkü fakat yani ancak işte şimdi yalnız ayrıca sonra ardından artık bugün burada aslında belki öncelikle yine özellikle hadi sadece hemen'.split())
def folded(text):
 return text.casefold().replace('\u0307','')
def clusterkey(r):
 a=r['sentence_good'].split();b=r['sentence_bad'].split()
 while len(a)>1 and len(b)>1 and folded(a[0])==folded(b[0]) and folded(a[0]) in PREFIXES:
  a=a[1:];b=b[1:]
 return (r['category'],folded(' '.join(a)),folded(' '.join(b)))
family=defaultdict(list)
for i,r in enumerate(source):family[clusterkey(r)].append(i)
cluster_rows=[]
for k,g in family.items():
 cluster_rows.append(dict(cluster_id=len(cluster_rows)+1,category=k[0],n=len(g),pair_ids=';'.join(ids[i] for i in g),normalized_good=k[1],normalized_bad=k[2],**{n+'_delta_pp':float((C[g,j].astype(float)-C[g,0]).mean()*100) for j,n in [(1,'q8'),(2,'q4')]}))
csvout('template_clusters.csv',cluster_rows)
rng=np.random.default_rng(SEED+1); cluster_boot=np.zeros((B,3));equal_family=np.zeros(3); percat={}
for c in cats:
 cg=[g for k,g in family.items() if k[0]==c]; sizes=np.array([len(g) for g in cg]);sums=np.array([C[g].sum(axis=0) for g in cg])
 equal_family+=(sums/sizes[:,None]).mean(axis=0)/4
 percat[c]=dict(clusters=len(cg),singleton_clusters=int((sizes==1).sum()),max_size=int(sizes.max()))
 for start in range(0,B,500):
  ix=rng.integers(0,len(cg),size=(min(500,B-start),len(cg)))
  cluster_boot[start:start+len(ix)]+=sums[ix].sum(axis=1)/sizes[ix].sum(axis=1)[:,None]/4
R['cluster_sensitivity']=dict(prefixes=sorted(PREFIXES),n_clusters=len(family),per_category=percat,equal_cluster_accuracy_percent=(equal_family*100).tolist(),equal_cluster_delta_pp=((equal_family-equal_family[0])*100).tolist(),pair_weighted_cluster_bootstrap={n:dict(delta_ci_pp=(np.quantile(cluster_boot[:,j]-cluster_boot[:,0],[.025,.975])*100).tolist()) for j,n in [(1,'q8'),(2,'q4')]})
# Keep original labels; separate heuristic linguistic triage and exclusion scenarios.
selected20=list(csv.DictReader((ROOT/'results/analysis/manual_review_20.csv').open(encoding='utf-8-sig')))
reviewids={r['pair_id'] for r in selected20}
changed=np.any(C[:,1:]!=C[:,[0]],axis=1)
reviews=[]
for i,r in enumerate(source):
 if not changed[i]:continue
 cat=r['category'];critical=r['critical_region']; status='etiket_ayrimi_destekleniyor';note=''
 if cat=='subject_agreement':
  note=f"Özne değişimi ({r['good_cue']} → {r['bad_cue']}) ile yüklemin kişi/sayı biçimi ({critical}) arasındaki uyum hedefleniyor. Standart tek konuşucu okumasında iyi cümledeki uyum daha açık."
  if r['pair_id'] in ['subject_agreement:0417','subject_agreement:0020','subject_agreement:0258']:
   status='baglama_duyarli';note+=' Tekil ad/üçüncü kişi ile çoğul yüklem; saygı, topluluk ve söylem bağlamı bazı yargıları etkileyebilir. Bazı + tekil ad kullanımında ek doğallık sorunu da vardır.'
  if r['pair_id']=='subject_agreement:0128':note+=' İç cümledeki yaptıklarınızı ikinci çoğul kişiye, ana cümledeki sen tekil kişiye gönderme yapabilir; bu tek başına iyi etiketi geçersiz kılmaz, fakat söylem gerektirir.'
  if r['pair_id']=='subject_agreement:0068':note+=' Yanına aldığın iç cümlesi başka bir kişiye gönderme yapabilir; Işte yazımı ve karmaşık gönderimler ek gürültüdür.'
 elif cat=='argument_structure_transitive':
  note=f"{critical} yükleminin istediği durum eki, {r['good_cue']} → {r['bad_cue']} değişimiyle karşılaştırılıyor. Bu okuma altında etiket ayrımı destekleniyor."
  if r['pair_id']=='argument_structure_transitive:0938':note+=' 2 muhalefet partisi ve hala yazımı doğallığı etkileyebilir; hedef erken seçimde/erken seçimi karşıtlığıdır.'
 elif cat=='irregular_forms':note='Verir/verer karşıtlığında standart geniş zaman biçimi verir; verer standart biçim değildir.'
 else:
  if critical in ['şaşırdı','tepki gösterdi']:
   status='iki_cumle_de_kabul_edilebilir';note='-mA olay adlaştırması ile -DIK olgu bildiren adlaştırma, bu yüklemle ayrı anlam vurgularında kabul edilebilir görünüyor. Anlamsal fark tek başına dilbilgisel yanlışlık kanıtı değildir; uzman yargısı gerekir.'
  elif critical=='inanıyor':
   status='iki_cumle_de_kabul_edilebilir';note='Doğruyu söylediğine inanmak olguya inanç; doğruyu söylemesine inanmak bağlama göre davranışa/güvenilirliğe inanç şeklinde okunabilir. Kesin yanlış olarak etiketlemek için bağlam ve uzman değerlendirmesi gerekir.'
  elif critical=='destekledi':
   status='uzman_incelemesi_gerekli';note='Uygulanmasını desteklemek doğal bir olay/istek tamamlayıcısıdır. Uygulandığını desteklemek bir iddia/kanıt bağlamında yorumlanabilir ancak bu cümlede daha az doğal; kesin iki-geçerli hükmü verilmedi.'
  elif critical=='düşünmemiştim':
   status='uzman_incelemesi_gerekli';note='Olduğunu düşünmek olgusal, olmasını düşünmek tasarlama/istemeyi değerlendirme okumasına açık olabilir. Verilen bağlam ve Benim başlangıcında gönderim ayrıca incelenmelidir.'
  else:
   note=f"{critical} yüklemiyle {r['good_cue']} / {r['bad_cue']} tamamlayıcısının seçimi hedefleniyor. İyi biçim daha doğal; olgu/olay ayrımı bu tümcede etiket ayrımını destekliyor. Bu bir insan paneli kararı değildir."
 reviews.append(dict(pair_id=ids[i],category=cat,selected_in_original20=ids[i] in reviewids,sentence_good=r['sentence_good'],sentence_bad=r['sentence_bad'],critical_region=critical,fp16_correct=int(C[i,0]),q8_correct=int(C[i,1]),q4_correct=int(C[i,2]),fp16_margin=M[i,0],q8_margin=M[i,1],q4_margin=M[i,2],linguistic_validity=status,manual_review_note=note,reviewer='Codex; ön dilbilimsel değerlendirme; insan uzman onayı yok'))
csvout('linguistic_review_58.csv',reviews)
byreview={r['pair_id']:r for r in reviews}
review20=[{**r,'manual_review_note':byreview[r['pair_id']]['manual_review_note'],'linguistic_validity':byreview[r['pair_id']]['linguistic_validity'],'reviewer':'Codex ön değerlendirmesi'} for r in selected20]
csvout('manual_review_20_annotated.csv',review20)
R['review']=dict(n=len(reviews),status_counts=dict(Counter(r['linguistic_validity'] for r in reviews)),selected20_status_counts=dict(Counter(r['linguistic_validity'] for r in review20)))
pilotids={r['pair_id'] for r in read_jsonl(ROOT/'data/processed/turblimp_pilot_20.jsonl')}
scenarios={'exclude_pilot20':np.array([p not in pilotids for p in ids]),'exclude_reviewed_ambiguous':np.array([byreview.get(p,{}).get('linguistic_validity')!='iki_cumle_de_kabul_edilebilir' for p in ids]),'exclude_all_nominalization':cat_array!='nominalization'}
# Uniform lexical trigger applied to the entire sample; does not condition on a decision flip.
flag=np.array([r['category']=='nominalization' and r['critical_region'] in ['şaşırdı','tepki gösterdi','inanıyor'] for r in source])
scenarios['exclude_uniform_predicate_screen']=~flag
csvout('predicate_screen_all_pairs.csv',[dict(pair_id=ids[i],critical_region=source[i]['critical_region'],sentence_good=source[i]['sentence_good'],sentence_bad=source[i]['sentence_bad'],screen='yükleme dayalı duyarlılık bayrağı; kesin etiket düzeltmesi değildir') for i in np.where(flag)[0]])
for c in cats:scenarios['leave_out_'+c]=cat_array!=c
R['sensitivity']={label:dict(n=int(mask.sum()),accuracy={n:float(C[mask,j].mean()*100) for j,n in enumerate(NAMES)},comparisons=holm({n:paired(C[mask,0],C[mask,j]) for j,n in enumerate(NAMES) if j})) for label,mask in scenarios.items()}
# All possible reversals of just the screened labels, with fixed n and no exclusion.
R['label_flip_bounds']={n:dict(screened_pairs=int(flag.sum()),delta_min_pp=float(((C[:,j].astype(int)-C[:,0]).sum()-2*np.maximum(C[flag,j].astype(int)-C[flag,0],0).sum())/10),delta_max_pp=float(((C[:,j].astype(int)-C[:,0]).sum()-2*np.minimum(C[flag,j].astype(int)-C[flag,0],0).sum())/10)) for j,n in [(1,'q8'),(2,'q4')]}
R['resources']={}
for name in NAMES:
 mets=metrics[name];times=np.array([m['score_seconds_chunk'] for m in mets]); tokens=sum(m['scored_tokens_chunk'] for m in mets)
 sizes=sum(p.stat().st_size for p in (ROOT/'models'/('eurollm-'+name)).glob('*.safetensors'))
 R['resources'][name]=dict(seconds=float(times.sum()),chunk_seconds=desc(times),tokens=tokens,tokens_per_second=float(tokens/times.sum()),pairs_per_second=float(1000/times.sum()),weights_gb=sizes/1e9,mlx_peak_gb=max(m['mlx_allocator_peak_gb_decimal_chunk'] for m in mets),rss_peak_gb=max(m['process_peak_rss_gb_decimal_so_far'] for m in mets),timestamp_min=min(m['timestamp_utc'] for m in mets),timestamp_max=max(m['timestamp_utc'] for m in mets))
for name in ['q8','q4']:
 x=R['resources'][name];ref=R['resources']['fp16']
 x.update(time_saving_percent=100*(1-x['seconds']/ref['seconds']),speedup=ref['seconds']/x['seconds'],mlx_saving_percent=100*(1-x['mlx_peak_gb']/ref['mlx_peak_gb']),weight_saving_percent=100*(1-x['weights_gb']/ref['weights_gb']))
# Save every pair in a self-contained numeric table for audit and plotting.
rows=[]
for i,r in enumerate(source):
 rows.append(dict(pair_id=ids[i],category=r['category'],good_tokens=int(L[i,0]),bad_tokens=int(L[i,1]),**{n+'_correct':int(C[i,j]) for j,n in enumerate(NAMES)},**{n+'_margin':M[i,j] for j,n in enumerate(NAMES)},q4_margin_change=M[i,2]-M[i,0],q4_flip=bool(C[i,0]!=C[i,2]),screened_predicate=bool(flag[i])))
csvout('pair_audit.csv',rows)
# Publication-exportable PNG and SVG; plot uncertainty explicitly.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
colors=['#375a7f','#2e8b76','#c96b38']
def save(fig,name):
 fig.tight_layout();fig.savefig(OUT/'figures'/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'figures'/f'{name}.svg',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));acc=np.array([R['accuracy'][n]['accuracy_percent'] for n in NAMES]);ci=np.array([R['accuracy'][n]['ci95_percent'] for n in NAMES]).T
ax.bar([n.upper() for n in NAMES],acc,color=colors);ax.errorbar(range(3),acc,yerr=np.array([acc-ci[0],ci[1]-acc]),fmt='none',color='black',capsize=5)
for i,v in enumerate(acc):ax.text(i,v+2.2,f'%{v:.1f}',ha='center')
ax.set(ylim=(0,100),ylabel='Doğruluk (%)',title='1.000 çift: doğruluk ve %95 kesin binom aralığı');save(fig,'01_accuracy')
fig,ax=plt.subplots(figsize=(9,4));x=np.arange(4)
for j,n in enumerate(NAMES):ax.bar(x+(j-1)*.25,[R['category'][c]['accuracy'][n] for c in cats],width=.25,label=n.upper(),color=colors[j])
ax.set_xticks(x,['Özne–yüklem','Argüman yapısı','Adlaştırma','Düzensiz biçimler']);ax.set(ylabel='Doğruluk (%)',ylim=(0,105),title='Her kategoride 250 çift');ax.legend(ncol=3);save(fig,'02_categories')
fig,ax=plt.subplots(figsize=(7,4));x=np.arange(2)
ax.bar(x-.18,[R['primary'][n]['down'] for n in ['q8','q4']],.36,label='Doğru → yanlış',color=colors[2]);ax.bar(x+.18,[R['primary'][n]['up'] for n in ['q8','q4']],.36,label='Yanlış → doğru',color=colors[1]);ax.set_xticks(x,['Q8','Q4']);ax.set(ylabel='Çift sayısı',title='FP16 kararına göre geçişler');ax.legend();save(fig,'03_transitions')
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].scatter(M[:,0],M[:,2],s=10,alpha=.35,color=colors[2]);lo=min(M[:,0].min(),M[:,2].min());hi=max(M[:,0].max(),M[:,2].max());axes[0].plot([lo,hi],[lo,hi],c='gray',ls='--');axes[0].axhline(0,c='gray',lw=.5);axes[0].axvline(0,c='gray',lw=.5);axes[0].set(xlabel='FP16 puan farkı',ylabel='Q4 puan farkı',title='Karar sınırı ve puan hareketi')
axes[1].bar(range(1,5),[r['flips'] for r in R['margin_quartiles']['q4']],color=colors[2]);axes[1].set(xlabel='FP16 mutlak fark çeyreği (küçük → büyük)',ylabel='Q4 karar değişimi',xticks=range(1,5),title='Her çeyrekte 250 çift');save(fig,'04_margins')
fig,axes=plt.subplots(1,3,figsize=(11,4))
for ax,key,title in zip(axes,['seconds','mlx_peak_gb','weights_gb'],['Puanlama süresi (sn)','MLX tepe (GB)','Ağırlık dosyaları (GB)']):
 vals=[R['resources'][n][key] for n in NAMES];ax.bar([n.upper() for n in NAMES],vals,color=colors);ax.set_title(title)
 for i,v in enumerate(vals):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom')
 ax.set_ylim(0,max(vals)*1.15)
fig.suptitle('Özgün deney: tek yürütmenin tanımlayıcı kaynak ölçümleri');save(fig,'05_resources')
fig,ax=plt.subplots(figsize=(8,4));labels=[];values=[];intervals=[]
for j,n in [(1,'q8'),(2,'q4')]:
 for method,key in [('Kesin muhafazakâr','primary'),('Çift bootstrap','stratified_bootstrap'),('Küme bootstrap','cluster_sensitivity')]:
  labels.append(n.upper()+' — '+method);values.append(R['primary'][n]['delta_pp']);intervals.append(R[key][n]['ci_pp'] if key=='primary' else R[key][n]['delta_ci_pp'] if key=='stratified_bootstrap' else R[key]['pair_weighted_cluster_bootstrap'][n]['delta_ci_pp'])
v=np.array(values);it=np.array(intervals).T;ax.errorbar(v,np.arange(6),xerr=[v-it[0],it[1]-v],fmt='o',capsize=4,color=colors[0]);ax.axvline(0,c='gray',ls='--');ax.set_yticks(range(6),labels);ax.set(xlabel='FP16’ya göre doğruluk farkı (yüzde puan)',title='%95 aralıklar; küme tanımı keşifseldir');save(fig,'06_uncertainty')
dump('extended_statistics.json',R)
print(json.dumps({k:R[k] for k in ['audit','accuracy','primary','cluster_sensitivity','review']},ensure_ascii=False,indent=2))
