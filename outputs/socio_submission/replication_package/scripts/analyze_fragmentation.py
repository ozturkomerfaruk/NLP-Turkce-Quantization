#!/usr/bin/env python3
"""Exploratory fragmentation analysis of frozen scores; CPU only, no model imports.
Run: .venv/bin/python scripts/analyze_fragmentation.py --bootstrap 2000
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
from collections import Counter
from pathlib import Path

# Grafik önbelleğini yazılabilir geçici dizinde tut; arayüz açmadan PNG üret.
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fragmentation-matplotlib')
import matplotlib
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Proje konumu çalışma dizininden değil bu betiğin konumundan bulunur.
# Yeni analiz çıktıları eski ana deney sonuçlarından ayrı tutulur.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/fragmentation_analysis'
MODELS = ('fp16', 'q8', 'q4')
METHOD = 'mean_logp_float32_bos_context_no_eos_v1'
SEED = 20261029


# 1. SÖZCÜK SAYIMI
# Noktalama işaretlerini paydadan çıkarır; Türkçe harfleri ve sözcük içi kesmeyi korur.
# Bu bir morfolojik çözümleyici değildir: ekleri veya kökleri ayırmaz.
def word_count(text):
    """Unicode letters/numbers; internal apostrophes stay inside orthographic words."""
    return len(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text, flags=re.UNICODE))


# 2. KİMLİK DENETİMİ
# Satır sırasına güvenmeden pair_id → kayıt sözlüğü kurar.
# Aynı kimlik iki kez varsa sözlük oluştururken sessizce üzerine yazılmasını engeller.
def unique_index(rows, label, expected=None):
    ids = [r['pair_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{label}: duplicate pair IDs')
    if expected is not None and set(ids) != set(expected):
        raise ValueError(f'{label}: missing or extra pair IDs')
    return dict(zip(ids, rows))


# Dosyanın SHA-256 parmak izini üretir. İçerik değişikliğini izler; dilsel doğruluğu ölçmez.
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# JSONL dosyasında her dolu satır bağımsız bir JSON kaydıdır; hepsini listeye alır.
def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# 3. VERİLERİ BİRLEŞTİRME VE ÇİFT DÜZEYİNDE ÖLÇÜTLER
# Üç modelin aynı cümle çiftlerini ve aynı token dizilerini kullandığını denetler.
# Model ağırlığı veya tokenizer yüklemez; daha önce kaydedilmiş puanları okur.
def load_pairs(root=ROOT):
    path = root / 'data/processed/turblimp_1000.jsonl'
    source = unique_index(read_jsonl(path), 'dataset')
    if len(source) != 1000 or len(set(r['category'] for r in source.values())) != 4:
        raise ValueError('Expected exactly 1000 pairs and four categories')
    files = [path]
    scores = {}
    for model in MODELS:
        paths = sorted((root / 'results/main' / model).glob('chunk_*.jsonl'))
        files.extend(paths)
        scores[model] = unique_index([r for p in paths for r in read_jsonl(p)], model, source)
    # Önceki analizde oluşturulmuş yaklaşık kümeleri yeniden kullan.
    # Her çift tam bir kümeye bağlı olmalı; gerçek üretim şablonu oldukları varsayılmaz.
    clusterpath = root / 'results/deep_analysis/template_clusters.csv'
    files.append(clusterpath)
    cluster_map = {}
    with clusterpath.open(encoding='utf-8-sig') as f:
        clusters = list(csv.DictReader(f))
    if len({c['cluster_id'] for c in clusters}) != len(clusters):
        raise ValueError('Duplicate cluster IDs')
    for c in clusters:
        members = c['pair_ids'].split(';')
        if len(members) != int(c['n']):
            raise ValueError('Cluster size mismatch')
        for pid in members:
            if pid in cluster_map or pid not in source or source[pid]['category'] != c['category']:
                raise ValueError('Invalid cluster membership')
            cluster_map[pid] = c['cluster_id']
    if set(cluster_map) != set(source):
        raise ValueError('Incomplete cluster coverage')
    rows = []
    for pid, original in source.items():
        row = {k: original[k] for k in ('pair_id','category','sentence_good','sentence_bad')}
        row['cluster_id'] = cluster_map[pid]
        for model in MODELS:
            s = scores[model][pid]
            # Pozitif fark iyi etiketli cümlenin tercih edildiği anlamına gelir.
            # Kaydedilmiş kararla puan farkı çelişiyorsa analizi sürdürme.
            margin = s['good']['mean_logp'] - s['bad']['mean_logp']
            if (s['category'] != row['category'] or s['model'] != model or
                s['dataset_sha256'] != sha(path) or s['method'] != METHOD or
                not np.isfinite(margin) or abs(margin-s['mean_logp_diff']) > 1e-9 or
                type(s['correct']) is not bool or s['correct'] != (margin > 0) or
                s['tie'] != (margin == 0)):
                raise ValueError(f'Invalid score record: {model}/{pid}')
            row[model+'_correct'] = int(s['correct'])
            row[model+'_decision'] = 'tie' if margin == 0 else ('good' if margin > 0 else 'bad')
            row[model+'_margin'] = margin
            for side in ('good','bad'):
                v = s[side]
                ref = scores['fp16'][pid][side]
                if (type(v['n_tokens']) is not int or v['n_tokens'] <= 0 or
                    v['n_tokens'] != ref['n_tokens'] or v['token_ids_sha256'] != ref['token_ids_sha256']):
                    raise ValueError(f'Token mismatch: {model}/{pid}')
                row[f'{model}_{side}_tokens'] = v['n_tokens']
        # n_tokens daha önce BOS hariç kaydedildi: tekrar 1 ÇIKARMA.
        # Noktalama tokenları payda yoktur ama kayıtlı token payında kalır.
        for side in ('good','bad'):
            words = word_count(row['sentence_'+side])
            if words == 0:
                raise ValueError(f'No words: {pid}')
            row[side+'_words'] = words
            row[side+'_fragmentation'] = row['fp16_'+side+'_tokens'] / words
        # F = (T_iyi/W_iyi + T_hatalı/W_hatalı) / 2.
        # İki oranın ortalaması, toplam token / toplam sözcük ile her zaman aynı değildir.
        row['fragmentation'] = (row['good_fragmentation'] + row['bad_fragmentation']) / 2
        row['fragmentation_difference'] = row['good_fragmentation'] - row['bad_fragmentation']
        row['fragmentation_abs_difference'] = abs(row['fragmentation_difference'])
        row['mean_words'] = (row['good_words'] + row['bad_words']) / 2
        row['mean_tokens'] = (row['fp16_good_tokens'] + row['fp16_bad_tokens']) / 2
        row['fp16_abs_margin'] = abs(row['fp16_margin'])
        for model in ('q8','q4'):
            a, b = row['fp16_correct'], row[model+'_correct']
            row['flip_'+model] = int(row['fp16_decision'] != row[model+'_decision'])
            row['degradation_'+model] = int(a and not b)
            row['improvement_'+model] = int(not a and b)
        rows.append(row)
    # Eski deneyle aynı olayları analiz ettiğimizi sabit toplamlarla doğrula.
    counts = [sum(r[k] for r in rows) for k in ('degradation_q4','improvement_q4','flip_q4','flip_q8')]
    if counts != [42,14,56,3]:
        raise ValueError(f'Historical transition totals differ: {counts}')
    if any(r[m+'_decision']=='tie' for r in rows for m in MODELS):
        raise ValueError('Unexpected ties require a separate transition definition')
    return rows, files


# 4. DÜZENLİLEŞTİRİLMİŞ LOJİSTİK REGRESYON
# x: açıklayıcı değişkenler; y: Q4 karar değişimi (0/1); weights: örnek ağırlıkları.
# Amaç, diğer değişkenler modelde tutulduğunda parçalanma ile ilişkiyi tahmin etmektir.
# Az olaylı kategorilerde aşırı katsayıları sınırlamak için L2/ridge cezası kullanılır.
def fit_ridge(x, y, weights, penalty=1.):
    """Weighted sum log-loss + lambda/2 ||slopes||²; intercept unpenalized.

    Refit scaling each resample; zero-variance dummy columns get SD=1.
    Analytic derivatives, stable logaddexp and explicit stationarity check.
    """
    # Her bootstrap tekrarında ağırlıklı ortalama ve ölçeği yeniden öğren.
    # Sabit sütunda sıfıra bölünmemek için standart sapmayı 1 kabul et.
    mean = np.average(x, axis=0, weights=weights)
    sd = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    sd = np.where(sd < 1e-12, 1., sd)
    # İlk sütun sabit terimdir; reg[0]=0 olduğundan bu terime ridge cezası uygulanmaz.
    z = np.column_stack([np.ones(len(x)), (x-mean)/sd])
    reg = np.r_[0., np.full(x.shape[1], penalty)]
    # eta = Z beta; p = sigmoid(eta). logaddexp büyük sayılarda taşmayı önler.
    # L = toplam_i w_i [log(1 + exp(eta_i)) - y_i eta_i] + lambda/2 * toplam_j beta_j².
    def objective(b):
        eta = z @ b
        return float(weights @ (np.logaddexp(0,eta)-y*eta) + .5*np.dot(reg*b,b))
    # Gradyan birinci türevdir: Zᵀ [w * (p-y)] + ceza * beta.
    def gradient(b):
        return z.T @ (weights*(expit(z@b)-y)) + reg*b
    # Hessian ikinci türev matrisidir; Newton-CG adımının eğriliğini belirler.
    def hessian(b):
        p=expit(z@b)
        return z.T @ ((weights*p*(1-p))[:,None]*z) + np.diag(reg)
    # Başlangıç sabiti gözlenen olay oranının log-odds değeridir.
    # Son kontrolde yalnızca optimizer mesajına değil gradyanın küçüklüğüne bakılır.
    start=np.zeros(z.shape[1])
    rate=np.average(y,weights=weights)
    if not 0 < rate < 1:
        raise ValueError('No outcome variation in resample')
    start[0]=np.log(rate/(1-rate))
    res=minimize(objective,start,jac=gradient,hess=hessian,method='Newton-CG',options={'xtol':1e-9,'maxiter':200})
    if not np.all(np.isfinite(res.x)) or np.max(np.abs(gradient(res.x))) > 1e-5:
        raise RuntimeError(f'Ridge convergence failed: {res.message}')
    return res.x, mean, sd


# Bootstrap dağılımının %2,5 ve %97,5 yüzdeliklerini alır: merkezdeki %95 aralık.
def interval(values):
    return np.quantile(values,[.025,.975]).tolist()


# Tek bir olay oranı için Clopper–Pearson kesin binom aralığı.
# Bu tanımlayıcı aralık küme bağımlılığını hesaba katmaz; küme bootstrap ayrı hesaplanır.
def binomial_ci(k,n):
    if not n:
        return None
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


# 5. GRUP ÖZETLERİ
# Tüm veri, kategori veya çeyrek için aynı doğruluk ve geçiş oranı hesabını uygular.
# Bozulma/iyileşme oranlarının paydası ilgili grubun TÜM çiftleridir.
def summary(rows):
    n=len(rows)
    ans={'n':n}
    for m in MODELS:
        k=sum(r[m+'_correct'] for r in rows)
        ans[m+'_accuracy']=k/n if n else None
        ans[m+'_correct']=k
    for m in ('q4','q8'):
        for event in ('flip','degradation','improvement'):
            k=sum(r[event+'_'+m] for r in rows)
            ans[event+'_'+m]={'count':k,'rate':k/n if n else None,'ci95_exact':binomial_ci(k,n)}
    ans['fragmentation_range']=[min(r['fragmentation'] for r in rows),max(r['fragmentation'] for r in rows)] if n else None
    return ans


# %25, %50 ve %75 eşiklerini bulur; eşit değerleri keyfi olarak iki gruba bölmez.
# side="left" eşikteki değeri alt gruba koyar; grup boyutları tam 250 olmayabilir.
def quartiles(values):
    edges=np.quantile(values,[.25,.5,.75])
    return np.searchsorted(edges,values,side='left')+1,edges.tolist()


# 6. REGRESYONUN AÇIKLAYICI DEĞİŞKEN MATRİSİ
# Her satır bir minimal çift, her sütun bir özelliktir.
# kind ile ana model ve farklı kontrol değişkenleri kullanan duyarlılık modelleri seçilir.
def design(rows, kind):
    features=['fragmentation']
    cats=sorted(set(r['category'] for r in rows))
    if kind != 'unadjusted':
        features += ['log_abs_margin' if kind=='log_margin' else 'fp16_abs_margin',
                     'mean_tokens' if kind=='token_length' else 'mean_words']
        if kind=='difference_control': features += ['fragmentation_difference']
    cols=[]
    for k in features:
        cols.append(np.array([np.log(r['fp16_abs_margin']+1e-4) if k=='log_abs_margin' else r[k] for r in rows]))
    # Bir kategori referans bırakılır; kalanlar 0/1 gösterge sütunları olur.
    # Böylece sabit terimle birlikte tüm kategori sütunlarının doğrusal bağımlılığı önlenir.
    if kind!='unadjusted':
        for c in cats[1:]:
            cols.append(np.array([float(r['category']==c) for r in rows]))
            features.append('category:'+c)
    return np.column_stack(cols),features


# Standartlaştırılmış parçalanma katsayısını özgün token/sözcük birimine geri çevirir.
# Parçalanma ilk özellik olduğu için katsayısı b[1], ölçeği sd[0] konumundadır.
def effect(x,y,w,penalty=1.):
    b,mean,sd=fit_ridge(x,y,w,penalty)
    return b[1]/sd[0], b, mean, sd


# 7. BOOTSTRAP İÇİN YENİDEN ÖRNEKLEME AĞIRLIKLARI
# Aynı satırı bellekte kopyalamak yerine kaç kez seçildiğini ağırlık olarak saklar.
# cluster=False: minimal çiftler; cluster=True: yaklaşık şablon kümeleri seçilir.
# Her iki durumda da dört dilbilgisel kategorinin özgün ağırlığı korunur.
def resample_weights(rows,rng,cluster=False):
    w=np.zeros(len(rows))
    cats=sorted(set(r['category'] for r in rows))
    for c in cats:
        ix=np.array([i for i,r in enumerate(rows) if r['category']==c])
        if cluster:
            ids=sorted(set(rows[i]['cluster_id'] for i in ix))
            # K kümeden yerine koyarak K seçim: counts her kümenin seçilme sayısıdır.
            # Aynı kümedeki çiftler birlikte ağırlık alır; bağımsız çift kabul edilmez.
            counts=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
            lookup=dict(zip(ids,counts))
            w[ix]=[lookup[rows[i]['cluster_id']] for i in ix]
            # Küme boyutları farklı olduğundan seçilen toplam çift ağırlığı değişebilir.
            # Kategori toplamını özgün n_c değerine getir: w ← w * n_c / toplam(w).
            w[ix]*=len(ix)/w[ix].sum()
        else:
            # n_c çiftten yerine koyarak n_c seçim; tüm kategori ağırlığı tam n_c olur.
            w[ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)))
    return w


# Bir modelin özgün 1.000 çift üzerindeki nokta tahminini ve ölçeklerini paketler.
# exp(katsayı) odds oranıdır; olasılık farkı veya risk oranı değildir.
def model_result(rows,kind,penalty=1.):
    x,names=design(rows,kind)
    y=np.array([r['flip_q4'] for r in rows],float)
    slope,b,mean,sd=effect(x,y,np.ones(len(rows)),penalty)
    return {'features':names,'penalty':penalty,'or_per_token_per_word':float(np.exp(slope)),
            'or_per_sd':float(np.exp(slope*np.std(x[:,0]))),'coefficients_standardized':b.tolist(),
            'means':mean.tolist(),'scales':sd.tolist()},x,y


# 8. TÜM ANALİZİN ÇALIŞMA AKIŞI
# Sıra: doğrula → ölç → grupla → regresyon → bootstrap → dosya koruma → çıktılar.
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap < 200: parser.error('Use at least 200 replicates (2000 recommended)')
    rows,files=load_pairs()
    # Önce/sonra hash karşılaştırması eski kaynakların değişmediğini denetler.
    # Bu betik ve __pycache__ dosyaları bu koruma listesinin dışında tutulur.
    protected=[p for folder in ('scripts','data','results/analysis','results/main','results/deep_analysis')
               for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='analyze_fragmentation.py']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    provenance={str(p.relative_to(ROOT)):sha(p) for p in files}
    q,edges=quartiles([r['fragmentation'] for r in rows])
    for r,v in zip(rows,q): r['quartile']=int(v)
    cats=sorted(set(r['category'] for r in rows))
    result={'exploratory':True,'seed':SEED,'bootstrap_replicates':args.bootstrap,
            'n_pairs':len(rows),'source_sha256':provenance,'script_sha256':sha(Path(__file__)),
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
            'overall':summary(rows),'quartile_edges':edges,
            'quartile_policy':'Linear empirical quantiles; equal values stay together in lower group; no arbitrary tie splitting',
            'quartiles':{str(i):summary([r for r in rows if r['quartile']==i]) for i in range(1,5)},
            'categories':{}}
    # Genel eşikler kategoriler arası kıyas, kategori-içi eşikler yerel dağılım içindir.
    for c in cats:
        subset=[r for r in rows if r['category']==c]
        cq,ce=quartiles([r['fragmentation'] for r in subset])
        for r,v in zip(subset,cq): r['category_quartile']=int(v)
        result['categories'][c]={'overall':summary(subset),'within_category_edges':ce,
            'global_quartiles':{str(i):summary([r for r in subset if r['quartile']==i]) for i in range(1,5)},
            'within_category_quartiles':{str(i):summary([r for r in subset if r['category_quartile']==i]) for i in range(1,5)}}
    result['models']={}
    matrices={}
    for kind in ('unadjusted','adjusted','token_length','difference_control','log_margin'):
        result['models'][kind],x,y=model_result(rows,kind)
        matrices[kind]=x
    # Aynı bulgunun yalnızca ceza gücüne veya tek bir kategoriye bağlılığını sorgula.
    # Bu iki duyarlılık ailesi nokta tahmini verir; bootstrap aralıkları hesaplanmaz.
    result['penalty_sensitivity']={str(p):model_result(rows,'adjusted',p)[0] for p in (.1,10.)}
    result['leave_one_category_out']={c:model_result([r for r in rows if r['category']!=c],'adjusted')[0] for c in cats}
    result['clusters']={'n':len(set(r['cluster_id'] for r in rows)),
        'per_category':{c:len(set(r['cluster_id'] for r in rows if r['category']==c)) for c in cats},
        'source':'Existing shared-leading-discourse-word stripping proxy; not verified true generation templates'}
    frag=np.array([r['fragmentation'] for r in rows]); sd_frag=float(frag.std())
    result['fragmentation_sd']=sd_frag
    result['correlations']={'fragmentation_vs_abs_margin':float(np.corrcoef(frag,[r['fp16_abs_margin'] for r in rows])[0,1]),
                            'fragmentation_vs_mean_words':float(np.corrcoef(frag,[r['mean_words'] for r in rows])[0,1])}
    result['bootstrap']={}
    # İki örnekleme yaklaşımı için ayrı ve tekrarlanabilir rastgele sayı akışları.
    # Her tekrarda aynı ağırlıklar beş modele uygulanır; FP16/Q4 eşleşmesi korunur.
    for mode in ('pair','cluster'):
        rng=np.random.default_rng(SEED+(mode=='cluster'))
        slopes={kind:[] for kind in matrices}
        rates=[]; contrasts=[]
        for b in range(args.bootstrap):
            w=resample_weights(rows,rng,mode=='cluster')
            for kind in slopes:
                slope,*_=effect(matrices[kind],y,w)
                slopes[kind].append(slope)
            # Çeyrek sınırları özgün veriden SABİT tutulur; her tekrarda yeniden seçilmez.
            # Böylece bu belirli grupların olay oranlarının belirsizliği ölçülür.
            qr=[float(np.average(y[q==i],weights=w[q==i])) if w[q==i].sum() else None for i in range(1,5)]
            if None in qr: raise ValueError('Empty quartile in bootstrap')
            rates.append(qr);contrasts.append(qr[3]-qr[0])
            if (b+1)%500==0: print(f'{mode}: {b+1}/{args.bootstrap}',flush=True)
        # Farklı tekrarların katsayılarını aynı özgün örneklem SS biriminde karşılaştır.
        # Önce yüzdelikler, sonra üstel dönüşüm: log-odds → odds oranı.
        result['bootstrap'][mode]={'successful':args.bootstrap,'failed':0,
            'models':{kind:{'or_per_sd_ci95':np.exp(np.quantile(v,[.025,.975])*sd_frag).tolist(),
                            'or_per_token_per_word_ci95':np.exp(np.quantile(v,[.025,.975])).tolist()} for kind,v in slopes.items()},
            'quartile_flip_rate_ci95':np.quantile(rates,[.025,.975],axis=0).T.tolist(),
            'highest_minus_lowest_rate_ci95':interval(contrasts)}
    result['bootstrap_method']='Category-stratified percentile intervals; pairs kept matched; cluster weights normalized to original category size; scaling refit; OR converted to original-sample units. Fixed lambda. No p-values.'
    result['highest_minus_lowest_rate']=result['quartiles']['4']['flip_q4']['rate']-result['quartiles']['1']['flip_q4']['rate']
    result['protected_files_unchanged']=all(sha(ROOT/p)==digest for p,digest in before.items())
    if not result['protected_files_unchanged']: raise RuntimeError('Protected input changed during analysis')
    result['protected_file_count']=len(before)
    # Tüm hesaplar ve koruma denetimi bitince dört ana çıktıyı yaz.
    # CSV UTF-8 BOM kullanır; JSON NaN/Infinity değerlerine izin vermez.
    OUT.mkdir(exist_ok=True)
    with (OUT/'fragmentation_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'fragmentation_statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    plot(result)
    report(result)
    print(json.dumps({'overall':result['overall'],'models':{k:v['or_per_sd'] for k,v in result['models'].items()},'bootstrap':result['bootstrap']},indent=2))


# 9. GRAFİK ÜRETİMİ
# Çubuklar gözlenen Q4 karar değişimi yüzdesi; çizgiler küme bootstrap %95 aralığı.
# Grafikte doğruluk veya yalnızca bozulma değil, iki yönlü TOPLAM değişim gösterilir.
def plot(s):
    rates=np.array([s['quartiles'][str(i)]['flip_q4']['rate'] for i in range(1,5)])*100
    ci=np.array(s['bootstrap']['cluster']['quartile_flip_rate_ci95'])*100
    fig,ax=plt.subplots(figsize=(9,5.5))
    fig.subplots_adjust(left=.10,right=.98,top=.88,bottom=.20)
    ax.bar(range(4),rates,color='#247e8e',width=.6)
    ax.vlines(range(4),ci[:,0],ci[:,1],color='#172e40',linewidth=2)
    for i,(v,lo,hi) in enumerate(zip(rates,ci[:,0],ci[:,1])):
        ax.hlines([lo,hi],i-.09,i+.09,color='#172e40',linewidth=2)
        n=s['quartiles'][str(i+1)]['n']; k=s['quartiles'][str(i+1)]['flip_q4']['count']
        ax.text(i,hi+.45,f'{k}/{n} · %{v:.1f}',ha='center',fontsize=10)
    ax.set_xticks(range(4),['En düşük','Düşük-orta','Yüksek-orta','En yüksek'])
    ax.set_ylabel('FP16 → Q4 karar değişimi (%)');ax.set_xlabel('Alt sözcük parçalanma çeyreği')
    ax.set_title('Parçalanma ve Q4 karar değişimi',loc='left',fontweight='bold')
    ax.set_ylim(0,max(ci[:,1])+3);ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.035,'Keşifsel · %95 kategori tabakalı yaklaşık-küme bootstrap aralığı',ha='center',fontsize=9)
    fig.savefig(OUT/'fragmentation_plot.png',dpi=180);plt.close(fig)


# 10. KISA SONUÇ RAPORUNU ÜRETME
# Hesaplanmış sözlükten tablolar ve yöntem açıklamaları oluşturur; yeniden model kurmaz.
# Ayrıntılı öğretici belge: ProjeMarkdown/ALT_SOZCUK_PARCALANMASI_KOD_VE_YONTEM_REHBERI.md.
def report(s):
    # Yalnızca sunum yardımcıları: 0–1 oranını yüzdeye ve grup sözlüklerini tabloya çevirir.
    def pct(v): return f'{100*v:.2f}' if v is not None else '—'
    def table(groups):
        out=['| Grup | n | FP16 % | Q4 % | Doğru→yanlış n (%) | Yanlış→doğru n (%) | Değişim n (%) |',
             '|---|---:|---:|---:|---:|---:|---:|']
        for label,g in groups.items():
            cells=[f"{g[e+'_q4']['count']} ({pct(g[e+'_q4']['rate'])})" for e in ('degradation','improvement','flip')]
            out.append(f"| {label} | {g['n']} | {pct(g['fp16_accuracy'])} | {pct(g['q4_accuracy'])} | "+' | '.join(cells)+' |')
        return '\n'.join(out)
    lines=['# Alt sözcük parçalanması ve Q4 duyarlılığı — keşifsel ek analiz','',
        '## Kapsam ve yöntem','',
        'Ana deneyden sonra tasarlanan bu analiz, önceden belirlenmiş doğrulayıcı bir hipotez testi değildir. Mevcut 1.000 minimal çift ve üç modelin kayıtları pair_id üzerinden birleştirildi. Eksik/tekrarlı kimlik, kategori, karar, veri hash’i veya token uyumsuzluğunda işlem durur. Hiçbir model yüklenmedi; puanlama tekrarlanmadı.',
        '', 'Sözcükler Unicode harf/sayı dizileridir; sözcük içi düz/kıvrık kesme korunur, tire ayırır, bağımsız noktalama sayılmaz. Sayılar sözcük birimi sayılır. Kayıtlı n_tokens BOS ve EOS içermez; ayrıca 1 çıkarılmadı. Noktalama tokenları mevcut deney sayısında bulunduğundan payda sözcük saymazken pay bunları içerir. Bu nedenle ölçüt saf sözcük-içi parçalanma veya morfem/ek sayısı değildir; cümle düzeyinde token/sözcük yoğunluğu vekilidir.',
        '', 'Ana ölçüt doğru ve hatalı cümlenin token/sözcük oranlarının aritmetik ortalamasıdır. İşaretli fark doğru−hatalı olarak, mutlak fark ayrıca saklandı. Uzunluk kontrolü iki cümlenin ortalama sözcük sayısıdır. FP16 marjı ortalama log-olasılık(doğru)−ortalama log-olasılık(hatalı) olup mutlak değeri kullanıldı.',
        '',f"Çeyrek eşikleri: {', '.join(f'{v:.6f}' for v in s['quartile_edges'])}. Sınırlardaki eşit değerler alt grupta tutuldu; gruplar bu nedenle tam 250 olmayabilir. Kategori tablolarında aynı genel eşikler kullanıldı; kategori-içi çeyrekler de JSON ve CSV’de sunuldu.",
        '', 'Düzenlileştirilmiş lojistik regresyonun sonucu flip_q4’tür. Amaç fonksiyonu toplam negatif log-olabilirlik + λ/2 × eğimlerin kareleri toplamıdır; sabit terim cezalandırılmaz. Ana λ=1; λ=0,1 ve 10 duyarlılıkları ayrıca hesaplandı. Sürekli değişkenler ve kategori kuklaları standartlaştırıldı. Ana model parçalanma, mutlak FP16 marjı, ortalama sözcük sayısı ve kategori içerir. OR, karar değişiminin odds oranıdır; risk oranı veya nedensel etki değildir.',
        '',f"Her yeniden örnekleme türünde {s['bootstrap_replicates']:,} tekrar, sabit tohum {s['seed']} kullanıldı. Minimal çift bütünlüğü korunarak kategori tabakalı çift bootstrap yapıldı. İkinci analiz mevcut {s['clusters']['n']} yaklaşık şablon kümesini kategori içinde bütün olarak yeniden örnekledi; kategori ağırlıkları başlangıç boyutuna normalize edildi. Her tekrarda ölçekleme ve regresyon yeniden kuruldu, OR özgün örneklem standart sapmasına çevrildi. %95 yüzdelik aralıklar düzenlileştirilmiş tahminin belirsizliğini gösterir; düzenlileştirme yanlılığını gidermez. Her iki türde başarısız tekrar sayısı 0’dır.",
        '', '## Sonuçlar','',table({'Tümü':s['overall']}),'',table(s['quartiles']),'',
        f"Q8: {s['overall']['flip_q8']['count']} karar değişimi; yalnızca tanımlayıcıdır. Genel Q8 doğruluğu %{pct(s['overall']['q8_accuracy'])}. Kategori ve çeyrek ayrıntıları JSON’dadır.",
        '', '| Model | Parçalanma OR / 1 SS | Çift bootstrap %95 GA | Küme bootstrap %95 GA |','|---|---:|---|---|']
    for kind,label in [('unadjusted','Kontrolsüz'),('adjusted','Marj + kategori + sözcük uzunluğu kontrollü')]:
        a=s['bootstrap']['pair']['models'][kind]['or_per_sd_ci95'];b=s['bootstrap']['cluster']['models'][kind]['or_per_sd_ci95']
        lines.append(f"| {label} | {s['models'][kind]['or_per_sd']:.3f} | [{a[0]:.3f}, {a[1]:.3f}] | [{b[0]:.3f}, {b[1]:.3f}] |")
    lines += ['',f"Bir standart sapma = {s['fragmentation_sd']:.4f} token/sözcük. En yüksek−en düşük çeyrek karar değişimi farkı {100*s['highest_minus_lowest_rate']:+.2f} yüzde puan; küme bootstrap %95 GA {np.round(np.array(s['bootstrap']['cluster']['highest_minus_lowest_rate_ci95'])*100,2).tolist()} yüzde puandır.", '', '## Kategori karşılaştırmaları','']
    for c,v in s['categories'].items(): lines += ['### '+c,'',table(v['global_quartiles']),'']
    lines += ['## Duyarlılık analizleri','', '| Değişiklik | Parçalanma OR / 1 SS | Küme bootstrap %95 GA |','|---|---:|---|']
    for kind in ('token_length','difference_control','log_margin'):
        ci_alt=s['bootstrap']['cluster']['models'][kind]['or_per_sd_ci95']
        lines.append(f"| {kind} | {s['models'][kind]['or_per_sd']:.3f} | [{ci_alt[0]:.3f}, {ci_alt[1]:.3f}] |")
    for p,v in s['penalty_sensitivity'].items(): lines.append(f"| λ={p} | {v['or_per_sd']:.3f} | Hesaplanmadı |")
    for c,v in s['leave_one_category_out'].items(): lines.append(f"| {c} dışarıda | {v['or_per_sd']:.3f} | Hesaplanmadı |")
    ci=s['bootstrap']['cluster']['models']['adjusted']['or_per_sd_ci95']
    conclusion=('Kontrollerden sonra pozitif ilişki küme bootstrap aralığında da korunuyor.' if ci[0]>1 else
                'Kontrollerden sonra negatif ilişki küme bootstrap aralığında da korunuyor.' if ci[1]<1 else
                'Kontrollerden sonra küme bootstrap aralığı 1’i içeriyor: parçalanmanın Q4 karar değişimleriyle bağımsız ilişkisi bu veriyle açık biçimde desteklenmiyor. Bu, ilişkinin olmadığına dair kanıt veya eşdeğerlik sonucu değildir.')
    lines += ['', '## Yorum ve sınırlılıklar','',conclusion,'',
        'Daha yüksek parçalanmanın daha fazla Q4 karar değişimiyle ilişkili olduğu beklentisi desteklenmedi. Çeyreklerde düzenli bir artış/azalış yoktur; en yüksek ve en düşük çeyrek farkının küme aralığı sıfırı içerir. Regresyonun ortak doğrusal log-odds eğimi bu düzensiz dağılımı özetler; kategori etkileşimlerini veya doğrusal olmayan ilişkiyi kanıtlamaz. Token uzunluğu kontrolünde etkinin zayıflaması nedeniyle negatif eğilim de yöntemden bağımsız, koruyucu bir etki olarak yorumlanmamalıdır. Kategori çıkarma tahminlerinin tümü aynı yöndedir; bu tek kategori açıklamasına karşı bir duyarlılık bulgusudur, alt grup ilişkilerinin eşitliğinin kanıtı değildir.', '',
        'Ceza katsayısı ve kategori çıkarma duyarlılıkları için yalnızca nokta tahminleri verildi; bunlar ayrı anlamlılık iddiaları değildir. Log-marj kontrolü log(|marj|+0,0001) kullanır. Kategori çıkarma analizleri tek bir kategorinin etkisini inceler, fakat alt grup olay sayıları sınırlıdır. Bozulma ve iyileşme ayrı tanımlayıcı sonuçlardır; ana regresyon toplam karar değişimini açıklar, doğrudan doğruluk kaybını açıklamaz.',
        '', 'Yaklaşık kümeler önceki analizde ortak baştaki söylem/zaman sözcüklerinin çıkarılmasıyla oluşturulmuştur; gerçek üretim şablonları değildir. Bootstrap bu kümelerin ötesindeki bağımlılığı gideremez. FP16 marjının fonksiyonel biçimi, uzunluk ile oran arasındaki matematiksel ilişki, noktalama katkısı ve sözcük tanımı sonuçları etkileyebilir. Bu gözlemsel analiz nedensellik göstermez ve tüm Türkçeye genellenemez. Çok sayıda keşifsel karşılaştırma için p-değerleri üretilmedi; aralıklar çoklu karşılaştırma için düzeltilmedi.',
        '', 'Tartışmalı etiketlerin bağımsız uzmanlarca kör incelenmesi ve gerçek üretim şablonlarının doğrulanması sonraki işlerdir; bu analiz bunları tamamlamış sayılmaz. Makale yazımı bu teslimatın kapsamına alınmadı.',
        '', '## Yeniden üretim ve koruma','',
        '`.venv/bin/python scripts/analyze_fragmentation.py --bootstrap 2000`',
        '',f"Girdi SHA-256 değerleri, sürümler, katsayılar, ölçekler ve bootstrap ayrıntıları fragmentation_statistics.json içindedir. Çalışma öncesi/sonrası {s['protected_file_count']} mevcut dosyanın hash’i karşılaştırıldı: değişiklik yok. Ana deney ve eski istatistik dosyaları korunmuştur.",
        '', 'Yöntem kaynakları: [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), [SciPy minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).',
        '', '![Parçalanma ve karar değişimi](fragmentation_plot.png)','']
    (OUT/'fragmentation_report.md').write_text('\n'.join(lines),encoding='utf-8')


# Dosya doğrudan çalıştırılırsa analizi başlat; testler import ederse başlatma.
if __name__=='__main__': main()
