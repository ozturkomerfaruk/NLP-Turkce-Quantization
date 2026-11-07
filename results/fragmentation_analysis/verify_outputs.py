"""Independent delivery checks, including source preservation and join failure cases."""
import csv, hashlib, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_fragmentation as a
OUT=Path(__file__).resolve().parent
s=json.loads((OUT/'fragmentation_statistics.json').read_text())
rows=list(csv.DictReader((OUT/'fragmentation_pairs.csv').open(encoding='utf-8-sig')))
assert len(rows)==len({r['pair_id'] for r in rows})==1000
assert [sum(int(r[k]) for r in rows) for k in ('degradation_q4','improvement_q4','flip_q4','flip_q8')]==[42,14,56,3]
for key,digest in s['source_sha256'].items():
    assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest()==digest
assert hashlib.sha256((ROOT/'scripts/analyze_fragmentation.py').read_bytes()).hexdigest()==s['script_sha256']
assert s['protected_files_unchanged']
for r in rows:
    expected=(int(r['fp16_good_tokens'])/int(r['good_words'])+int(r['fp16_bad_tokens'])/int(r['bad_words']))/2
    assert abs(expected-float(r['fragmentation']))<1e-12
    assert int(r['flip_q4'])==int(r['degradation_q4'])+int(r['improvement_q4'])
for i in range(1,5):
    sub=[r for r in rows if int(r['quartile'])==i]
    assert len(sub)==s['quartiles'][str(i)]['n']
    assert sum(int(r['flip_q4']) for r in sub)==s['quartiles'][str(i)]['flip_q4']['count']
# Independent BFGS fit on saved design scalings, compared with Newton-CG result.
loaded,_=a.load_pairs()
x,names=a.design(loaded,'adjusted');y=np.array([r['flip_q4'] for r in loaded])
m=s['models']['adjusted'];z=np.column_stack([np.ones(1000),(x-m['means'])/m['scales']])
def objective(b): return np.logaddexp(0,z@b).sum()-y@(z@b)+.5*np.sum(b[1:]**2)
def jac(b): return z.T@(expit(z@b)-y)+np.r_[0,b[1:]]
r=minimize(objective,np.zeros(z.shape[1]),jac=jac,method='BFGS',options={'gtol':1e-7})
assert np.max(np.abs(jac(r.x)))<1e-5
np.testing.assert_allclose(r.x,m['coefficients_standardized'],atol=1e-5)
# Real loading path must reject both duplicate and missing chunk records.
with tempfile.TemporaryDirectory() as tmp:
    base=Path(tmp)
    for rel in s['source_sha256']:
        dest=base/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    target=base/'results/main/q4/chunk_00.jsonl';original=target.read_text();lines=original.splitlines()
    for bad in ('\n'.join(lines+[lines[0]])+'\n','\n'.join(lines[1:])+'\n'):
        target.write_text(bad)
        try: a.load_pairs(base)
        except ValueError: pass
        else: raise AssertionError('Loader accepted invalid pair identity coverage')
    target.write_text(original)
for mode in ('pair','cluster'):
    assert s['bootstrap'][mode]['successful']==2000 and s['bootstrap'][mode]['failed']==0
print('PASS: 1000 rows, transitions, all source hashes, protected-file audit, ratios, quartiles, independent optimizer, duplicate/missing rejection, 4000 bootstrap replicates.')
