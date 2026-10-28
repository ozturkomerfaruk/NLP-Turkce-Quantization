#!/usr/bin/env python3
"""TurBLiMP main evaluation, preserving the proven pilot scoring rules.

Place this file next to score_pilot.py in scripts/.

First chunk: python scripts/score_main.py --model fp16 --max-chunks 1
Next chunks: python scripts/score_main.py --model fp16 --max-chunks 19
Runs one model per process, checkpoints atomically every 50 sentence pairs.
Standalone: includes the same scoring formula and protocol as the verified pilot.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Pin scoring parameters to the completed pilot protocol.
METHOD = 'mean_logp_float32_bos_context_no_eos_v1'
MODELS = {'fp16': 'eurollm-fp16', 'q8': 'eurollm-q8', 'q4': 'eurollm-q4'}
TIE_TOLERANCE = 0.0  # Pilot karari: delta > 0; sadece tam esitlik tie sayilir


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_mlx_memory(mx, kind):
    fn = getattr(mx, kind, None)
    if fn is None:
        fn = getattr(getattr(mx, 'metal', None), kind, None)
    if callable(fn):
        try:
            return fn()
        except (RuntimeError, AttributeError, ValueError):
            pass
    return None


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('ascii')).hexdigest()


def score_sentence(text, model, tokenizer, mx, nn, context_limit):
    # Exact pilot scoring protocol: BOS as context; all sentence tokens scored; EOS absent.
    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError('BOS token ID yok; puanlama kuralları değiştirilmemeli.')
    sentence_ids = tokenizer.encode(text, add_special_tokens=False)
    if not sentence_ids:
        raise ValueError(f'Boş tokenlaştırma: {text!r}')
    if bos in sentence_ids:
        raise ValueError('Cümlede BOS token var; tokenizer kurallarını kontrol edin.')
    ids = [int(bos)] + [int(t) for t in sentence_ids]
    if len(ids) > context_limit:
        raise ValueError(f'Cümle uzunluğu bağlam sınırını aştı: {len(ids)}>{context_limit}')
    inputs = mx.array([ids[:-1]], dtype=mx.int32)
    targets = mx.array([ids[1:]], dtype=mx.int32)
    logits = model(inputs)
    log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    selected = mx.take_along_axis(log_probs, targets[:, :, mx.newaxis], axis=-1)
    # Pilot score_pilot.py ile ayni MLX float32 toplama ve ortalama.
    total_score = mx.sum(selected)
    average_score = total_score / len(sentence_ids)
    mx.eval(total_score, average_score)
    total = float(total_score.item())
    mean = float(average_score.item())
    if not math.isfinite(total) or not math.isfinite(mean):
        raise ValueError('Gecersiz logp puani.')
    return {
        'total_logp': total,
        'mean_logp': mean,
        'n_tokens': len(sentence_ids),
        'token_ids_sha256': token_hash(sentence_ids),
    }

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'data' / 'processed' / 'turblimp_1000.jsonl'
MANIFEST = ROOT / 'data' / 'processed' / 'manifest.json'
RESULTS = ROOT / 'results' / 'main'
CHUNK_SIZE = 50
EXPECTED_SHA = 'bcb2c7f25475f8290d1511d742e1e9f8f515fd29df11dd3e3f64f2f53bcb4835'


def atomic_jsonl(path, records):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_json(path, record):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def validate_dataset():
    if not DATASET.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError('Ana veri veya manifest eksik. Önce finalize_pilot_dataset.py çalıştırın.')
    actual_sha = sha256(DATASET)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if actual_sha != EXPECTED_SHA or manifest['main_dataset']['sha256'] != actual_sha:
        raise ValueError('Ana veri SHA-256 değişmiş. Deneyi DURDURUN, veriyi değiştirmeyin.')
    rows = read_jsonl(DATASET)
    if (len(rows) != 1000 or len({r['pair_id'] for r in rows}) != 1000 or
            len({(r['sentence_good'], r['sentence_bad']) for r in rows}) != 1000):
        raise ValueError('Ana veri 1000 benzersiz çift içermiyor.')
    if Counter(r['category'] for r in rows) != Counter({c: 250 for c in manifest['category_order']}):
        raise ValueError('Ana veri kategori dağılımı bozulmuş.')
    if any(not r['sentence_good'] or not r['sentence_bad'] or r['sentence_good'] == r['sentence_bad'] for r in rows):
        raise ValueError('Ana veride geçersiz cümle çifti var.')
    return rows, actual_sha


def check_saved_chunk(path, expected_rows, model_name, dataset_sha):
    if not path.exists():
        return False
    saved = read_jsonl(path)
    if len(saved) != len(expected_rows):
        raise ValueError(f'Bozuk/eksik parça: {path}; otomatik üzerine yazılmadı.')
    for row, result in zip(expected_rows, saved):
        if (result.get('pair_id') != row['pair_id'] or
                result.get('category') != row['category'] or
                result.get('model') != model_name or
                result.get('dataset_sha256') != dataset_sha or
                result.get('method') != METHOD or
                not isinstance(result.get('correct'), bool) or
                not isinstance(result.get('tie'), bool) or
                not all(isinstance(result.get(side, {}).get(key), (int, float))
                            for side in ('good', 'bad') for key in ('total_logp', 'mean_logp', 'n_tokens')) or
                not math.isfinite(result.get('mean_logp_diff', float('nan')))):
            raise ValueError(f'Eski parça ana veri/kuralla uyuşmuyor: {path}')
    return True


def main():
    p = argparse.ArgumentParser(description='Ana TurBLiMP değerlendirmesi: her parça 50 çift')
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--max-chunks', type=int, default=1,
                   help='Bu süreçte işlenecek EN FAZLA yeni 50 çiftlik parça (varsayılan 1)')
    p.add_argument('--pause-seconds', type=float, default=1.0,
                   help='Parçalar arası bekleme saniyesi (varsayılan 1)')
    args = p.parse_args()
    if args.max_chunks < 1 or args.pause_seconds < 0 or not math.isfinite(args.pause_seconds):
        p.error('--max-chunks pozitif; --pause-seconds negatif olmayan sonlu sayı olmalı')

    rows, dataset_sha = validate_dataset()
    model_dir = ROOT / 'models' / MODELS[args.model]
    if not (model_dir / 'config.json').is_file():
        raise FileNotFoundError(f'Model bulunamadı: {model_dir}')
    outdir = RESULTS / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    chunks = [rows[i:i + CHUNK_SIZE] for i in range(0, len(rows), CHUNK_SIZE)]
    todo = []
    done = 0
    for idx, part in enumerate(chunks):
        out = outdir / f'chunk_{idx:02d}.jsonl'
        met = outdir / f'chunk_{idx:02d}.metrics.json'
        if check_saved_chunk(out, part, args.model, dataset_sha):
            done += 1
            if not met.exists():
                print(f'UYARI: Kaydedilmiş puanların ölçüm dosyası eksik: {met}')
        else:
            if met.exists():
                raise ValueError(f'Puanlama dosyası olmadan eski ölçüm dosyası var: {met}')
            todo.append((idx, part))
    if not todo:
        print(f'{args.model.upper()}: 1000/1000 çift tamamlanmış, model yüklenmedi.')
        return
    work = todo[:args.max_chunks]
    print(f'{args.model.upper()} | tamamlanmış {done}/20 parça | bu oturum: {len(work)} parça | '
          f'kalan toplam: {len(todo)} parça', flush=True)

    # The pilot protocol is implemented above; one process loads only one model.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    started_load = time.perf_counter()
    model, tokenizer = load(str(model_dir))
    model.eval()  # Pilot ile ayni degerlendirme modu.
    model_load_s = time.perf_counter() - started_load
    conf = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    context_limit = int(conf.get('max_position_embeddings', 4096))
    started_warmup = time.perf_counter()
    score_sentence(work[0][1][0]['sentence_good'], model, tokenizer, mx, nn, context_limit)
    mx.clear_cache()
    warmup_s = time.perf_counter() - started_warmup
    print(f'Model yüklendi: {model_load_s:.2f} sn | ısınma {warmup_s:.2f} sn | '
          f'BOS={tokenizer.bos_token_id} | yöntem={METHOD}', flush=True)
    reset = getattr(mx, 'reset_peak_memory', None) or getattr(getattr(mx, 'metal', None), 'reset_peak_memory', None)

    for i, (idx, part) in enumerate(work):
        if callable(reset):
            reset()
        tic = time.perf_counter()
        records = []
        tokens = 0
        for row in part:
            good = score_sentence(row['sentence_good'], model, tokenizer, mx, nn, context_limit)
            bad = score_sentence(row['sentence_bad'], model, tokenizer, mx, nn, context_limit)
            diff = good['mean_logp'] - bad['mean_logp']
            tied = abs(diff) <= TIE_TOLERANCE
            records.append({
                'pair_id': row['pair_id'], 'category': row['category'],
                'model': args.model, 'dataset_sha256': dataset_sha, 'method': METHOD,
                'good': good, 'bad': bad, 'mean_logp_diff': diff,
                'total_logp_diff': good['total_logp'] - bad['total_logp'],
                'tie': tied, 'correct': bool(diff > TIE_TOLERANCE),
            })
            tokens += good['n_tokens'] + bad['n_tokens']
            mx.clear_cache()  # pilotla ayni dusuk-bellekli scoring duzeni
        # score_sentence() already evaluates each token score synchronously.
        score_seconds = time.perf_counter() - tic
        peak = get_mlx_memory(mx, 'get_peak_memory')
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) if sys.platform == 'darwin' else None
        out = outdir / f'chunk_{idx:02d}.jsonl'
        met = outdir / f'chunk_{idx:02d}.metrics.json'
        # Save scores first. Missing metrics can be regenerated from the environment only by rerunning;
        # we report missing metrics explicitly rather than claiming they exist.
        atomic_jsonl(out, records)
        metric = {
            'model': args.model, 'chunk_index': idx, 'pairs_in_chunk': len(part),
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'dataset_sha256': dataset_sha, 'method': METHOD,
            'model_config_sha256': sha256(model_dir / 'config.json'),
            'model_load_seconds_this_process': model_load_s,
            'warmup_seconds_this_process': warmup_s,
            'score_seconds_chunk': score_seconds,
            'scored_tokens_chunk': tokens,
            'scored_tokens_per_second_chunk': tokens / score_seconds if score_seconds > 0 else None,
            'mlx_allocator_peak_gb_decimal_chunk': peak / 1e9 if peak is not None else None,
            'process_peak_rss_gb_decimal_so_far': rss / 1e9 if rss is not None else None,
            'batch_size': 1, 'mlx_lm_version': package_version('mlx-lm'),
            'mlx_version': package_version('mlx'), 'python_version': platform.python_version(),
            'score_dtype': 'float32',
            'tokenization': 'encode(text, add_special_tokens=False); single BOS context; EOS absent',
            'memory_note': 'MLX allocator peak measures allocator, not full system memory. OS peak RSS is a cumulative process high-water mark.',
        }
        atomic_json(met, metric)
        mx.clear_cache()
        print(f'[{idx + 1:02d}/20] 50 çift kaydedildi | doğru={sum(r["correct"] for r in records)}/50 | '
              f'puanlama={score_seconds:.2f} sn | MLX peak={metric["mlx_allocator_peak_gb_decimal_chunk"]} GB', flush=True)
        if i < len(work) - 1 and args.pause_seconds:
            time.sleep(args.pause_seconds)
    print(f'{args.model.upper()} oturum tamamlandı. Kalan parça: {len(todo) - len(work)}. '
          'Aynı komut kaldığı yerden devam eder.', flush=True)


if __name__ == '__main__':
    main()
