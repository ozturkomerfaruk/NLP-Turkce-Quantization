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
TIE_TOLERANCE = 0.0  # Pilot rule: delta > 0 is correct; only exact equality is a tie.


# Bind datasets and configurations to their exact file contents.
def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# Preserve record order so chunk positions remain aligned with the frozen benchmark.
def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


# Capture installed dependency versions for reproducibility; tolerate unavailable metadata.
def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# Support both current and older MLX memory APIs without inventing missing measurements.
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


# Fingerprint the exact token sequence to detect tokenization drift across precisions.
def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('ascii')).hexdigest()


# Compute total and mean next-token log-probability under the frozen pilot protocol.
# Use float32 for log-softmax and reductions even when model weights are quantized.
def score_sentence(text, model, tokenizer, mx, nn, context_limit):
    # Exact pilot scoring protocol: BOS as context; all sentence tokens scored; EOS absent.
    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError('BOS token ID is missing; the scoring protocol must be preserved.')
    sentence_ids = tokenizer.encode(text, add_special_tokens=False)
    if not sentence_ids:
        raise ValueError(f'Empty tokenization: {text!r}')
    if bos in sentence_ids:
        raise ValueError('Sentence contains a BOS token; check tokenizer rules.')
    ids = [int(bos)] + [int(t) for t in sentence_ids]
    if len(ids) > context_limit:
        raise ValueError(f'Sentence length exceeds the context limit: {len(ids)}>{context_limit}')
    inputs = mx.array([ids[:-1]], dtype=mx.int32)
    targets = mx.array([ids[1:]], dtype=mx.int32)
    logits = model(inputs)
    log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    selected = mx.take_along_axis(log_probs, targets[:, :, mx.newaxis], axis=-1)
    # Match the pilot's MLX float32 reduction and length normalization.
    total_score = mx.sum(selected)
    average_score = total_score / len(sentence_ids)
    mx.eval(total_score, average_score)
    total = float(total_score.item())
    mean = float(average_score.item())
    if not math.isfinite(total) or not math.isfinite(mean):
        raise ValueError('Invalid log-probability score.')
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


# Flush and sync the complete chunk before replacing its checkpoint file.
def atomic_jsonl(path, records):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


# Commit metrics only after their full JSON payload has been flushed to disk.
def atomic_json(path, record):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


# Reject altered data, duplicate pairs or category imbalance before starting evaluation.
def validate_dataset():
    if not DATASET.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError('Main dataset or manifest is missing. Run finalize_pilot_dataset.py first.')
    actual_sha = sha256(DATASET)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if actual_sha != EXPECTED_SHA or manifest['main_dataset']['sha256'] != actual_sha:
        raise ValueError('Main dataset SHA-256 has changed. STOP the experiment and preserve the data.')
    rows = read_jsonl(DATASET)
    if (len(rows) != 1000 or len({r['pair_id'] for r in rows}) != 1000 or
            len({(r['sentence_good'], r['sentence_bad']) for r in rows}) != 1000):
        raise ValueError('The main dataset does not contain 1000 unique pairs.')
    if Counter(r['category'] for r in rows) != Counter({c: 250 for c in manifest['category_order']}):
        raise ValueError('Main dataset category distribution is inconsistent.')
    if any(not r['sentence_good'] or not r['sentence_bad'] or r['sentence_good'] == r['sentence_bad'] for r in rows):
        raise ValueError('The main dataset contains an invalid sentence pair.')
    return rows, actual_sha


# Resume only from saved chunks matching the expected data, model and scoring protocol.
def check_saved_chunk(path, expected_rows, model_name, dataset_sha):
    if not path.exists():
        return False
    saved = read_jsonl(path)
    if len(saved) != len(expected_rows):
        raise ValueError(f'Corrupt or incomplete chunk: {path}; not overwritten automatically.')
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
            raise ValueError(f'Saved chunk does not match the dataset or protocol: {path}')
    return True


# Validate checkpoints first, then load one model and score only unfinished chunks.
def main():
    p = argparse.ArgumentParser(description='Main TurBLiMP evaluation: 50 pairs per chunk')
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--max-chunks', type=int, default=1,
                   help='Maximum number of new 50-pair chunks in this process (default: 1)')
    p.add_argument('--pause-seconds', type=float, default=1.0,
                   help='Pause between chunks in seconds (default: 1)')
    args = p.parse_args()
    if args.max_chunks < 1 or args.pause_seconds < 0 or not math.isfinite(args.pause_seconds):
        p.error('--max-chunks must be positive; --pause-seconds must be finite and nonnegative')

    rows, dataset_sha = validate_dataset()
    model_dir = ROOT / 'models' / MODELS[args.model]
    if not (model_dir / 'config.json').is_file():
        raise FileNotFoundError(f'Model not found: {model_dir}')
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
                print(f'WARNING: Saved scores have no metrics file: {met}')
        else:
            if met.exists():
                raise ValueError(f'A metrics file exists without a score file: {met}')
            todo.append((idx, part))
    if not todo:
        print(f'{args.model.upper()}: 1000/1000 pairs complete; model loading skipped.')
        return
    work = todo[:args.max_chunks]
    print(f'{args.model.upper()} | completed {done}/20 chunks | this session: {len(work)} chunks | '
          f'total remaining: {len(todo)} chunks', flush=True)

    # The pilot protocol is implemented above; one process loads only one model.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    # Measure loading and warmup separately from per-chunk scoring time.
    started_load = time.perf_counter()
    model, tokenizer = load(str(model_dir))
    model.eval()  # Use the same evaluation mode as the pilot.
    model_load_s = time.perf_counter() - started_load
    conf = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    context_limit = int(conf.get('max_position_embeddings', 4096))
    started_warmup = time.perf_counter()
    score_sentence(work[0][1][0]['sentence_good'], model, tokenizer, mx, nn, context_limit)
    mx.clear_cache()
    warmup_s = time.perf_counter() - started_warmup
    print(f'Model loaded: {model_load_s:.2f} s | warmup {warmup_s:.2f} s | '
          f'BOS={tokenizer.bos_token_id} | method={METHOD}', flush=True)
    reset = getattr(mx, 'reset_peak_memory', None) or getattr(getattr(mx, 'metal', None), 'reset_peak_memory', None)

    # Each chunk is a checkpoint: compare both sentences, persist scores, then metrics.
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
            mx.clear_cache()  # Keep the pilot's cache-clearing policy between pairs.
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
        print(f'[{idx + 1:02d}/20] 50 pairs saved | correct={sum(r["correct"] for r in records)}/50 | '
              f'scoring={score_seconds:.2f} s | MLX peak={metric["mlx_allocator_peak_gb_decimal_chunk"]} GB', flush=True)
        if i < len(work) - 1 and args.pause_seconds:
            time.sleep(args.pause_seconds)
    print(f'{args.model.upper()} session complete. Remaining chunks: {len(todo) - len(work)}. '
          'The same command resumes from the next unfinished chunk.', flush=True)


if __name__ == '__main__':
    main()
