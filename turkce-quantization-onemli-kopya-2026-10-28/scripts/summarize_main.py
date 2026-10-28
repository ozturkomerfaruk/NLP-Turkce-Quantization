#!/usr/bin/env python3
"""Validate complete 1000-pair x 3-model experiment and print descriptive results.
No MLX dependency. Use after score_main.py finishes all 20 chunks per model.
"""

import json
from collections import Counter
from pathlib import Path

from score_main import METHOD, MODELS, read_jsonl, sha256

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'data' / 'processed' / 'turblimp_1000.jsonl'
MANIFEST = ROOT / 'data' / 'processed' / 'manifest.json'
OUT = ROOT / 'results' / 'main'


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    dataset_sha = sha256(DATASET)
    if dataset_sha != manifest['main_dataset']['sha256']:
        raise ValueError('Ana veri manifest hash uyumsuzluğu')
    source = read_jsonl(DATASET)
    ids = [r['pair_id'] for r in source]
    cats = list(manifest['category_order'])
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError('1000 benzersiz çift bulunamadı')
    scores = {}
    resource_totals = {}
    for name in MODELS:
        model_rows = []
        metric_list = []
        for idx in range(20):
            file = OUT / name / f'chunk_{idx:02d}.jsonl'
            metric_file = OUT / name / f'chunk_{idx:02d}.metrics.json'
            if not file.is_file() or not metric_file.is_file():
                raise FileNotFoundError(f'{name}: parça {idx + 1}/20 veya metrik dosyası eksik: {file}')
            part = read_jsonl(file)
            if len(part) != 50:
                raise ValueError(f'{file}: 50 çift bekleniyor')
            for original, result in zip(source[idx * 50:(idx + 1) * 50], part):
                if (original['pair_id'] != result['pair_id'] or original['category'] != result['category'] or
                        result['dataset_sha256'] != dataset_sha or result['model'] != name or result['method'] != METHOD):
                    raise ValueError(f'Örnek/model/yöntem eşleşmedi: {file}')
            model_rows.extend(part)
            metrics = json.loads(metric_file.read_text(encoding='utf-8'))
            if (metrics['dataset_sha256'] != dataset_sha or metrics['model'] != name or
                    metrics['method'] != METHOD or metrics['chunk_index'] != idx):
                raise ValueError(f'Metrik dosyası eşleşmedi: {metric_file}')
            metric_list.append(metrics)
        if len(model_rows) != 1000 or len({r['pair_id'] for r in model_rows}) != 1000:
            raise ValueError(f'{name} için 1000 benzersiz çift bulunamadı')
        scores[name] = {r['pair_id']: r for r in model_rows}
        resource_totals[name] = {
            'score_seconds_sum': sum(m['score_seconds_chunk'] for m in metric_list),
            'mlx_allocator_peak_gb_max': max((m['mlx_allocator_peak_gb_decimal_chunk'] or 0) for m in metric_list),
            'process_rss_gb_max': max((m['process_peak_rss_gb_decimal_so_far'] or 0) for m in metric_list),
            'total_scored_tokens': sum(m['scored_tokens_chunk'] for m in metric_list),
            'note': 'Time excludes model load/warmup; RSS high water mark by process, not device total.'
        }
    for pid in ids:
        fp = scores['fp16'][pid]
        for name in ('q8', 'q4'):
            other = scores[name][pid]
            for side in ('good', 'bad'):
                if (fp[side]['token_ids_sha256'] != other[side]['token_ids_sha256'] or
                        fp[side]['n_tokens'] != other[side]['n_tokens']):
                    raise ValueError(f'{pid}: {name} tokenizer farkı')
    summary = {
        'dataset_sha256': dataset_sha, 'method': METHOD, 'n_pairs': 1000,
        'models': {}, 'categories': {}, 'transitions_from_fp16': {},
        'resource_summary': resource_totals,
    }
    print('ANA DENEY DOSYA VE TOKEN KONTROLLERİ BAŞARILI: 1000 çift × 3 model')
    print('\nGENEL DOĞRULUK')
    for name in MODELS:
        hits = sum(scores[name][pid]['correct'] for pid in ids)
        ties = sum(scores[name][pid]['tie'] for pid in ids)
        summary['models'][name] = {'correct': hits, 'total': 1000, 'accuracy': hits / 1000, 'ties': ties}
        print(f'{name.upper():4s}: {hits}/1000 (%{hits / 10:.1f}), eşit={ties}')
    print('\nKATEGORİ BAZINDA DOĞRULUK')
    for cat in cats:
        group = [pid for pid in ids if scores['fp16'][pid]['category'] == cat]
        if len(group) != 250:
            raise ValueError(f'{cat} için 250 çift bekleniyor')
        summary['categories'][cat] = {}
        counts = []
        for name in MODELS:
            hits = sum(scores[name][pid]['correct'] for pid in group)
            summary['categories'][cat][name] = {'correct': hits, 'total': 250, 'accuracy': hits / 250}
            counts.append(f'{name.upper()}={hits}/250')
        print(cat + ': ' + ' '.join(counts))
    print('\nFP16 REFERANSINA GÖRE EŞLEŞMİŞ KARAR GEÇİŞLERİ')
    for name in ('q8', 'q4'):
        down = [pid for pid in ids if scores['fp16'][pid]['correct'] and not scores[name][pid]['correct']]
        up = [pid for pid in ids if not scores['fp16'][pid]['correct'] and scores[name][pid]['correct']]
        summary['transitions_from_fp16'][name] = {
            'correct_to_incorrect': down, 'incorrect_to_correct': up,
            'n_down': len(down), 'n_up': len(up), 'net_percentage_points': 100 * (len(up) - len(down)) / 1000
        }
        print(f'{name.upper()}: doğru->yanlış {len(down)} | yanlış->doğru {len(up)} | '
              f'net {100*(len(up)-len(down))/1000:+.1f} yüzde puan')
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'main_comparison.json'
    tmp = target.with_suffix(target.suffix + '.tmp')
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(target)
    print(f'\nÖzet kaydedildi: {target.relative_to(ROOT)}')
    print('Not: Güven aralıkları, anlamlılık testleri ve ayrıntılı hata analizi sonraki aşamada yapılacak.')

if __name__ == '__main__':
    main()
