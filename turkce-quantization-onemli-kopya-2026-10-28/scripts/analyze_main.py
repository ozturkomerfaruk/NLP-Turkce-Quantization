#!/usr/bin/env python3
"""Stage 5: statistical analysis of the *saved* TurBLiMP main experiment.

Runs with Python 3.12 standard library only; NEVER loads or executes an MLX model.
Place beside score_main.py and summarize_main.py under scripts/.

Methods:
  * Primary: exact two-sided McNemar for two prespecified comparisons vs FP16;
    Holm correction across two primary tests.
  * Paired accuracy difference: conservative >=95% confidence interval based
    on simultaneous Clopper-Pearson intervals for the two discordant rates,
    with Bonferroni correction. (NOT an independent-groups CI.)
  * Category-specific results and total-logp sensitivity are descriptive.
  * Qualitative review: 20 stratified, reproducibly selected decision switches.
"""

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from score_main import METHOD, MODELS, TIE_TOLERANCE, read_jsonl, sha256

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'data' / 'processed' / 'turblimp_1000.jsonl'
MANIFEST = ROOT / 'data' / 'processed' / 'manifest.json'
RESULTS = ROOT / 'results' / 'main'
OUTPUT = ROOT / 'results' / 'analysis'
SEED = 42
EXPECTED_N = 1000
MODEL_ORDER = ('fp16', 'q8', 'q4')
COMPARISONS = ('q8', 'q4')


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def binom_cdf(k, n, prob):
    """P(X <= k) for X~Binomial(n,prob), robust for small k (discordances)."""
    if k < 0:
        return 0.0
    if k >= n or prob <= 0:
        return 1.0
    if prob >= 1:
        return 0.0
    v = math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) -
                 math.lgamma(n - k + 1) + k * math.log(prob) +
                 (n - k) * math.log1p(-prob))
    total = v
    for j in range(k, 0, -1):
        v *= j * (1 - prob) / ((n - j + 1) * prob)
        total += v
    return min(1.0, max(0.0, total))


def clopper_pearson(k, n, alpha):
    """Two-sided exact binomial CI with noncoverage <= alpha."""
    if not (0 <= k <= n and n > 0 and 0 < alpha < 1):
        raise ValueError('Invalid binomial CI inputs')
    if k == 0:
        lower = 0.0
    else:
        left, right = 0.0, 1.0
        for _ in range(65):
            mid = (left + right) / 2
            if binom_cdf(k - 1, n, mid) > 1 - alpha / 2:
                left = mid
            else:
                right = mid
        lower = (left + right) / 2
    if k == n:
        upper = 1.0
    else:
        left, right = 0.0, 1.0
        for _ in range(65):
            mid = (left + right) / 2
            if binom_cdf(k, n, mid) > alpha / 2:
                left = mid
            else:
                right = mid
        upper = (left + right) / 2
    return lower, upper


def paired_difference_ci(down, up, n):
    """Conservative at-least-95% CI for delta=(up-down)/n.

    Each of p_up and p_down is a binomial marginal over the *same* n pairs.
    CP 97.5% per marginal, combined by Bonferroni => joint >=95%.
    Subtraction bounds therefore cover the paired risk difference >=95%.
    """
    down_lo, down_hi = clopper_pearson(down, n, alpha=0.025)
    up_lo, up_hi = clopper_pearson(up, n, alpha=0.025)
    return [up_lo - down_hi, up_hi - down_lo]


def mcnemar_exact(down, up):
    """Two-sided exact conditional McNemar, binomial tail under p=0.5."""
    discordant = down + up
    if discordant == 0:
        return 1.0
    k = min(down, up)
    # Integer arithmetic ensures stable exact tail for n<=1000.
    numerator = 2 * sum(math.comb(discordant, x) for x in range(k + 1))
    denominator = 2 ** discordant
    return min(1.0, numerator / denominator)


def holm_adjust(pvalues):
    """Holm family-wise error correction; each key is a primary comparison."""
    ordered = sorted(pvalues, key=pvalues.get)
    count = len(ordered)
    adjusted = {}
    cumulative = 0.0
    for rank, key in enumerate(ordered):
        cumulative = max(cumulative, min(1.0, (count - rank) * pvalues[key]))
        adjusted[key] = cumulative
    return adjusted


def load_data():
    if not (DATASET.is_file() and MANIFEST.is_file()):
        raise FileNotFoundError('Ana veri seti/manifest bulunamadı.')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    data_sha = sha256(DATASET)
    if data_sha != manifest['main_dataset']['sha256']:
        raise ValueError('Manifest ana veri SHA-256 uyumsuzluğu')
    source = read_jsonl(DATASET)
    ids = [r['pair_id'] for r in source]
    if len(ids) != EXPECTED_N or len(set(ids)) != EXPECTED_N:
        raise ValueError('Ana veri: 1000 benzersiz çift bekleniyor')
    cats = manifest['category_order']
    if Counter(r['category'] for r in source) != Counter({c: 250 for c in cats}):
        raise ValueError('Dört kategori için 250 çift bekleniyor')
    scores = {}
    metrics_by_model = {}
    for name in MODEL_ORDER:
        results, chunk_metrics = [], []
        for idx in range(20):
            filename = RESULTS / name / f'chunk_{idx:02d}.jsonl'
            metric_file = RESULTS / name / f'chunk_{idx:02d}.metrics.json'
            if not filename.is_file() or not metric_file.is_file():
                raise FileNotFoundError(f'{name}, {idx:02d}: eksik parça veya metrik dosyası')
            part = read_jsonl(filename)
            if len(part) != 50:
                raise ValueError(f'{filename}: 50 çift bekleniyor')
            for original, item in zip(source[idx * 50:(idx + 1) * 50], part):
                if (item['pair_id'] != original['pair_id'] or
                    item['category'] != original['category'] or
                    item['dataset_sha256'] != data_sha or
                    item['method'] != METHOD or item['model'] != name):
                    raise ValueError(f'Çift / yöntem / veri kaynağı uyuşmazlığı: {filename}')
                margin = item['good']['mean_logp'] - item['bad']['mean_logp']
                if not (math.isfinite(margin) and math.isclose(margin, item['mean_logp_diff'], abs_tol=1e-9)):
                    raise ValueError(f'Saklanan puan farkı tutarsız: {item["pair_id"]}')
                expected_tie = abs(margin) <= TIE_TOLERANCE
                if item['tie'] != expected_tie or item['correct'] != (margin > TIE_TOLERANCE):
                    raise ValueError(f'Saklanan karar tutarsız: {item["pair_id"]}')
            results.extend(part)
            met = json.loads(metric_file.read_text(encoding='utf-8'))
            if (met['model'] != name or met['chunk_index'] != idx or
                met['dataset_sha256'] != data_sha or met['method'] != METHOD):
                raise ValueError(f'Metrik uyuşmazlığı: {metric_file}')
            chunk_metrics.append(met)
        scores[name] = {r['pair_id']: r for r in results}
        if len(scores[name]) != EXPECTED_N:
            raise ValueError(f'{name}: kayıt/ID tekrarları')
        metrics_by_model[name] = chunk_metrics

    for pid in ids:
        fp = scores['fp16'][pid]
        for name in COMPARISONS:
            comp = scores[name][pid]
            for side in ('good', 'bad'):
                if (fp[side]['token_ids_sha256'] != comp[side]['token_ids_sha256'] or
                    fp[side]['n_tokens'] != comp[side]['n_tokens']):
                    raise ValueError(f'{pid}: {name} token ID uyuşmazlığı')
    summary_file = RESULTS / 'main_comparison.json'
    if not summary_file.is_file():
        raise FileNotFoundError('Önce python scripts/summarize_main.py çalıştır.')
    existing_summary = json.loads(summary_file.read_text(encoding='utf-8'))
    if existing_summary['dataset_sha256'] != data_sha or existing_summary['method'] != METHOD:
        raise ValueError('main_comparison.json farklı veri veya yöntem kullanıyor')
    for name in MODEL_ORDER:
        count = sum(int(scores[name][pid]['correct']) for pid in ids)
        if existing_summary['models'][name]['correct'] != count:
            raise ValueError(f'{name} özet doğruluk uyuşmazlığı')
    return source, ids, cats, scores, metrics_by_model, data_sha


def comparison(pid_list, scores, model):
    n = len(pid_list)
    fp = scores['fp16']
    other = scores[model]
    down = [pid for pid in pid_list if fp[pid]['correct'] and not other[pid]['correct']]
    up = [pid for pid in pid_list if not fp[pid]['correct'] and other[pid]['correct']]
    both_correct = sum(fp[pid]['correct'] and other[pid]['correct'] for pid in pid_list)
    both_wrong = sum((not fp[pid]['correct']) and (not other[pid]['correct']) for pid in pid_list)
    lower, upper = paired_difference_ci(len(down), len(up), n)
    return {
        'n_pairs': n, 'both_correct': both_correct, 'both_wrong': both_wrong,
        'correct_to_wrong': len(down), 'wrong_to_correct': len(up),
        'delta_accuracy': (len(up) - len(down)) / n,
        'paired_ci95_conservative': [lower, upper],
        'mcnemar_exact_p_unadjusted': mcnemar_exact(len(down), len(up)),
        'down_ids': down, 'up_ids': up,
    }


def review_selection(comps):
    rng = random.Random(SEED)
    picked = []
    used = set()
    strata = (
        ('Q4 doğru->yanlış', comps['q4']['down_ids'], 12),
        ('Q4 yanlış->doğru', comps['q4']['up_ids'], 5),
        ('Q8 yanlış->doğru', comps['q8']['up_ids'], 3),
    )
    for label, population, target in strata:
        candidates = [p for p in population if p not in used]
        for pid in rng.sample(candidates, min(target, len(candidates))):
            used.add(pid)
            picked.append((pid, label))
    if len(picked) < 20:
        remaining = [pid for pid in comps['q4']['down_ids'] + comps['q4']['up_ids'] +
                     comps['q8']['up_ids'] if pid not in used]
        for pid in rng.sample(remaining, min(20 - len(picked), len(remaining))):
            picked.append((pid, 'Ek karar değişimi'))
            used.add(pid)
    return picked


def main():
    source, ids, categories, scores, metrics, data_sha = load_data()
    source_by_id = {r['pair_id']: r for r in source}
    comps = {name: comparison(ids, scores, name) for name in COMPARISONS}
    corrected = holm_adjust({name: comps[name]['mcnemar_exact_p_unadjusted'] for name in COMPARISONS})
    counts = {name: sum(int(scores[name][pid]['correct']) for pid in ids) for name in MODEL_ORDER}
    print('ANA DENEY İSTATİSTİKSEL ANALİZİ | 1000 eşleştirilmiş çift')
    print('FP16: {}/1000 | Q8: {}/1000 | Q4: {}/1000'.format(*(counts[n] for n in MODEL_ORDER)))

    report = {
        'dataset_sha256': data_sha, 'scoring_method': METHOD, 'analysis_seed': SEED,
        'n_pairs': len(ids),
        'primary_methods': {
            'test': 'two-sided exact McNemar (conditional binomial; discordant pairs)',
            'multiplicity': 'Holm correction for 2 prespecified primary comparisons vs FP16',
            'ci': 'conservative >=95% paired difference CI by two 97.5% Clopper-Pearson marginal CIs for discordant event rates + Bonferroni, then interval subtraction',
            'unit': 'same TurBLiMP minimal pair assessed by both models',
            'notes': 'CIs describe sampling uncertainty conditional on model/dataset construction; no claim of broad Turkish-language generalization.',
        },
        'model_accuracy': {name: {'n_correct': counts[name], 'n_pairs': len(ids), 'accuracy': counts[name] / len(ids)} for name in MODEL_ORDER},
        'primary_comparisons': {}, 'category_descriptive': {},
        'margin_diagnostics': {}, 'total_logp_sensitivity': {}, 'resource_summary': {},
    }

    for name in COMPARISONS:
        c = comps[name]
        report['primary_comparisons'][name] = {
            k: v for k, v in c.items() if k not in ('down_ids', 'up_ids')
        }
        report['primary_comparisons'][name]['mcnemar_p_holm'] = corrected[name]
        ci = c['paired_ci95_conservative']
        print(f'{name.upper()} vs FP16 | fark {100*c["delta_accuracy"]:+.2f} yp | '
              f'95% eşleşmiş GA [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] yp | '
              f'↓{c["correct_to_wrong"]} ↑{c["wrong_to_correct"]} | '
              f'McNemar p={c["mcnemar_exact_p_unadjusted"]:.7f}, Holm={corrected[name]:.7f}')

    category_csv = []
    for cat in categories:
        cat_ids = [r['pair_id'] for r in source if r['category'] == cat]
        model_acc = {n: sum(int(scores[n][p]['correct']) for p in cat_ids) / len(cat_ids) for n in MODEL_ORDER}
        cat_entry = {'n_pairs': len(cat_ids), 'accuracy': model_acc, 'comparisons': {}}
        row = {'category': cat, 'n_pairs': len(cat_ids),
               **{f'{n}_accuracy_percent': round(model_acc[n] * 100, 4) for n in MODEL_ORDER}}
        for name in COMPARISONS:
            c = comparison(cat_ids, scores, name)
            cat_entry['comparisons'][name] = {k: v for k, v in c.items() if k not in ('down_ids', 'up_ids')}
            row.update({
                f'{name}_delta_pp': round(c['delta_accuracy'] * 100, 4),
                f'{name}_ci_low_pp': round(c['paired_ci95_conservative'][0] * 100, 4),
                f'{name}_ci_high_pp': round(c['paired_ci95_conservative'][1] * 100, 4),
                f'{name}_correct_to_wrong': c['correct_to_wrong'],
                f'{name}_wrong_to_correct': c['wrong_to_correct'],
            })
        report['category_descriptive'][cat] = cat_entry
        category_csv.append(row)
        print(f'  {cat:34s} | FP16 {model_acc["fp16"]*100:5.1f}% | '
              f'Q8 {model_acc["q8"]*100:5.1f}% | Q4 {model_acc["q4"]*100:5.1f}%')
    write_csv(OUTPUT / 'category_results.csv', category_csv, list(category_csv[0]))

    # All model decisions under alternative total-logp scoring (not the primary outcome).
    for name in MODEL_ORDER:
        total_hits = sum(scores[name][pid]['total_logp_diff'] > TIE_TOLERANCE for pid in ids)
        disagreement = sum((scores[name][pid]['total_logp_diff'] > TIE_TOLERANCE) != scores[name][pid]['correct'] for pid in ids)
        report['total_logp_sensitivity'][name] = {
            'total_logp_accuracy': total_hits / len(ids),
            'total_logp_correct': total_hits,
            'total_vs_mean_decision_disagreements': disagreement,
        }

    all_abs = sorted((abs(scores['fp16'][pid]['mean_logp_diff']), pid) for pid in ids)
    quartiles = [all_abs[i * 250:(i + 1) * 250] for i in range(4)]
    for name in COMPARISONS:
        fp = scores['fp16']
        comp = scores[name]
        changed = [pid for pid in ids if fp[pid]['correct'] != comp[pid]['correct']]
        unchanged = [pid for pid in ids if fp[pid]['correct'] == comp[pid]['correct']]
        diag = {
            'n_changed': len(changed),
            'mean_abs_fp16_margin_changed': (sum(abs(fp[p]['mean_logp_diff']) for p in changed) / len(changed) if changed else None),
            'mean_abs_fp16_margin_unchanged': sum(abs(fp[p]['mean_logp_diff']) for p in unchanged) / len(unchanged),
            'flip_rate_by_fp16_abs_margin_quartile': []
        }
        for i, bucket in enumerate(quartiles, 1):
            bucket_ids = [pid for _, pid in bucket]
            flips = sum(fp[p]['correct'] != comp[p]['correct'] for p in bucket_ids)
            diag['flip_rate_by_fp16_abs_margin_quartile'].append({
                'quartile': i, 'n_pairs': len(bucket_ids), 'n_flips': flips,
                'flip_rate': flips / len(bucket_ids),
                'fp16_abs_margin_min': bucket[0][0], 'fp16_abs_margin_max': bucket[-1][0],
            })
        report['margin_diagnostics'][name] = diag

    for name in MODEL_ORDER:
        items = metrics[name]
        model_dir = ROOT / 'models' / MODELS[name]
        files = list(model_dir.rglob('*.safetensors'))
        report['resource_summary'][name] = {
            'safetensors_bytes': sum(f.stat().st_size for f in files) if files else None,
            'safetensors_gb_decimal': sum(f.stat().st_size for f in files) / 1e9 if files else None,
            'scoring_seconds_sum_excludes_load_and_warmup': sum(m['score_seconds_chunk'] for m in items),
            'mlx_peak_allocator_gb_decimal_across_chunks': max((m['mlx_allocator_peak_gb_decimal_chunk'] or 0) for m in items),
            'process_peak_rss_gb_decimal_across_chunks': max((m['process_peak_rss_gb_decimal_so_far'] or 0) for m in items),
            'scored_tokens': sum(m['scored_tokens_chunk'] for m in items),
            'note': 'MLX allocator and process RSS are distinct. Time excludes model loading, warmup, pauses and file writing.',
        }

    review_rows = []
    for pid, stratum in review_selection(comps):
        original = source_by_id[pid]
        rr = {'pair_id': pid, 'category': original['category'], 'review_stratum': stratum,
              'sentence_good': original['sentence_good'], 'sentence_bad': original['sentence_bad'],
              'good_cue': original.get('good_cue', ''), 'bad_cue': original.get('bad_cue', ''),
              'critical_region': original.get('critical_region', ''),
              'fp16_margin': scores['fp16'][pid]['mean_logp_diff'],
              'q8_margin': scores['q8'][pid]['mean_logp_diff'],
              'q4_margin': scores['q4'][pid]['mean_logp_diff'],
              'fp16_correct': scores['fp16'][pid]['correct'],
              'q8_correct': scores['q8'][pid]['correct'],
              'q4_correct': scores['q4'][pid]['correct'],
              'manual_review_note': '', 'linguistic_validity': '',
        }
        review_rows.append(rr)
    write_csv(OUTPUT / 'manual_review_20.csv', review_rows, list(review_rows[0]))

    detail_rows = []
    for pid in ids:
        d = source_by_id[pid]
        r = {'pair_id': pid, 'category': d['category'],
             'sentence_good': d['sentence_good'], 'sentence_bad': d['sentence_bad']}
        for name in MODEL_ORDER:
            v = scores[name][pid]
            r[f'{name}_correct'] = int(v['correct'])
            r[f'{name}_mean_diff'] = v['mean_logp_diff']
            r[f'{name}_total_diff'] = v['total_logp_diff']
            r[f'{name}_good_token_n'] = v['good']['n_tokens']
            r[f'{name}_bad_token_n'] = v['bad']['n_tokens']
        detail_rows.append(r)
    write_csv(OUTPUT / 'per_pair_scores.csv', detail_rows, list(detail_rows[0]))

    write_json(OUTPUT / 'main_statistics.json', report)
    print('\nDosyalar:')
    for filename in ('main_statistics.json', 'category_results.csv', 'manual_review_20.csv', 'per_pair_scores.csv'):
        print('  ' + str((OUTPUT / filename).relative_to(ROOT)))
    print('BİTTİ: yalnızca kayıtlı puanlar kullanıldı; model yüklenmedi.')
    print('UYARI: 20 örnek amaçlı/katmanlı seçilmiştir; temsilî rastgele örneklem değildir.')


if __name__ == '__main__':
    main()
