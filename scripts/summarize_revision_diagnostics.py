#!/usr/bin/env python3
"""Summarize saved resource measurements and reproduce the predicate screen."""

import csv
import math
from collections import Counter

from prepare_revision_analysis import METHODS, build_analysis_rows
from score_main import ROOT, sha256
from score_revision import FAMILIES, PRECISIONS, RUN_DIRECTORIES, load_dataset, read_json

PREDICATES = frozenset(("şaşırdı", "tepki gösterdi", "inanıyor"))
RESULTS = RUN_DIRECTORIES["main"]
OUTPUT = RESULTS / "diagnostics"


def write_csv(name, rows):
    path = OUTPUT / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def screen(rows):
    return [
        row for row in rows
        if row["category"] == "nominalization" and row["critical_region"] in PREDICATES
    ]


def resources():
    verification = read_json(RESULTS / "verification.json")
    summaries = []
    for family in FAMILIES:
        model = read_json(ROOT / "docs" / f"{family}_revision_manifest.json")
        for precision in PRECISIONS:
            name = f"{family}/{precision}"
            report = verification["runs"][name]
            weights = model["models"][precision]["files"]["model.safetensors"]
            if weights["sha256"] != report["weights_sha256"]:
                raise ValueError(f"Weights do not match the run manifest: {name}")
            metrics = []
            for filename, digest in sorted(report["checkpoint_sha256"].items()):
                path = RESULTS / name / filename
                if sha256(path) != digest:
                    raise ValueError(f"Expected the verified checkpoint: {path}")
                metrics.append(read_json(path)["metrics"])
            rss = [row["process_peak_rss_bytes_so_far"] for row in metrics]
            pairs = sum(row["pairs_in_chunk"] for row in metrics)
            if (pairs != 3662 or rss != sorted(rss)
                    or len({row["model_load_seconds_this_process"] for row in metrics}) != 1):
                raise ValueError(f"Unexpected process boundaries or pair count: {name}")
            seconds = math.fsum(row["score_seconds_chunk"] for row in metrics)
            tokens = sum(row["scored_tokens_chunk"] for row in metrics)
            summaries.append({
                "model": family, "precision": precision, "pairs": pairs,
                "scoring_seconds": seconds, "scored_tokens": tokens,
                "tokens_per_second": tokens / seconds,
                "weights_bytes": weights["bytes"],
                "mlx_active_peak_bytes": max(row["mlx_allocator_peak_bytes_chunk"] for row in metrics),
                "process_peak_rss_bytes": max(rss),
            })
    return summaries


def sensitivity(rows, flagged_ids):
    groups = {
        "full_3662": rows,
        "old_1000": [row for row in rows if row["in_baseline"]],
        "new_2662": [row for row in rows if not row["in_baseline"]],
    }
    summaries = []
    for group, sample in groups.items():
        for scenario, selected in (
            ("all_pairs", sample),
            ("predicate_screen_excluded", [row for row in sample if row["pair_id"] not in flagged_ids]),
        ):
            for model in FAMILIES:
                for method in METHODS:
                    baseline = [row[f"{model}_fp16_{method}"] == 1 for row in selected]
                    for precision in PRECISIONS:
                        current = [row[f"{model}_{precision}_{method}"] == 1 for row in selected]
                        loss = sum(a and not b for a, b in zip(baseline, current, strict=True))
                        gain = sum(not a and b for a, b in zip(baseline, current, strict=True))
                        n = len(selected)
                        categories = Counter(row["category"] for row in selected)
                        correct = Counter(row["category"] for row, flag in zip(selected, current, strict=True) if flag)
                        summaries.append({
                            "sample": group, "scenario": scenario, "n": n, "excluded": len(sample) - n,
                            "model": model, "precision": precision, "method": method,
                            "correct": sum(current), "micro_pct": 100 * sum(current) / n,
                            "macro_pct": 100 * sum(correct[c] / count for c, count in categories.items()) / len(categories),
                            "delta_vs_fp16_pp": 100 * (gain - loss) / n,
                            "correct_to_incorrect": loss, "incorrect_to_correct": gain,
                        })
    return summaries


def main():
    source, _ = load_dataset()
    flagged = screen(source)
    with (ROOT / "results/deep_analysis/predicate_screen_all_pairs.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        historical = {row["pair_id"] for row in csv.DictReader(stream)}
    if len(historical) != 17 or {row["pair_id"] for row in flagged if row["in_baseline"]} != historical:
        raise ValueError("Predicate screen does not reproduce the 17 historical pair IDs")
    keys = ("pair_id", "category", "in_baseline", "source_record_index", "critical_region",
            "sentence_good", "sentence_bad")
    pairs = [{key: row[key] for key in keys} for row in flagged]
    metrics = resources()
    effects = sensitivity(build_analysis_rows(), {row["pair_id"] for row in flagged})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv("predicate_screen_pairs.csv", pairs)
    write_csv("predicate_sensitivity.csv", effects)
    write_csv("resource_measurements.csv", metrics)
    print(f"{len(metrics)} resource summaries, {len(flagged)} flagged pairs ({len(historical)} historical), "
          f"{len(effects)} descriptive sensitivity rows: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
