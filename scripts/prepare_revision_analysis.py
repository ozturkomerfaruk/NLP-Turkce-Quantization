#!/usr/bin/env python3
"""Export one aligned row per pair from the verified revision checkpoints."""

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

from score_main import ROOT, sha256
from score_revision import FAMILIES, PRECISIONS, RUN_DIRECTORIES, load_dataset, read_json

PREFIXES = frozenset(
    "ama ve evet çünkü fakat yani ancak işte şimdi yalnız ayrıca sonra ardından artık "
    "bugün burada aslında belki öncelikle yine özellikle hadi sadece hemen".split()
)
METHODS = ("mean", "total")
ANALYSIS_INPUT = ROOT / "outputs/revision_v2/statistics-input.csv"


def folded(text):
    return text.casefold().replace("\u0307", "")


def cluster_key(row):
    """Use the shared-prefix clustering rule from the submitted experiment."""
    good = row["sentence_good"].split()
    bad = row["sentence_bad"].split()
    while len(good) > 1 and len(bad) > 1 and folded(good[0]) == folded(bad[0]) and folded(good[0]) in PREFIXES:
        good = good[1:]
        bad = bad[1:]
    return row["category"], folded(" ".join(good)), folded(" ".join(bad))


def verified_records(audit, model, precision, results_dir=RUN_DIRECTORIES["main"]):
    run = f"{model}/{precision}"
    report = audit["runs"][run]
    directory = results_dir / run
    expected = report["checkpoint_sha256"]
    if len(expected) != 74 or set(expected) != {path.name for path in directory.glob("chunk_*.json")}:
        raise ValueError(f"{run}: expected 74 verified checkpoints")
    for index, name in enumerate(sorted(expected)):
        path = directory / name
        if sha256(path) != expected[name]:
            raise ValueError(f"{path}: checkpoint differs from the Stage 8 audit")
        chunk = read_json(path)
        identity = chunk["identity"]
        if (
            chunk["chunk_index"] != index or identity["model"] != model
            or identity["precision"] != precision or identity["mode"] != "main"
            or identity["dataset_sha256"] != audit["dataset_sha256"]
            or identity["method"] != audit["method"]
            or identity["source_revision"] != report["source_revision"]
            or identity["model_files"]["model.safetensors"] != report["weights_sha256"]
            or identity["code_sha256"] != audit["scoring_code_sha256"]
        ):
            raise ValueError(f"{path}: incompatible checkpoint identity")
        yield from chunk["records"]


def build_analysis_rows():
    dataset, digest = load_dataset()
    audit = read_json(RUN_DIRECTORIES["main"] / "verification.json")
    if audit["dataset_sha256"] != digest or audit["total_pair_evaluations"] != 6 * len(dataset):
        raise ValueError("The Stage 8 audit does not match the revision dataset")

    cluster_ids = {}
    rows = []
    for source in dataset:
        key = cluster_key(source)
        cluster = cluster_ids.setdefault(key, len(cluster_ids) + 1)
        rows.append({
            "pair_id": source["pair_id"], "category": source["category"],
            "in_baseline": source["in_baseline"], "cluster": cluster,
        })

    for model in FAMILIES:
        reference = []
        for precision in PRECISIONS:
            for index, (row, score) in enumerate(zip(rows, verified_records(audit, model, precision), strict=True)):
                for key in ("pair_id", "category", "in_baseline"):
                    if row[key] != score[key]:
                        raise ValueError(f"{model}/{precision}: paired record does not match {row['pair_id']}")
                tokens = tuple((score[side]["n_tokens"], score[side]["token_ids_sha256"]) for side in ("good", "bad"))
                if precision == "fp16":
                    reference.append(tokens)
                    row[f"{model}_length_difference"] = tokens[0][0] - tokens[1][0]
                elif tokens != reference[index]:
                    raise ValueError(f"{model}/{precision}: tokenization changed for {row['pair_id']}")
                for method in METHODS:
                    difference = score[f"{method}_logp_diff"]
                    if not math.isfinite(difference):
                        raise ValueError(f"{model}/{precision}: non-finite score for {row['pair_id']}")
                    decision = int(difference > 0) - int(difference < 0)
                    if score[f"{method}_correct"] != (decision == 1) or score[f"{method}_tie"] != (decision == 0):
                        raise ValueError(f"{model}/{precision}: invalid decision for {row['pair_id']}")
                    row[f"{model}_{precision}_{method}"] = decision
                if tokens[0][0] == tokens[1][0]:
                    if row[f"{model}_{precision}_mean"] != row[f"{model}_{precision}_total"]:
                        raise ValueError(f"{model}/{precision}: Mean and Total rankings differ at equal token length")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ANALYSIS_INPUT)
    output = parser.parse_args().output
    rows = build_analysis_rows()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({**row, "in_baseline": int(row["in_baseline"])} for row in rows)
    os.replace(stream.name, output)
    print(f"{len(rows)} verified pairs, {len({row['cluster'] for row in rows})} clusters: {output}")


if __name__ == "__main__":
    main()
