#!/usr/bin/env python3
"""Verify all six revision score sets against the frozen data and model files."""

import itertools
import math

from score_main import ROOT, atomic_json, sha256
from score_revision import (
    CHUNK_SIZES, FAMILIES, METHOD, PRECISIONS, RUN_DIRECTORIES, SIDES,
    bos_context, expected_tokens, load_dataset, read_json, validate_chunk,
    verified_model,
)


def verify_run(family, precision, rows, expected, identity):
    directory = RUN_DIRECTORIES["main"] / family / precision
    size = CHUNK_SIZES["main"]
    starts = range(0, len(rows), size)
    paths = [directory / f"chunk_{index:03d}.json" for index in range(len(starts))]
    if set(directory.glob("chunk_*.json")) != set(paths):
        raise ValueError(f"{family}/{precision}: missing or unexpected checkpoints")

    manifest = {}
    scored_sentences = 0
    for index, (path, start) in enumerate(zip(paths, starts, strict=True)):
        checkpoint = read_json(path)
        validate_chunk(checkpoint, identity, index, rows[start:start + size], expected[start:start + size])
        records = checkpoint["records"]
        tokens = sum(record[side]["n_tokens"] for record, side in itertools.product(records, SIDES))
        metrics = checkpoint["metrics"]
        if (
            metrics["pairs_in_chunk"] != len(records)
            or metrics["scored_tokens_chunk"] != tokens
            or not math.isfinite(metrics["score_seconds_chunk"])
            or metrics["score_seconds_chunk"] <= 0
        ):
            raise ValueError(f"{path}: checkpoint metrics do not match its records")
        scored_sentences += len(records) * len(SIDES)
        manifest[path.name] = sha256(path)
    return {
        "pairs": len(rows),
        "scored_sentences": scored_sentences,
        "chunks": len(paths),
        "weights_sha256": identity["model_files"]["model.safetensors"],
        "source_revision": identity["source_revision"],
        "bos_context_id": identity["bos_context_id"],
        "runtime": identity["runtime"],
        "checkpoint_sha256": manifest,
    }


def main():
    from mlx_lm.utils import load_tokenizer

    rows, dataset_sha = load_dataset()
    code = {
        name: sha256(ROOT / "scripts" / name)
        for name in ("score_main.py", "score_revision.py")
    }
    runs = {}
    runtime = None
    for family in FAMILIES:
        model_manifest = read_json(ROOT / "docs" / f"{family}_revision_manifest.json")
        reference, reference_hashes = verified_model(model_manifest, family, "fp16")
        config = read_json(reference / "config.json")
        tokenizer = load_tokenizer(reference, eos_token_ids=config.get("eos_token_id"))
        bos = bos_context(family, config, tokenizer)
        expected = expected_tokens(rows, tokenizer, bos, config["max_position_embeddings"])

        for precision in PRECISIONS:
            run = f"{family}/{precision}"
            _, model_hashes = (
                (reference, reference_hashes) if precision == "fp16"
                else verified_model(model_manifest, family, precision)
            )
            first = RUN_DIRECTORIES["main"] / run / "chunk_000.json"
            if runtime is None:
                runtime = read_json(first)["identity"]["runtime"]
            identity = {
                "model": family, "precision": precision, "mode": "main", "method": METHOD,
                "dataset_sha256": dataset_sha, "source_revision": model_manifest["source_revision"],
                "model_files": model_hashes, "fp16_reference_files": reference_hashes,
                "bos_context_id": bos, "runtime": runtime, "code_sha256": code,
            }
            runs[run] = verify_run(family, precision, rows, expected, identity)
            print(f"{run}: {runs[run]['pairs']} pairs, {runs[run]['chunks']} checkpoints verified", flush=True)

    report = {
        "dataset_sha256": dataset_sha,
        "method": METHOD,
        "scoring_code_sha256": code,
        "runs": runs,
        "total_pair_evaluations": sum(run["pairs"] for run in runs.values()),
        "total_sentence_evaluations": sum(run["scored_sentences"] for run in runs.values()),
    }
    path = RUN_DIRECTORIES["main"] / "verification.json"
    atomic_json(path, report)
    print(
        f"Verified {report['total_pair_evaluations']} pair evaluations, "
        f"{report['total_sentence_evaluations']} sentence scores. Report: {path.relative_to(ROOT)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
