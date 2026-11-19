#!/usr/bin/env python3
"""Score the revision benchmark with resumable, provenance-checked chunks."""

import argparse
import fcntl
import json
import math
import platform
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from score_main import (
    ROOT, atomic_json, get_mlx_memory, package_version, read_jsonl,
    score_sentence, sha256, token_hash,
)

FAMILIES = ("eurollm", "qwen")
PRECISIONS = ("fp16", "q8", "q4")
SIDES = ("good", "bad")
DATASET_DIRECTORY = ROOT / "data/revision_v2"
DATASET_SHA256 = "f04ee6b48fa27056261e85741139f64a59e1c4078928f2855fdf381cc30c94e9"
RUN_DIRECTORIES = {
    "smoke": ROOT / "outputs/revision_v2/scoring_smoke",
    "main": ROOT / "results/revision_v2",
}
CHUNK_SIZES = {"smoke": 4, "main": 50}
METHOD = "mean_total_float32_bos_context_no_eos_v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset():
    manifest = read_json(DATASET_DIRECTORY / "manifest.json")
    path = DATASET_DIRECTORY / "turblimp_3662.jsonl"
    digest = sha256(path)
    if digest != DATASET_SHA256 or manifest["dataset"]["sha256"] != digest:
        raise ValueError(f"Unexpected revision dataset SHA-256: {path}")

    rows = read_jsonl(path)
    pairs = {(row["sentence_good"], row["sentence_bad"]) for row in rows}
    if len(rows) != 3662 or len({row["pair_id"] for row in rows}) != 3662 or len(pairs) != 3662:
        raise ValueError("Expected 3,662 distinct revision pairs and IDs")
    if any(not good or not bad or good == bad for good, bad in pairs):
        raise ValueError("The revision dataset contains an invalid sentence pair")

    if any(type(row["in_baseline"]) is not bool for row in rows):
        raise ValueError("Each revision pair must have a boolean baseline membership")
    counts = Counter(row["category"] for row in rows)
    baseline = Counter(row["category"] for row in rows if row["in_baseline"])
    if (
        counts != manifest["category_counts"]
        or baseline != manifest["baseline"]["category_counts"]
        or counts - baseline != manifest["previously_unscored"]["category_counts"]
    ):
        raise ValueError("Revision category counts or baseline membership do not match the manifest")
    return rows, digest


def select_rows(rows, mode):
    if mode == "main":
        return rows
    counts = Counter()
    selected = []
    for row in rows:
        category = row["category"]
        if row["in_baseline"] and counts[category] < 2:
            selected.append(row)
            counts[category] += 1
    if len(selected) != 8 or len(counts) != 4:
        raise ValueError("Expected two previously scored pairs in each of four categories")
    return selected


def verified_model(manifest, family, precision):
    path = ROOT / "models/revision_v2" / family / precision
    record = manifest["models"][precision]
    if record["path"] != path.relative_to(ROOT).as_posix():
        raise ValueError(f"Unexpected location in the model manifest: {path}")
    hashes = {}
    for filename, entry in record["files"].items():
        if Path(filename).name != filename:
            raise ValueError(f"Invalid model artifact filename: {filename}")
        file = path / filename
        digest = sha256(file)
        if digest != entry["sha256"]:
            raise ValueError(f"Model artifact differs from the verified build: {file}")
        hashes[filename] = digest
    required = {"model.safetensors", "model.safetensors.index.json", "config.json",
                "tokenizer.json", "tokenizer_config.json"}
    if required - hashes.keys() or {file.name for file in path.iterdir() if file.is_file()} != hashes.keys():
        raise ValueError(f"Unexpected model artifacts: {path}")
    return path, hashes


def bos_context(family, config, tokenizer):
    if family == "qwen":
        bos = config["bos_token_id"]
        if bos != 151643 or tokenizer.bos_token_id is not None or tokenizer.eos_token_id != bos:
            raise ValueError("The verified Qwen BOS context does not match the loaded tokenizer")
        return bos
    bos = tokenizer.bos_token_id
    if type(bos) is not int or bos != config.get("bos_token_id"):
        raise ValueError("The EuroLLM BOS context does not match the model configuration")
    return bos


def expected_tokens(rows, tokenizer, bos, context_limit):
    signatures = []
    for row in rows:
        pair = []
        for side in SIDES:
            ids = tokenizer.encode(row[f"sentence_{side}"], add_special_tokens=False)
            if not ids or bos in ids or len(ids) + 1 > context_limit:
                raise ValueError(f"Invalid tokenization for {row['pair_id']} ({side})")
            pair.append((len(ids), token_hash(ids)))
        signatures.append(tuple(pair))
    return signatures


def pair_scores(row, good, bad):
    record = {key: row[key] for key in ("pair_id", "category", "in_baseline")}
    record.update(good=good, bad=bad)
    for method in ("mean", "total"):
        difference = good[f"{method}_logp"] - bad[f"{method}_logp"]
        record[f"{method}_logp_diff"] = difference
        record[f"{method}_correct"] = difference > 0
        record[f"{method}_tie"] = difference == 0
    return record


def validate_chunk(checkpoint, identity, index, rows, signatures):
    records = checkpoint.get("records")
    if (
        checkpoint.get("identity") != identity or checkpoint.get("chunk_index") != index
        or not isinstance(records, list) or len(records) != len(rows)
    ):
        raise ValueError(f"Chunk {index}: unexpected identity or record count")

    for row, record, expected in zip(rows, records, signatures, strict=True):
        for key in ("pair_id", "category", "in_baseline"):
            if type(record.get(key)) is not type(row[key]) or record[key] != row[key]:
                raise ValueError(f"Chunk {index}: does not match expected pair {row['pair_id']}")
        for side, (n_tokens, token_digest) in zip(SIDES, expected, strict=True):
            stats = record.get(side, {})
            total = stats.get("total_logp")
            mean = stats.get("mean_logp")
            if (
                type(stats.get("n_tokens")) is not int or stats["n_tokens"] != n_tokens
                or stats.get("token_ids_sha256") != token_digest
                or any(type(value) not in (int, float) or not math.isfinite(value) or value > 0
                       for value in (total, mean))
                or not math.isclose(mean, total / n_tokens, rel_tol=1e-6, abs_tol=1e-7)
            ):
                raise ValueError(f"Chunk {index}: invalid score or tokenization for {row['pair_id']} ({side})")

        expected_record = pair_scores(row, record["good"], record["bad"])
        for method in ("mean", "total"):
            for suffix in ("logp_diff", "correct", "tie"):
                key = f"{method}_{suffix}"
                value = record.get(key)
                if type(value) is not type(expected_record[key]) or value != expected_record[key]:
                    raise ValueError(f"Chunk {index}: inconsistent {key} for {row['pair_id']}")


def pending_chunks(outdir, identity, rows, signatures):
    size = CHUNK_SIZES[identity["mode"]]
    chunks = [
        (index, rows[start:start + size], signatures[start:start + size])
        for index, start in enumerate(range(0, len(rows), size))
    ]
    expected_paths = {outdir / f"chunk_{index:03d}.json" for index, _, _ in chunks}
    unexpected = set(outdir.glob("chunk_*.json")) - expected_paths
    if unexpected:
        raise ValueError(f"Unexpected checkpoints in {outdir}: {sorted(unexpected)}")

    pending = []
    for index, pairs, tokens in chunks:
        path = outdir / f"chunk_{index:03d}.json"
        if path.exists():
            validate_chunk(read_json(path), identity, index, pairs, tokens)
        else:
            pending.append((path, index, pairs, tokens))
    return pending, len(chunks)


def run_pending(pending, identity, path, family, config, bos):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    started = time.perf_counter()
    model, tokenizer = load(str(path))
    model.eval()
    load_seconds = time.perf_counter() - started
    if bos_context(family, config, tokenizer) != bos:
        raise ValueError("Model BOS context differs from its FP16 reference")

    context_limit = config["max_position_embeddings"]

    def score(text):
        return score_sentence(text, model, tokenizer, mx, nn, context_limit, bos_token_id=bos)

    started = time.perf_counter()
    score(pending[0][2][0]["sentence_good"])
    mx.clear_cache()
    warmup_seconds = time.perf_counter() - started
    for checkpoint_path, index, rows, signatures in pending:
        mx.reset_peak_memory()
        started = time.perf_counter()
        records = []
        for row in rows:
            good, bad = (score(row[f"sentence_{side}"]) for side in SIDES)
            records.append(pair_scores(row, good, bad))
            mx.clear_cache()

        seconds = time.perf_counter() - started
        peak = get_mlx_memory(mx, "get_peak_memory")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if sys.platform == "darwin" else None
        checkpoint = {
            "identity": identity,
            "chunk_index": index,
            "metrics": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "pairs_in_chunk": len(records),
                "scored_tokens_chunk": sum(record[side]["n_tokens"] for record in records for side in SIDES),
                "score_seconds_chunk": seconds,
                "model_load_seconds_this_process": load_seconds,
                "warmup_seconds_this_process": warmup_seconds,
                "mlx_allocator_peak_bytes_chunk": peak,
                "process_peak_rss_bytes_so_far": rss,
            },
            "records": records,
        }
        validate_chunk(checkpoint, identity, index, rows, signatures)
        atomic_json(checkpoint_path, checkpoint)
        print(f"Chunk {index + 1}: {len(rows)} pairs saved ({seconds:.2f} s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=FAMILIES, required=True)
    parser.add_argument("--precision", choices=PRECISIONS, required=True)
    parser.add_argument("--mode", choices=RUN_DIRECTORIES, required=True)
    parser.add_argument("--max-chunks", type=int, default=1, help="Maximum number of new chunks")
    args = parser.parse_args()
    if args.max_chunks < 1:
        parser.error("--max-chunks must be positive")

    from mlx_lm.utils import load_tokenizer

    rows, dataset_hash = load_dataset()
    family, precision, mode = args.model, args.precision, args.mode
    manifest = read_json(ROOT / "docs" / f"{family}_revision_manifest.json")
    path, model_hashes = verified_model(manifest, family, precision)
    reference_path, reference_hashes = (
        (path, model_hashes) if precision == "fp16" else verified_model(manifest, family, "fp16")
    )
    config = read_json(path / "config.json")
    reference_config = read_json(reference_path / "config.json")
    tokenizer = load_tokenizer(reference_path, eos_token_ids=reference_config.get("eos_token_id"))
    bos = bos_context(family, reference_config, tokenizer)
    if config["max_position_embeddings"] != reference_config["max_position_embeddings"]:
        raise ValueError("Model context limit differs from its FP16 reference")

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{name: package_version(name) for name in ("mlx", "mlx-lm", "transformers", "tokenizers")},
    }
    identity = {
        "model": family, "precision": precision, "mode": mode, "method": METHOD,
        "dataset_sha256": dataset_hash, "source_revision": manifest["source_revision"],
        "model_files": model_hashes, "fp16_reference_files": reference_hashes,
        "bos_context_id": bos, "runtime": runtime,
        "code_sha256": {
            name: sha256(ROOT / "scripts" / name) for name in ("score_main.py", "score_revision.py")
        },
    }
    if mode == "main":
        smoke = select_rows(rows, "smoke")
        smoke_tokens = expected_tokens(smoke, tokenizer, bos, reference_config["max_position_embeddings"])
        missing, _ = pending_chunks(
            RUN_DIRECTORIES["smoke"] / family / precision, {**identity, "mode": "smoke"},
            smoke, smoke_tokens,
        )
        if missing:
            raise ValueError("Complete the smoke run for this configuration before main scoring")

    rows = select_rows(rows, mode)
    signatures = expected_tokens(rows, tokenizer, bos, reference_config["max_position_embeddings"])
    outdir = RUN_DIRECTORIES[mode] / family / precision
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, f"Another scoring process is using {outdir}\n")
        pending, count = pending_chunks(outdir, identity, rows, signatures)
        print(f"{family}/{precision} {mode}: {count - len(pending)}/{count} checkpoints verified", flush=True)
        if pending:
            work = pending[:args.max_chunks]
            run_pending(work, identity, path, family, config, bos)
            print(f"{len(pending) - len(work)} chunks remaining", flush=True)
        else:
            print("Already complete; model weights were not loaded.", flush=True)


if __name__ == "__main__":
    main()
