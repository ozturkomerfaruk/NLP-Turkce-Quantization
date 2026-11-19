"""Check Qwen model files and independently verify BOS-conditioned sentence scores."""

import argparse
import json
import math
import platform
import tempfile
from pathlib import Path

from prepare_dataset import sha256
from score_main import package_version, read_jsonl, score_sentence
from verify_eurollm_revision import WEIGHTS, describe_files, verify_model_files

SMOKE_PAIRS = (
    ("Biz yarın erkenden yola çıkacağız.", "Biz yarın erkenden yola çıkacaksınız."),
    ("Terzi eski ceketi dikkatle onardı.", "Terzi eski ceketle dikkatle onardı."),
    ("Onun toplantıya geç kalması hepimizi şaşırttı.", "Onun toplantıya geç kalmak hepimizi şaşırttı."),
)
TOLERANCE = 5e-5


def check_context(config, tokenizer, expected_bos):
    if config["model_type"] != "qwen3" or config["bos_token_id"] != expected_bos:
        raise ValueError("Unexpected Qwen model type or configured BOS")
    if tokenizer.bos_token_id is not None:
        raise ValueError("Expected the pinned Qwen tokenizer to have no BOS token")
    if tokenizer.eos_token_id != expected_bos:
        raise ValueError("Qwen start context must match the tokenizer's end-of-text ID")


def verify_scores(directory, bits, source_config, bos, source_tokens, sentences):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    metadata = verify_model_files(directory, bits)
    model, tokenizer, config = load(str(directory), return_config=True)
    check_context(config, tokenizer, bos)
    context_limit = config["max_position_embeddings"]
    if context_limit != source_config["max_position_embeddings"]:
        raise ValueError(f"{directory}: model context limit changed")

    scores = []
    for sentence, tokens in zip(sentences, source_tokens, strict=True):
        if tokenizer.encode(sentence, add_special_tokens=False) != tokens:
            raise ValueError(f"{directory}: tokenizer differs from the pinned source")
        score = score_sentence(sentence, model, tokenizer, mx, nn, context_limit, bos_token_id=bos)
        if score["n_tokens"] != len(tokens):
            raise ValueError(f"{directory}: unexpected number of scored sentence tokens")

        # Independently reduce the next-token distribution for each sentence position.
        inputs = mx.array([[bos, *tokens[:-1]]], dtype=mx.int32)
        logits = model(inputs)[0].astype(mx.float32)
        normalizers = mx.logsumexp(logits, axis=-1).tolist()
        total = math.fsum(
            float(logits[index, token].item()) - normalizers[index]
            for index, token in enumerate(tokens)
        )
        mean = total / len(tokens)
        if not (
            math.isclose(score["total_logp"], total, rel_tol=0, abs_tol=TOLERANCE)
            and math.isclose(score["mean_logp"], mean, rel_tol=0, abs_tol=TOLERANCE)
        ):
            raise ValueError(f"{directory}: scoring does not match the per-token calculation")
        scores.append({**score, "reference_total_logp": total, "reference_mean_logp": mean})

    mx.clear_cache()
    return {**metadata, "smoke_scores": scores}


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("repo")
    parser.add_argument("revision")
    args = parser.parse_args()

    dataset = read_jsonl(Path("data/revision_v2/turblimp_3662.jsonl"))
    benchmark_sentences = {
        row[key] for row in dataset for key in ("sentence_good", "sentence_bad")
    }
    sentences = [sentence for pair in SMOKE_PAIRS for sentence in pair]
    if benchmark_sentences.intersection(sentences):
        raise ValueError("A smoke sentence overlaps the main benchmark")

    source = args.root / "source"
    source_config = json.loads((source / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    bos = 151643
    check_context(source_config, tokenizer, bos)
    source_tokens = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]
    if any(not tokens or bos in tokens for tokens in source_tokens):
        raise ValueError("Smoke sentences unexpectedly include an empty sequence or context token")

    manifest = {
        "source_repo": args.repo,
        "source_revision": args.revision,
        "source_path": str(source),
        "source_files": describe_files(source, sha256(source / WEIGHTS)),
        "build_script": "scripts/build_qwen_revision.sh",
        "build_log": "outputs/revision_v2/qwen/build.log",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{name: package_version(name) for name in ("mlx", "mlx-lm", "transformers", "tokenizers")},
        },
        "scoring": {
            "tokenizer_bos_token_id": tokenizer.bos_token_id,
            "model_bos_context_id": bos,
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "append_eos": False,
            "add_special_tokens": False,
            "reference_tolerance": TOLERANCE,
            "smoke_pairs": SMOKE_PAIRS,
            "sentence_token_ids": source_tokens,
        },
        "models": {},
    }
    for variant, bits in (("fp16", None), ("q8", 8), ("q4", 4)):
        directory = args.root / variant
        verification = verify_scores(directory, bits, source_config, bos, source_tokens, sentences)
        manifest["models"][variant] = {
            "path": str(directory),
            "files": describe_files(directory, sha256(directory / WEIGHTS)),
            **verification,
        }
        print(f"{variant}: verified {len(SMOKE_PAIRS)} minimal pairs, tokenizer and Mean/Total scores", flush=True)

    destination = Path("docs/qwen_revision_manifest.json")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    Path(stream.name).replace(destination)
    print(f"Manifest: {destination}", flush=True)


if __name__ == "__main__":
    main()
