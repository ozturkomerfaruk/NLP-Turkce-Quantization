"""Verify EuroLLM revision artifacts, compare baseline weights and run smoke scores."""

import argparse
import hashlib
import json
import mmap
import platform
import struct
import tempfile
from pathlib import Path

from prepare_dataset import sha256
from score_main import package_version, score_sentence

BASELINE_HASHES = {
    "source": "658a7ea60f10d4e283c2cc61b4685898b98df25768f4ee9d34450a9efe3eafbd",
    "fp16": "179b442ef5a4a22cad87640f6c89a11d9dafe81674cdcf9df6d11a7f4545bac9",
    "q8": "69553d7d3fcd6e5cab96a4227e877c83b63df16efc5bfa12f5c9f56364a3a942",
    "q4": "a653c92f03b133b79f57b05cf680ace89ee6f2e5313baca8ccde1fd4e2581577",
}
WEIGHTS = "model.safetensors"
SENTENCES = ("Çocuklar bahçede oynuyorlar.", "Çocuklar bahçede oynuyorum.")


def tensor_data(buffer):
    header_size = struct.unpack_from("<Q", buffer)[0]
    start = 8 + header_size
    tensors = json.loads(buffer[8:start])
    tensors.pop("__metadata__", None)
    return tensors, start


def tensor_signature(buffer, start, info):
    first, last = info["data_offsets"]
    with memoryview(buffer)[start + first:start + last] as payload:
        return info["dtype"], info["shape"], hashlib.sha256(payload).digest()


def compare_weights(old: Path, new: Path) -> dict:
    old_hash = sha256(old)
    new_hash = sha256(new)
    identical = old_hash == new_hash
    changed = []
    if not identical:
        with (
            old.open("rb") as old_file,
            new.open("rb") as new_file,
            mmap.mmap(old_file.fileno(), 0, access=mmap.ACCESS_READ) as old_bytes,
            mmap.mmap(new_file.fileno(), 0, access=mmap.ACCESS_READ) as new_bytes,
        ):
            old_tensors, old_start = tensor_data(old_bytes)
            new_tensors, new_start = tensor_data(new_bytes)
            for name in sorted(old_tensors.keys() | new_tensors.keys()):
                if name not in old_tensors or name not in new_tensors:
                    changed.append(name)
                    continue
                old_signature = tensor_signature(old_bytes, old_start, old_tensors[name])
                new_signature = tensor_signature(new_bytes, new_start, new_tensors[name])
                if old_signature != new_signature:
                    changed.append(name)

    return {
        "baseline_sha256": old_hash,
        "revision_sha256": new_hash,
        "file_identical": identical,
        "tensors_identical": not changed,
        "different_tensors": changed,
    }


def verify_model_files(directory: Path, bits: int | None) -> dict:
    config = json.loads((directory / "config.json").read_text())
    quantization = config.get("quantization")
    expected = {"bits": bits, "group_size": 64, "mode": "affine"} if bits else None
    if quantization != expected:
        raise ValueError(f"{directory}: unexpected quantization: {quantization}")

    index = json.loads((directory / "model.safetensors.index.json").read_text())
    with (directory / WEIGHTS).open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        tensors = json.loads(stream.read(header_size))
    tensors.pop("__metadata__", None)
    if index["weight_map"] != dict.fromkeys(tensors, WEIGHTS):
        raise ValueError(f"{directory}: weight index does not match the safetensors file")
    dtypes = sorted({tensor["dtype"] for tensor in tensors.values()})
    if dtypes != (["F16", "U32"] if bits else ["F16"]):
        raise ValueError(f"{directory}: unexpected weight dtypes: {dtypes}")
    return {"quantization": quantization, "tensor_dtypes": dtypes}


def verify_model(directory: Path, bits: int | None) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    metadata = verify_model_files(directory, bits)
    config = json.loads((directory / "config.json").read_text())
    model, tokenizer = load(str(directory))
    scores = [
        score_sentence(sentence, model, tokenizer, mx, nn, config["max_position_embeddings"])
        for sentence in SENTENCES
    ]
    return {**metadata, "smoke_scores": scores}


def describe_files(directory: Path, weight_hash: str) -> dict:
    return {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": weight_hash if path.name == WEIGHTS else sha256(path),
        }
        for path in sorted(directory.iterdir()) if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("repo")
    parser.add_argument("revision")
    args = parser.parse_args()
    models = {}
    token_hashes = None
    for variant, bits in (("source", None), ("fp16", None), ("q8", 8), ("q4", 4)):
        directory = args.root / variant
        baseline = Path("models") / f"eurollm-{variant}"
        baseline_hash = BASELINE_HASHES[variant]
        has_baseline = (baseline / WEIGHTS).is_file()
        if has_baseline:
            comparison = compare_weights(baseline / WEIGHTS, directory / WEIGHTS)
            if comparison["baseline_sha256"] != baseline_hash:
                raise ValueError(f"{baseline}: original weight hash differs from the Aşama 1 audit")
        else:
            revision_hash = sha256(directory / WEIGHTS)
            identical = revision_hash == baseline_hash
            comparison = {
                "baseline_sha256": baseline_hash,
                "revision_sha256": revision_hash,
                "file_identical": identical,
                "tensors_identical": True if identical else None,
                "different_tensors": [] if identical else None,
            }
        if variant == "source" and not comparison["file_identical"]:
            raise ValueError("Pinned download does not match the original source weights")

        files = describe_files(directory, comparison["revision_sha256"])
        baseline_files = describe_files(baseline, comparison["baseline_sha256"])
        tokenizer_matches = {
            name: name in baseline_files and files[name]["sha256"] == baseline_files[name]["sha256"]
            for name in files if name.startswith(("tokenizer", "special_tokens"))
        }
        if "tokenizer.json" not in tokenizer_matches:
            raise ValueError(f"{directory}: missing tokenizer.json")
        model_record = {
            "path": str(directory),
            "files": files,
            "baseline_weights_available": has_baseline,
            "baseline_files": baseline_files,
            "weights_comparison": comparison,
            "tokenizer_matches_baseline": tokenizer_matches,
        }
        if variant != "source":
            validation = verify_model(directory, bits)
            current_hashes = [score["token_ids_sha256"] for score in validation["smoke_scores"]]
            if token_hashes is not None and current_hashes != token_hashes:
                raise ValueError(f"{directory}: tokenization differs from FP16")
            token_hashes = current_hashes
            model_record.update(validation)

        models[variant] = model_record
        print(
            f"{variant}: verified, weight file matches baseline={comparison['file_identical']}, "
            f"tensors match baseline={comparison['tensors_identical']}",
            flush=True,
        )

    manifest = {
        "source_repo": args.repo,
        "source_revision": args.revision,
        "build_script": "scripts/rebuild_eurollm_revision.sh",
        "build_log": "outputs/revision_v2/eurollm/rebuild.log",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{name: package_version(name) for name in ("mlx", "mlx-lm", "transformers", "tokenizers")},
        },
        "smoke_sentences": SENTENCES,
        "models": models,
    }
    destination = Path("docs/eurollm_revision_manifest.json")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    Path(stream.name).replace(destination)
    print(f"Manifest: {destination}", flush=True)


if __name__ == "__main__":
    main()
