#!/usr/bin/env bash
# Run from the project root with the pinned local Python environment.
set -euo pipefail

revision=e00c97a9a389cc184174163cff4ba58facdb8e28
repo=utter-project/EuroLLM-1.7B
source_sha256=658a7ea60f10d4e283c2cc61b4685898b98df25768f4ee9d34450a9efe3eafbd
root=models/revision_v2/eurollm
source_dir="$root/source"

source scripts/build_revision_model.sh

set -x
mkdir -p "$source_dir"
for file in config.json generation_config.json model.safetensors special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json; do
    if [[ ! -f "$source_dir/$file" ]]; then
        curl --fail --location --retry 3 --connect-timeout 20 --continue-at - \
            --output "$source_dir/$file.part" "https://huggingface.co/$repo/resolve/$revision/$file"
        mv "$source_dir/$file.part" "$source_dir/$file"
    fi
done
printf '%s  %s\n' "$source_sha256" "$source_dir/model.safetensors" | shasum -a 256 --check

convert_revision_model "$root" fp16 --hf-path "$source_dir"
convert_revision_model "$root" q8 --hf-path "$root/fp16" --quantize --q-bits 8 --q-group-size 64 --q-mode affine
convert_revision_model "$root" q4 --hf-path "$root/fp16" --quantize --q-bits 4 --q-group-size 64 --q-mode affine

.venv/bin/python scripts/verify_eurollm_revision.py "$root" "$repo" "$revision"
