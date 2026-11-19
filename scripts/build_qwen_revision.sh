#!/usr/bin/env bash
# Run from the project root with the pinned local Python environment.
set -euo pipefail

repo=Qwen/Qwen3-1.7B-Base
revision=ea980cb0a6c2ae4b936e82123acc929f1cec04c1
source_sha256=6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841
root=models/revision_v2/qwen
source_dir="$root/source"

source scripts/build_revision_model.sh

set -x
mkdir -p "$source_dir"
for file in config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt; do
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

.venv/bin/python scripts/verify_qwen_revision.py "$root" "$repo" "$revision"
