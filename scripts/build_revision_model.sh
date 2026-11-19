# Shared by the EuroLLM and Qwen build scripts.
convert_revision_model() (
    local root=$1 variant=$2
    shift 2
    local destination="$root/$variant"
    if [[ -f "$destination/model.safetensors" ]]; then
        return
    fi

    local staging
    staging=$(mktemp -d "$root/.conversion.XXXXXX")
    trap 'rm -rf "$staging"' EXIT
    .venv/bin/python -m mlx_lm convert "$@" --mlx-path "$staging/model" --dtype float16

    mkdir -p "$destination"
    local file name
    for file in "$staging/model/"*; do
        name=${file##*/}
        if [[ "$name" == model.safetensors ]]; then
            continue
        fi
        if [[ -e "$destination/$name" ]]; then
            if ! cmp -s "$file" "$destination/$name"; then
                printf 'Converted metadata differs from %s\n' "$destination/$name" >&2
                return 1
            fi
        else
            mv "$file" "$destination/$name"
        fi
    done
    # Move the completed weights last so an interrupted conversion can resume.
    mv "$staging/model/model.safetensors" "$destination/model.safetensors"
)
