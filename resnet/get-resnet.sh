#!/usr/bin/env bash
# Download selected directories from the ResNet Hugging Face bucket.

set -Eeuo pipefail

BUCKET_ROOT="hf://buckets/MarusGradinaru/romanian-tv-resnet-classifier"

usage() {
    cat <<'USAGE'
Usage:
  ./get-resnet.sh <folder> [folder ...]
  ./get-resnet.sh --help

Download (sync) one or more folders from the ResNet Hugging Face bucket into the
current working directory. Folder paths are relative to the bucket root.

Examples:
  ./get-resnet.sh models dataset
  ./get-resnet.sh models
  ./get-resnet.sh artifacts/exported

Existing local files are updated when needed. Extra local files are deleted !
USAGE
}

if (($# == 0)); then
    usage
    exit 0
fi

if (($# == 1)) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

if ! command -v hf >/dev/null 2>&1; then
    printf '[Error] Hugging Face CLI (hf) was not found. Install huggingface_hub and ensure hf is in PATH.\n' >&2
    exit 127
fi

# Validate each path as a relative path beneath the bucket root.
# Nested subdirectories are allowed, but absolute paths and traversal are not.
for path in "$@"; do
    if [[ -z "$path" || "$path" == /* || "$path" == *\\* ]]; then
        printf '[Error] Invalid folder path: %q\nUse relative paths beneath the bucket root.\n' "$path" >&2
        exit 2
    fi

    IFS='/' read -r -a components <<< "$path"
    for component in "${components[@]}"; do
        if [[ -z "$component" || "$component" == "." || "$component" == ".." ]]; then
            printf '[Error] Invalid folder path: %q\nEmpty, '.' and '..' path components are not allowed.\n' "$path" >&2
            exit 2
        fi
    done
done

printf 'ResNet resource downloader\n'
printf 'Bucket: %s\n' "$BUCKET_ROOT"
printf 'Destination: %s\n\n' "$PWD"

count=$#
index=0
for path in "$@"; do
    ((index += 1))
    remote_path="${BUCKET_ROOT}/${path}"
    local_path="${PWD}/${path}"

    printf '[%d/%d] Downloading: %s\n' "$index" "$count" "$path"
    printf '      Destination: %s\n' "$local_path"
    mkdir -p -- "$(dirname -- "$local_path")"

    if ! hf buckets sync --delete "$remote_path" "$local_path"; then
        printf '\n[Error] Failed to download %q. Stopping.\n' "$path" >&2
        exit 1
    fi

    printf '[OK] Downloaded: %s\n\n' "$path"
done

printf 'All requested folders were downloaded successfully.\n'
