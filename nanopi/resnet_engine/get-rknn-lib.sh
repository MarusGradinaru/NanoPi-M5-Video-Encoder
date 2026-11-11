#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/airockchip/rknn-toolkit2.git"
TAG="v2.3.2"
BASE_PATH="rknpu2/runtime/Linux/librknn_api"
DEST_NAME="librknn_api"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

command -v git >/dev/null 2>&1 || { echo "Error: git is not installed (sudo apt install git)" >&2; exit 1; }

echo "Sparse cloning tag $TAG (only the required folders)..."
if ! git clone --quiet --depth 1 --filter=blob:none --sparse \
        --branch "$TAG" "$REPO_URL" "$TMP_DIR/repo"; then
    echo "Error: clone failed. Does the tag '$TAG' exist in $REPO_URL ?" >&2
    exit 1
fi

git -C "$TMP_DIR/repo" sparse-checkout set --no-cone \
    "/$BASE_PATH/include/" \
    "/$BASE_PATH/aarch64/"

SRC="$TMP_DIR/repo/$BASE_PATH"

for d in include aarch64; do
    if [ ! -d "$SRC/$d" ]; then
        echo "Error: folder '$BASE_PATH/$d' not found in tag '$TAG'." >&2
        exit 1
    fi
done

rm -rf "$SCRIPT_DIR/$DEST_NAME"
mv "$SRC" "$SCRIPT_DIR/$DEST_NAME"

echo
echo "Successfully downloaded:"
echo "  $SCRIPT_DIR/$DEST_NAME"
