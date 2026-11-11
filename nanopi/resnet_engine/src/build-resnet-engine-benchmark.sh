#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENGINE_INCLUDE_DIR="$PROJECT_DIR/include"
ENGINE_LIB_DIR="$PROJECT_DIR/lib"
ENGINE_HEADER="$ENGINE_INCLUDE_DIR/resnet_engine.h"
ENGINE_LIBRARY="$ENGINE_LIB_DIR/libresnet_engine.so"
SOURCE_PATH="$SCRIPT_DIR/resnet_engine_benchmark.cpp"
OUTPUT_PATH="$PROJECT_DIR/resnet-engine-benchmark"

usage() {
    cat <<USAGE
Usage: ./$(basename "$0")

Build the ResNetEngine benchmark executable against the shared C++ engine:
  resnet-engine-benchmark -> libresnet_engine.so -> RKNN / MPP / RGA

Required files:
  Public header: $ENGINE_HEADER
  Shared library: $ENGINE_LIBRARY
  Benchmark source: $SOURCE_PATH

Output:
  $OUTPUT_PATH

Run manually from the resnet_engine directory:
  ./resnet-engine-benchmark --batch-size 4 --used-images 256 --warm-target 16
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    echo "[Error] This build script does not accept arguments." >&2
    usage >&2
    exit 2
fi

for path in "$ENGINE_HEADER" "$ENGINE_LIBRARY" "$SOURCE_PATH"; do
    if [[ ! -f "$path" ]]; then
        echo "[Error] Required file not found: $path" >&2
        exit 1
    fi
done

for command_name in g++ ldd realpath; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "[Error] Required command not found: $command_name" >&2
        exit 1
    fi
done

printf '%s\n' "[Build] Source:       $SOURCE_PATH"
printf '%s\n' "[Build] Engine:       $ENGINE_LIBRARY"
printf '%s\n' "[Build] Output:       $OUTPUT_PATH"
printf '%s\n' "[Build] Compiling benchmark against shared ResNetEngine library..."

TEMP_OUTPUT="${OUTPUT_PATH}.tmp"
trap 'rm -f "$TEMP_OUTPUT"' EXIT

g++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic \
    -I"$ENGINE_INCLUDE_DIR" \
    "$SOURCE_PATH" \
    -L"$ENGINE_LIB_DIR" \
    -Wl,--no-as-needed \
    -lresnet_engine \
    -pthread \
    -Wl,-rpath,'$ORIGIN/lib' \
    -o "$TEMP_OUTPUT"

if [[ ! -s "$TEMP_OUTPUT" ]]; then
    echo "[Error] Compiler did not produce a non-empty benchmark executable." >&2
    exit 1
fi

LDD_OUTPUT="$(env -u LD_LIBRARY_PATH ldd "$TEMP_OUTPUT")"
printf '\n%s\n' "[Link] Runtime dependencies (LD_LIBRARY_PATH unset):"
printf '%s\n' "$LDD_OUTPUT"

RESOLVED_ENGINE="$(awk '$1 == "libresnet_engine.so" && $2 == "=>" { print $3; exit }' <<< "$LDD_OUTPUT")"
if [[ -z "$RESOLVED_ENGINE" || ! -e "$RESOLVED_ENGINE" ]]; then
    echo "[Error] libresnet_engine.so was not resolved." >&2
    exit 1
fi
EXPECTED_ENGINE="$(realpath -e -- "$ENGINE_LIBRARY")"
ACTUAL_ENGINE="$(realpath -e -- "$RESOLVED_ENGINE")"
if [[ "$ACTUAL_ENGINE" != "$EXPECTED_ENGINE" ]]; then
    echo "[Error] Benchmark resolved a different ResNetEngine library." >&2
    echo "        Expected: $EXPECTED_ENGINE" >&2
    echo "        Resolved: $ACTUAL_ENGINE" >&2
    exit 1
fi

if grep -q 'not found' <<< "$LDD_OUTPUT"; then
    echo "[Error] One or more runtime dependencies are unresolved." >&2
    exit 1
fi

mv -f "$TEMP_OUTPUT" "$OUTPUT_PATH"
trap - EXIT
chmod 0755 "$OUTPUT_PATH"

echo
echo "[Done] Benchmark compiled against libresnet_engine.so successfully."
echo "[Output] $OUTPUT_PATH"
echo "[Run]    $OUTPUT_PATH --help"
