#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENGINE_INCLUDE_DIR="$PROJECT_DIR/include"
ENGINE_LIB_DIR="$PROJECT_DIR/lib"
PYTHON_DIR="$PROJECT_DIR/python"
ENGINE_HEADER="$ENGINE_INCLUDE_DIR/resnet_engine.h"
ENGINE_LIBRARY="$ENGINE_LIB_DIR/libresnet_engine.so"
BINDINGS_SOURCE="$SCRIPT_DIR/resnet_bindings.cpp"
PYTHON="${PYTHON:-python3}"

usage() {
    cat <<USAGE
Usage: $(basename "$0")

Build the Python extension for the standalone ResNetEngine C++ library.
The extension contains only pybind11 bindings and links to:
  $ENGINE_LIBRARY

Environment overrides:
  PYTHON  Python interpreter whose pybind11 installation will be used (default: python3)

Required files:
  Public header:    $ENGINE_HEADER
  Shared library:   $ENGINE_LIBRARY
  Bindings source:  $BINDINGS_SOURCE

Output:
  $PYTHON_DIR/resnet_engine<Python extension suffix>

Install pybind11 into the selected Python environment before building:
  $PYTHON -m pip install pybind11

Build the shared C++ library first, if needed:
  ./src/build-resnet-engine-library.sh
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

for path in "$ENGINE_HEADER" "$ENGINE_LIBRARY" "$BINDINGS_SOURCE"; do
    if [[ ! -f "$path" ]]; then
        echo "[Error] Required file not found: $path" >&2
        if [[ "$path" == "$ENGINE_LIBRARY" ]]; then
            echo "        Build the shared C++ library first:" >&2
            echo "        ./src/build-resnet-engine-library.sh" >&2
        fi
        exit 1
    fi
done

for command_name in g++ ldd realpath; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "[Error] Required command not found: $command_name" >&2
        exit 1
    fi
done
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "[Error] Python interpreter not found: $PYTHON" >&2
    exit 1
fi

if ! "$PYTHON" -m pybind11 --includes >/dev/null 2>&1; then
    echo "[Error] pybind11 is not installed for: $PYTHON" >&2
    echo "        Install it with: $PYTHON -m pip install pybind11" >&2
    exit 1
fi

PYBIND_INCLUDES="$($PYTHON -m pybind11 --includes)"
read -r -a PYBIND_INCLUDE_FLAGS <<< "$PYBIND_INCLUDES"
EXT_SUFFIX="$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"
PYTHON_VERSION="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

mkdir -p "$PYTHON_DIR"
OUTPUT_PATH="$PYTHON_DIR/resnet_engine${EXT_SUFFIX}"
TEMP_OUTPUT="${OUTPUT_PATH}.tmp"
trap 'rm -f "$TEMP_OUTPUT"' EXIT

printf '%s\n' "[Build] Python:       $PYTHON ($PYTHON_VERSION)"
printf '%s\n' "[Build] Project:      $PROJECT_DIR"
printf '%s\n' "[Build] Engine:       $ENGINE_LIBRARY"
printf '%s\n' "[Build] pybind11:     $($PYTHON -m pybind11 --version)"
printf '%s\n' "[Build] Output:       $OUTPUT_PATH"
printf '%s\n' "[Build] Compiling Python bindings against libresnet_engine.so..."

# Python and pybind11 include flags are intentionally expanded as separate arguments.
g++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wno-unused-parameter \
    -fPIC -fvisibility=hidden -shared \
    -I"$ENGINE_INCLUDE_DIR" \
    "${PYBIND_INCLUDE_FLAGS[@]}" \
    "$BINDINGS_SOURCE" \
    -L"$ENGINE_LIB_DIR" \
    -Wl,--no-as-needed \
    -lresnet_engine \
    -Wl,--as-needed \
    -pthread \
    -Wl,-rpath,'$ORIGIN/../lib' \
    -o "$TEMP_OUTPUT"

if [[ ! -s "$TEMP_OUTPUT" ]]; then
    echo "[Error] Compiler did not produce a non-empty Python extension." >&2
    exit 1
fi

LDD_OUTPUT="$(env -u LD_LIBRARY_PATH ldd "$TEMP_OUTPUT")"
printf '\n%s\n' "[Link] Runtime dependencies (LD_LIBRARY_PATH unset):"
printf '%s\n' "$LDD_OUTPUT"

RESOLVED_ENGINE="$(awk '$1 == "libresnet_engine.so" && $2 == "=>" { print $3; exit }' <<< "$LDD_OUTPUT")"
if [[ -z "$RESOLVED_ENGINE" || ! -e "$RESOLVED_ENGINE" ]]; then
    echo "[Error] libresnet_engine.so was not resolved from the project lib directory." >&2
    exit 1
fi
EXPECTED_ENGINE="$(realpath -e -- "$ENGINE_LIBRARY")"
ACTUAL_ENGINE="$(realpath -e -- "$RESOLVED_ENGINE")"
if [[ "$ACTUAL_ENGINE" != "$EXPECTED_ENGINE" ]]; then
    echo "[Error] Python extension resolved an unexpected ResNetEngine library." >&2
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
echo "[Done] ResNetEngine Python extension built successfully."
echo "[Import] Add '$PYTHON_DIR' to sys.path, then: import resnet_engine"
echo "[Output] $OUTPUT_PATH"
