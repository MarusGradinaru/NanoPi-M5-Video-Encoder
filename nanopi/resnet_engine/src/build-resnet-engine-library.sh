#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENGINE_INCLUDE_DIR="$PROJECT_DIR/include"
RKNN_INCLUDE_DIR="$PROJECT_DIR/librknn_api/include"
RKNN_LIB_DIR="$PROJECT_DIR/librknn_api/aarch64"
RKMPP_PREFIX="${RKMPP_PREFIX:-/opt/rkmpp-stack}"
MPP_INCLUDE_DIR="$RKMPP_PREFIX/include/rockchip"
RGA_INCLUDE_DIR="$RKMPP_PREFIX/include/rga"
LIB_DIR="$PROJECT_DIR/lib"
SOURCE="$SCRIPT_DIR/resnet_engine.cpp"
OUTPUT="$LIB_DIR/libresnet_engine.so"

usage() {
    cat <<USAGE
Usage: $(basename "$0")

Build the standalone C++ ResNetEngine shared library:
  JPEG -> MPP MJPEG decode -> RGA RGB888 -> LUT/FP16 -> RKNN Zero-Copy Multi-Batch

Environment overrides:
  RKMPP_PREFIX  MPP/RGA install prefix (default: /opt/rkmpp-stack)

Required files:
  Public header: $ENGINE_INCLUDE_DIR/resnet_engine.h
  RKNN header:   $RKNN_INCLUDE_DIR/rknn_api.h
  RKNN library:  $RKNN_LIB_DIR/librknnrt.so
  Engine source: $SOURCE

Output:
  $OUTPUT

The library is independent of Python and can be linked by C++ applications.
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

for path in \
    "$ENGINE_INCLUDE_DIR/resnet_engine.h" \
    "$RKNN_INCLUDE_DIR/rknn_api.h" \
    "$RKNN_INCLUDE_DIR/rknn_custom_op.h" \
    "$RKNN_INCLUDE_DIR/rknn_matmul_api.h" \
    "$RKNN_LIB_DIR/librknnrt.so" \
    "$MPP_INCLUDE_DIR/rk_type.h" \
    "$MPP_INCLUDE_DIR/rk_mpi.h" \
    "$MPP_INCLUDE_DIR/rk_mpi_cmd.h" \
    "$MPP_INCLUDE_DIR/mpp_buffer.h" \
    "$MPP_INCLUDE_DIR/mpp_frame.h" \
    "$MPP_INCLUDE_DIR/mpp_packet.h" \
    "$MPP_INCLUDE_DIR/mpp_meta.h" \
    "$RGA_INCLUDE_DIR/im2d.h" \
    "$RGA_INCLUDE_DIR/im2d_buffer.h" \
    "$SOURCE"; do
    if [[ ! -f "$path" ]]; then
        echo "[Error] Required file not found: $path" >&2
        exit 1
    fi
done

for command_name in g++ pkg-config ldd realpath; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "[Error] Required command not found: $command_name" >&2
        exit 1
    fi
done

export PKG_CONFIG_PATH="$RKMPP_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
if ! pkg-config --exists rockchip_mpp librga; then
    echo "[Error] pkg-config could not find rockchip_mpp and librga." >&2
    echo "        Check $RKMPP_PREFIX/lib/pkgconfig." >&2
    exit 1
fi
read -r -a MPP_RGA_CFLAGS <<< "$(pkg-config --cflags rockchip_mpp librga)"
read -r -a MPP_RGA_LIBS <<< "$(pkg-config --libs rockchip_mpp librga)"

mkdir -p "$LIB_DIR"
TEMP_OUTPUT="$OUTPUT.tmp"
trap 'rm -f "$TEMP_OUTPUT"' EXIT

printf '%s\n' "[Build] Project:     $PROJECT_DIR"
printf '%s\n' "[Build] RKNN SDK:    $RKNN_LIB_DIR"
printf '%s\n' "[Build] MPP/RGA:     $RKMPP_PREFIX"
printf '%s\n' "[Build] Source:      $SOURCE"
printf '%s\n' "[Build] Output:      $OUTPUT"
printf '%s\n' "[Build] Compiling standalone ResNetEngine shared library..."

g++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wno-unused-parameter \
    -fPIC -shared \
    -I"$ENGINE_INCLUDE_DIR" \
    -I"$RKNN_INCLUDE_DIR" \
    -I"$MPP_INCLUDE_DIR" \
    -I"$RGA_INCLUDE_DIR" \
    "${MPP_RGA_CFLAGS[@]}" \
    "$SOURCE" \
    -L"$RKNN_LIB_DIR" \
    -Wl,--no-as-needed \
    -lrknnrt \
    "${MPP_RGA_LIBS[@]}" \
    -pthread \
    -Wl,-soname,libresnet_engine.so \
    -Wl,-rpath,'$ORIGIN/../librknn_api/aarch64' \
    -Wl,-rpath,"$RKMPP_PREFIX/lib" \
    -o "$TEMP_OUTPUT"

if [[ ! -s "$TEMP_OUTPUT" ]]; then
    echo "[Error] Compiler did not produce a non-empty shared library." >&2
    exit 1
fi

LDD_OUTPUT="$(env -u LD_LIBRARY_PATH ldd "$TEMP_OUTPUT")"
printf '\n%s\n' "[Link] Runtime dependencies (LD_LIBRARY_PATH unset):"
printf '%s\n' "$LDD_OUTPUT"

check_resolved_library() {
    local soname_prefix="$1"
    local expected_prefix="$2"
    local resolved
    local resolved_real
    local expected_real

    resolved="$(awk -v prefix="$soname_prefix" \
        '$2 == "=>" && ($1 == prefix || index($1, prefix ".") == 1) { print $3; exit }' \
        <<< "$LDD_OUTPUT")"
    if [[ -z "$resolved" || ! -e "$resolved" ]]; then
        echo "[Error] Could not resolve $soname_prefix (including versioned SONAMEs)." >&2
        exit 1
    fi

    resolved_real="$(realpath -e -- "$resolved")"
    expected_real="$(realpath -e -- "$expected_prefix")"
    if [[ "$resolved_real" != "$expected_real"/* ]]; then
        echo "[Error] $soname_prefix resolved from an unexpected location: $resolved" >&2
        echo "        Canonical path: $resolved_real" >&2
        echo "        Expected under: $expected_real" >&2
        exit 1
    fi
}

check_resolved_library "librknnrt.so" "$RKNN_LIB_DIR"
check_resolved_library "librockchip_mpp.so" "$RKMPP_PREFIX/lib"
check_resolved_library "librga.so" "$RKMPP_PREFIX/lib"

mv -f "$TEMP_OUTPUT" "$OUTPUT"
trap - EXIT
chmod 0755 "$OUTPUT"

echo
printf '%s\n' "[Done] ResNetEngine shared library built successfully."
printf '%s\n' "[Output] $OUTPUT"
printf '%s\n' "[Header] $ENGINE_INCLUDE_DIR/resnet_engine.h"
