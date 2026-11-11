#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOURCE="${SCRIPT_DIR}/rkmpp_jpeg_decode.cpp"
OUTPUT="${PROJECT_DIR}/rkmpp-jpeg-decode"
RKMPP_PREFIX="/opt/rkmpp-stack"

if [[ ! -f "${SOURCE}" ]]; then
    echo "[Error] Source file not found: ${SOURCE}" >&2
    exit 1
fi

MPP_INCLUDE_DIR="${RKMPP_PREFIX}/include/rockchip"
RGA_INCLUDE_DIR="${RKMPP_PREFIX}/include/rga"

for required in \
    "${MPP_INCLUDE_DIR}/rk_type.h" \
    "${MPP_INCLUDE_DIR}/rk_mpi.h" \
    "${MPP_INCLUDE_DIR}/rk_mpi_cmd.h" \
    "${MPP_INCLUDE_DIR}/mpp_buffer.h" \
    "${MPP_INCLUDE_DIR}/mpp_frame.h" \
    "${MPP_INCLUDE_DIR}/mpp_packet.h" \
    "${MPP_INCLUDE_DIR}/mpp_meta.h" \
    "${RGA_INCLUDE_DIR}/im2d.h" \
    "${RGA_INCLUDE_DIR}/im2d_buffer.h"; do
    if [[ ! -f "${required}" ]]; then
        echo "[Error] Required MPP/RGA header not found: ${required}" >&2
        exit 1
    fi
done

export PKG_CONFIG_PATH="${RKMPP_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"

if ! pkg-config --exists rockchip_mpp librga; then
    echo "[Error] pkg-config could not find rockchip_mpp and librga under ${RKMPP_PREFIX}." >&2
    echo "        Check ${RKMPP_PREFIX}/lib/pkgconfig and the FFmpeg stack installation." >&2
    exit 1
fi

read -r -a DEP_CFLAGS <<< "$(pkg-config --cflags rockchip_mpp librga)"
read -r -a DEP_LIBS <<< "$(pkg-config --libs rockchip_mpp librga)"

if ! command -v g++ >/dev/null 2>&1; then
    echo "[Error] g++ is not installed." >&2
    exit 1
fi
if ! pkg-config --exists zlib; then
    echo "[Error] zlib development package is missing (install zlib1g-dev)." >&2
    exit 1
fi
read -r -a ZLIB_CFLAGS <<< "$(pkg-config --cflags zlib)"
read -r -a ZLIB_LIBS <<< "$(pkg-config --libs zlib)"

TEMP_OUTPUT="${OUTPUT}.tmp"
trap 'rm -f "${TEMP_OUTPUT}"' EXIT

echo "[Build] Source:     ${SOURCE}"
echo "[Build] MPP/RGA:    ${RKMPP_PREFIX}"
echo "[Build] Output:     ${OUTPUT}"
echo "[Build] Compiling MPP + RGA JPEG preprocessing utility..."

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
    -I"${MPP_INCLUDE_DIR}" -I"${RGA_INCLUDE_DIR}" \
    "${DEP_CFLAGS[@]}" "${ZLIB_CFLAGS[@]}" \
    "${SOURCE}" \
    -o "${TEMP_OUTPUT}" \
    "${DEP_LIBS[@]}" "${ZLIB_LIBS[@]}" \
    -pthread \
    -Wl,-rpath,"${RKMPP_PREFIX}/lib"

chmod 0755 "${TEMP_OUTPUT}"
mv -f "${TEMP_OUTPUT}" "${OUTPUT}"

trap - EXIT

echo
echo "[Link] Runtime dependencies (ignoring any external LD_LIBRARY_PATH):"
DEPENDENCIES="$(env -u LD_LIBRARY_PATH ldd "${OUTPUT}")"
printf '%s\n' "${DEPENDENCIES}"
for library in librockchip_mpp.so librga.so; do
    dependency_line="$(printf '%s\n' "${DEPENDENCIES}" | grep -m1 "${library//./\.}" || true)"
    if [[ -z "${dependency_line}" || "${dependency_line}" != *"${RKMPP_PREFIX}/lib/"* ]]; then
        echo "[Error] ${library} is not resolved from ${RKMPP_PREFIX}/lib." >&2
        exit 1
    fi
done
echo
echo "[Done] Build completed. Executable: ${OUTPUT}"
