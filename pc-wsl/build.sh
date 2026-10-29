#!/usr/bin/env bash

set -Eeuo pipefail

REMOTE_PATH_DEFAULT='~/NanoPi-M5-Video-Encoder/resnet/artifacts/'

SCRIPTS=(
    "check_export_env.py"
    "export_resnet_onnx.py"
    "make_onnx_nchw.py"
    "convert_onnx_rknn.py"
)

LABELS=(
    "Checking the Python environment"
    "Exporting ResNet50V2 to ONNX"
    "Converting ONNX input layout from NHWC to NCHW"
    "Converting the NCHW ONNX model to RKNN"
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ARTIFACTS_DIR="$(realpath -m "$SCRIPT_DIR/../resnet/artifacts")"
RKNN_MODEL="$ARTIFACTS_DIR/resnet50v2_feature_extractor_rk3576_fp16_320x180.rknn"
VALIDATION_FILE="$ARTIFACTS_DIR/resnet50v2_rknn_validation.npz"

# Select an interpreter. BUILD_PYTHON may explicitly override automatic selection.
if [[ -n "${BUILD_PYTHON:-}" ]]; then
    PYTHON_BIN="$BUILD_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -x "$SCRIPT_DIR/.venv-resnet/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv-resnet/bin/python"
else
    printf '[Error] Python was not found. Activate the project virtual environment or set BUILD_PYTHON.\n' >&2
    exit 127
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
    printf '[Error] Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
    exit 127
fi

printf '%s\n' '===================================================================='
printf '%s\n' 'RESNET50V2 -> ONNX -> RKNN BUILD'
printf '%s\n' '===================================================================='
printf '[Project  ]    %s\n' "$SCRIPT_DIR"
printf '[Artifacts]    %s\n' "$ARTIFACTS_DIR"
printf '[Python   ]    %s\n\n' "$PYTHON_BIN"

# Fail early if the build sequence is incomplete, before running any step.
for script in "${SCRIPTS[@]}"; do
    if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
        printf '[Error] Required script not found: %s\n' "$SCRIPT_DIR/$script" >&2
        exit 2
    fi
done

mkdir -p "$ARTIFACTS_DIR"

for index in "${!SCRIPTS[@]}"; do
    step=$((index + 1))
    "$PYTHON_BIN" "${SCRIPTS[$index]}" || {
        status=$?
        printf '\n[FAIL] Step %d/%d failed: %s (exit code %d).\n' "$step" "${#SCRIPTS[@]}" "${SCRIPTS[$index]}" "$status" >&2
        printf '[STOP] Remaining steps were not executed.\n' >&2
        exit "$status"
    }
done

# Require both outputs before offering to transfer them.
if [[ ! -s "$RKNN_MODEL" || ! -s "$VALIDATION_FILE" ]]; then
    printf '\n[Error] The build scripts completed, but one or both expected output files are missing or empty.\n' >&2
    printf '        Expected: %s\n' "$RKNN_MODEL" >&2
    printf '        Expected: %s\n' "$VALIDATION_FILE" >&2
    exit 3
fi

printf '%s\n' '===================================================================='
printf '\n[Done] All build steps completed successfully.\n'
printf '[Artifact] %s (%s bytes)\n' "$RKNN_MODEL" "$(stat -c '%s' "$RKNN_MODEL")"
printf '[Artifact] %s (%s bytes)\n' "$VALIDATION_FILE" "$(stat -c '%s' "$VALIDATION_FILE")"

printf '\n'
if ! read -r -p 'Send the RKNN model and validation data to the NanoPi? [Y/n] ' send_answer; then
    send_answer='n'
fi
send_answer="${send_answer,,}"

case "$send_answer" in
    ''|y|yes)
        ;;
    n|no)
        printf '[Info] Transfer skipped. Build artifacts remain in resnet/artifacts/.\n'
        exit 0
        ;;
    *)
        printf '[Info] Unrecognized answer; transfer skipped. Build artifacts remain in /resnet/artifacts/.\n'
        exit 0
        ;;
esac

printf '\nEnter the NanoPi SSH target as user@host/IP or an SSH config alias.\n'
if ! read -r -p 'NanoPi SSH target: ' remote_target; then
    remote_target=''
fi

if [[ -z "$remote_target" ]]; then
    printf '[Info] No SSH target specified. Transfer skipped; build completed successfully.\n'
    exit 0
fi

if [[ "$remote_target" == -* || "$remote_target" =~ [[:space:]] ]]; then
    printf '[Error] Invalid SSH target. Use user@host/IP or a configured SSH alias without spaces.\n' >&2
    exit 2
fi

if ! read -r -p "Remote destination path [${REMOTE_PATH_DEFAULT}]: " remote_path; then
    remote_path=''
fi

remote_path="${remote_path:-$REMOTE_PATH_DEFAULT}"

printf '\n[Transfer] Sending artifacts to %s:%s ...\n' \
    "$remote_target" "$remote_path"

if scp -- "$RKNN_MODEL" "$VALIDATION_FILE" "${remote_target}:${remote_path}"; then
    printf '[Done] Both artifacts were transferred successfully.\n'
else
    status=$?
    printf '[Error] SCP failed with exit code %d. Build artifacts remain in %s.\n' \
        "$status" "$ARTIFACTS_DIR" >&2
    exit "$status"
fi
