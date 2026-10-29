#!/usr/bin/env bash
set -Eeuo pipefail

# Set up the Python environment used to convert the Keras ResNet model to RKNN.
# Run this script from NanoPi-M5-Video-Encoder/pc-wsl.

readonly EXPECTED_REPO_DIR="NanoPi-M5-Video-Encoder"
readonly EXPECTED_WORK_DIR="pc-wsl"
readonly VENV_DIR=".venv-resnet"
readonly VENV_PYTHON="${VENV_DIR}/bin/python"
readonly ACTIVATE_FILE="${VENV_DIR}/bin/activate"

log() {
    printf '\n[INFO] %s\n' "$*"
}

warn() {
    printf '\n[WARNING] %s\n' "$*" >&2
}

fail() {
    printf '\n[ERROR] %s\n' "$*" >&2
    exit 1
}

on_error() {
    local status=$?
    printf '\n[ERROR] A command failed near line %s (exit code %s).\n' \
        "${BASH_LINENO[0]:-unknown}" "$status" >&2
    printf '[ERROR] Read the output above, correct the cause, and run this script again.\n' >&2
    exit "$status"
}
trap on_error ERR

# Require the expected project-relative working directory before changing anything.
CURRENT_DIR="$(pwd -P)"
CURRENT_NAME="$(basename -- "$CURRENT_DIR")"
PARENT_NAME="$(basename -- "$(dirname -- "$CURRENT_DIR")")"

if [[ "$CURRENT_NAME" != "$EXPECTED_WORK_DIR" || "$PARENT_NAME" != "$EXPECTED_REPO_DIR" ]]; then
    fail "Wrong working directory: ${CURRENT_DIR}
Run this script from the project's '${EXPECTED_REPO_DIR}/${EXPECTED_WORK_DIR}' directory.
For example:
  cd /path/to/${EXPECTED_REPO_DIR}/${EXPECTED_WORK_DIR}
  bash setup_resnet_conversion_venv.sh"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "$SCRIPT_DIR" != "$CURRENT_DIR" ]]; then
    fail "The script must also be located in the current '${EXPECTED_REPO_DIR}/${EXPECTED_WORK_DIR}' directory."
fi

# This environment targets CPython 3.11 on 64-bit Linux x86-64 (WSL Ubuntu).
MACHINE_ARCH="$(uname -m)"
if [[ "$MACHINE_ARCH" != "x86_64" ]]; then
    fail "Unsupported machine architecture: ${MACHINE_ARCH}.
This setup targets Linux x86-64, as expected for Ubuntu running under WSL on a typical PC."
fi

if ! command -v pyenv >/dev/null 2>&1; then
    fail "pyenv is not available in PATH.
Install pyenv and configure it in your shell startup files, then open a new terminal and rerun this script.
See: https://github.com/pyenv/pyenv#installation"
fi

if [[ -n "${PYENV_VERSION:-}" ]]; then
    fail "PYENV_VERSION is set to '${PYENV_VERSION}' and would override the local Python selection.
Run 'unset PYENV_VERSION' in this terminal, then rerun this script."
fi

log "Checking for an installed pyenv Python 3.11 release..."
if ! pyenv versions --bare | grep -Eq '^3\.11\.[0-9]+$'; then
    fail "No installed pyenv Python 3.11.x version was found.
Install one with:
  pyenv install 3.11
Then rerun this script from ${EXPECTED_REPO_DIR}/${EXPECTED_WORK_DIR}.
If the build fails, install pyenv's Python build dependencies first."
fi

# pc-wsl is a separate project directory, so its Python pin is independent of nanopi/.python-version.
log "Selecting Python 3.11 for this project with 'pyenv local 3.11'..."
pyenv local 3.11

CURRENT_PYTHON_PIN="$(tr -d '\r\n' < .python-version)"
if [[ "$CURRENT_PYTHON_PIN" != "3.11" ]]; then
    fail "Expected .python-version to contain '3.11' after 'pyenv local 3.11', found '${CURRENT_PYTHON_PIN}'."
fi

PYTHON_BIN="$(pyenv which python)"
if [[ ! -x "$PYTHON_BIN" ]]; then
    fail "pyenv selected Python, but the interpreter could not be found at: ${PYTHON_BIN}"
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ "$PYTHON_VERSION" != 3.11.* ]]; then
    fail "pyenv selected Python ${PYTHON_VERSION}, not Python 3.11.x.
Check .python-version and your installed pyenv versions, then rerun the script."
fi
log "Using Python ${PYTHON_VERSION}: ${PYTHON_BIN}"

# Reuse a valid environment. Repair an interrupted creation in place when safe;
# never delete the environment or create a backup automatically.
if [[ -e "$VENV_DIR" && ! -d "$VENV_DIR" ]]; then
    fail "${VENV_DIR} already exists but is not a directory. Inspect it before rerunning this script."
fi

if [[ -d "$VENV_DIR" ]]; then
    if [[ ! -f "${VENV_DIR}/pyvenv.cfg" ]]; then
        unexpected_entry=""
        for entry in "${VENV_DIR}"/* "${VENV_DIR}"/.[!.]* "${VENV_DIR}"/..?*; do
            [[ -e "$entry" ]] || continue
            entry_name="$(basename -- "$entry")"
            case "$entry_name" in
                bin|include|lib|lib64)
                    ;;
                *)
                    unexpected_entry="$entry_name"
                    break
                    ;;
            esac
        done

        if [[ -n "$unexpected_entry" ]]; then
            fail "${VENV_DIR} exists without pyvenv.cfg and contains unexpected entry '${unexpected_entry}'.
The script will not modify or delete an unknown directory. Inspect it and remove or rename it manually if appropriate, then rerun."
        fi

        log "Found a partially created ${VENV_DIR} without pyvenv.cfg. Attempting an in-place repair..."
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi

    if [[ ! -f "${VENV_DIR}/pyvenv.cfg" ]]; then
        fail "${VENV_DIR}/pyvenv.cfg is missing after the repair attempt. Inspect the error above and rerun after correcting the cause."
    fi

    CONFIGURED_MINOR="$(sed -nE 's/^version = ([0-9]+\.[0-9]+)\..*/\1/p' "${VENV_DIR}/pyvenv.cfg" | head -n 1)"
    if [[ -z "$CONFIGURED_MINOR" ]]; then
        fail "Could not determine the base Python version from ${VENV_DIR}/pyvenv.cfg. Inspect the environment before continuing."
    fi
    if [[ "$CONFIGURED_MINOR" != "3.11" ]]; then
        fail "${VENV_DIR} was created with Python ${CONFIGURED_MINOR}, not Python 3.11.
To avoid mixing packages from different Python versions, the script will not replace it automatically. Inspect the environment and remove it manually only if you intend to rebuild it, then rerun."
    fi

    if [[ -x "$VENV_PYTHON" ]]; then
        if VENV_MINOR="$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"; then
            if [[ "$VENV_MINOR" == "3.11" ]]; then
                log "Reusing the existing Python ${VENV_MINOR} environment at ${VENV_DIR}."
            else
                fail "The interpreter in ${VENV_DIR} reports Python ${VENV_MINOR}, not Python 3.11. The script will not mix packages from different Python versions. Inspect the environment and remove it manually only if you intend to rebuild it, then rerun."
            fi
        else
            log "The existing environment's Python interpreter does not start. Attempting an in-place repair..."
            "$PYTHON_BIN" -m venv "$VENV_DIR"
        fi
    else
        log "The existing environment is missing its Python executable. Attempting an in-place repair..."
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
else
    log "Creating virtual environment at ${VENV_DIR}..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if [[ ! -x "$VENV_PYTHON" || ! -f "$ACTIVATE_FILE" || ! -f "${VENV_DIR}/pyvenv.cfg" ]]; then
    fail "The virtual environment is still incomplete. Expected '${VENV_PYTHON}', '${ACTIVATE_FILE}', and '${VENV_DIR}/pyvenv.cfg'. Inspect the error above, correct the cause, and rerun."
fi

VENV_VERSION="$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ "$VENV_VERSION" != 3.11.* ]]; then
    fail "${VENV_DIR} is using Python ${VENV_VERSION}, not Python 3.11.x. The environment was not deleted or replaced."
fi

# Use the venv interpreter directly; activation is not required for installation.
if ! "$VENV_PYTHON" -m pip --version; then
    log "pip is not available in the virtual environment. Bootstrapping it with ensurepip..."
    "$VENV_PYTHON" -m ensurepip --upgrade
fi

log "Upgrading pip and wheel..."
"$VENV_PYTHON" -m pip install --upgrade pip wheel

log "Installing the pinned conversion and RKNN support dependencies (excluding PyTorch and RKNN Toolkit)..."
"$VENV_PYTHON" -m pip install \
    "setuptools==80.10.2" \
    "numpy==1.26.4" \
    "Pillow==13.0.0" \
    "protobuf==4.25.4" \
    "psutil==7.2.2" \
    "scikit-learn==1.7.1" \
    "ruamel.yaml==0.19.1" \
    "scipy==1.17.1" \
    "tqdm==4.70.1" \
    "opencv-python==4.11.0.86" \
    "fast-histogram==0.14" \
    "huggingface_hub==2.3.0" \
    "tensorflow==2.15.1" \
    "onnx==1.18.0" \
    "onnxruntime==1.22.1" \
    "tf2onnx==1.17.0"

log "Installing the CPU-only PyTorch 2.4.0 build..."
"$VENV_PYTHON" -m pip install \
    "torch==2.4.0" \
    --index-url https://download.pytorch.org/whl/cpu

log "Installing RKNN Toolkit 2 version 2.3.2 without dependency resolution..."
log "The published PyPI wheel for CPython 3.11/Linux x86-64 will be used; no Git clone is needed."
"$VENV_PYTHON" -m pip install --no-deps "rknn-toolkit2==2.3.2"

log "Verifying required imports and exact package versions..."
"$VENV_PYTHON" - <<'PYTHON_CHECK'
from importlib import import_module, metadata

packages = (
    ("setuptools", "setuptools", "80.10.2"),
    ("numpy", "numpy", "1.26.4"),
    ("Pillow", "PIL", "13.0.0"),
    ("protobuf", "google.protobuf", "4.25.4"),
    ("psutil", "psutil", "7.2.2"),
    ("scikit-learn", "sklearn", "1.7.1"),
    ("ruamel.yaml", "ruamel.yaml", "0.19.1"),
    ("scipy", "scipy", "1.17.1"),
    ("tqdm", "tqdm", "4.70.1"),
    ("opencv-python", "cv2", "4.11.0.86"),
    ("fast-histogram", "fast_histogram", "0.14"),
    ("huggingface_hub", "huggingface_hub", "2.3.0"),
    ("tensorflow", "tensorflow", "2.15.1"),
    ("onnx", "onnx", "1.18.0"),
    ("onnxruntime", "onnxruntime", "1.22.1"),
    ("tf2onnx", "tf2onnx", "1.17.0"),
    ("rknn-toolkit2", "rknn.api", "2.3.2"),
)

for distribution, module_name, expected_version in packages:
    print(f"Checking {distribution} ({module_name})...", flush=True)
    module = import_module(module_name)
    installed_version = metadata.version(distribution)
    if installed_version != expected_version:
        raise RuntimeError(
            f"{distribution}: expected {expected_version}, found {installed_version}"
        )
    print(
        f"  OK: version {installed_version}; module path: {module.__file__}",
        flush=True,
    )

print("Checking CPU-only PyTorch...", flush=True)
import torch

torch_version = metadata.version("torch")
if torch_version != "2.4.0+cpu":
    raise RuntimeError(
        f"Expected CPU-only PyTorch 2.4.0+cpu, found {torch_version}"
    )
if torch.version.cuda is not None:
    raise RuntimeError(
        f"Expected CPU-only PyTorch, but torch.version.cuda is {torch.version.cuda!r}"
    )
print(f"  OK: torch {torch_version}; CUDA support is absent.", flush=True)

print("All required imports and pinned package versions are valid.")
PYTHON_CHECK

log "Checking installed dependency consistency with pip check..."
"$VENV_PYTHON" -m pip check

log "Setup completed successfully."
printf 'Python environment: %s\n' "$VENV_DIR"
printf 'Python interpreter: %s\n' "$VENV_PYTHON"
printf 'Project Python pin: %s\n' "$(cat .python-version)"
printf 'To activate it in your current terminal, run:\n  source %s/bin/activate\n' "$VENV_DIR"
