#!/usr/bin/env bash
set -Eeuo pipefail

# Set up the project's Python environment for the ResNet/RKNN tools.
# Run this script from NanoPi-M5-Video-Encoder/nanopi.

readonly EXPECTED_REPO_DIR="NanoPi-M5-Video-Encoder"
readonly EXPECTED_WORK_DIR="nanopi"
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

# Require the exact project-relative working directory before changing anything.
CURRENT_DIR="$(pwd -P)"
CURRENT_NAME="$(basename -- "$CURRENT_DIR")"
PARENT_NAME="$(basename -- "$(dirname -- "$CURRENT_DIR")")"

if [[ "$CURRENT_NAME" != "$EXPECTED_WORK_DIR" || "$PARENT_NAME" != "$EXPECTED_REPO_DIR" ]]; then
    fail "Wrong working directory: ${CURRENT_DIR}
Run this script from the project's '${EXPECTED_REPO_DIR}/${EXPECTED_WORK_DIR}' directory.
For example:
  cd /path/to/${EXPECTED_REPO_DIR}/${EXPECTED_WORK_DIR}
  bash setup_resnet_venv.sh"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "$SCRIPT_DIR" != "$CURRENT_DIR" ]]; then
    fail "The script must also be located in the current '${EXPECTED_REPO_DIR}/${EXPECTED_WORK_DIR}' directory."
fi

# RKNN Toolkit Lite 2 provides its supported Linux wheel for AArch64.
MACHINE_ARCH="$(uname -m)"
case "$MACHINE_ARCH" in
    aarch64|arm64)
        ;;
    *)
        fail "Unsupported machine architecture: ${MACHINE_ARCH}.
This setup targets 64-bit ARM Linux (aarch64), as used by the NanoPi M5."
        ;;
esac

if ! command -v pyenv >/dev/null 2>&1; then
    fail "pyenv is not available in PATH.
Install pyenv and configure it in your shell startup files, then open a new terminal and rerun this script.
See: https://github.com/pyenv/pyenv#installation"
fi

if [[ -n "${PYENV_VERSION:-}" ]]; then
    fail "PYENV_VERSION is set to '${PYENV_VERSION}' and would override the project's local Python selection.
Run 'unset PYENV_VERSION' in this terminal, then rerun this script."
fi

log "Checking for an installed pyenv Python 3.12 release..."
if ! pyenv versions --bare | grep -Eq '^3\.12\.[0-9]+$'; then
    fail "No installed pyenv Python 3.12.x version was found.
Install one with:
  pyenv install 3.12
Then rerun this script from ${EXPECTED_REPO_DIR}/${EXPECTED_WORK_DIR}.
If the build fails, install pyenv's Python build dependencies first."
fi

log "Selecting the latest installed pyenv Python 3.12.x for this project..."
pyenv local 3.12

PYTHON_BIN="$(pyenv which python)"
if [[ ! -x "$PYTHON_BIN" ]]; then
    fail "pyenv selected Python, but the interpreter could not be found at: ${PYTHON_BIN}"
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ "$PYTHON_VERSION" != 3.12.* ]]; then
    fail "pyenv selected Python ${PYTHON_VERSION}, not Python 3.12.x.
Check the project's .python-version and installed pyenv versions, then rerun the script."
fi
log "Using Python ${PYTHON_VERSION}: ${PYTHON_BIN}"

# Reuse a valid environment. If its files are incomplete, attempt an in-place
# repair; never delete the environment or create a backup automatically.
if [[ -e "$VENV_DIR" && ! -d "$VENV_DIR" ]]; then
    fail "${VENV_DIR} already exists but is not a directory. Inspect it before rerunning this script."
fi

if [[ -d "$VENV_DIR" ]]; then
    if [[ ! -f "${VENV_DIR}/pyvenv.cfg" ]]; then
        # Repair an interrupted venv creation only when the directory contains
        # no unexpected top-level entries. Never touch an unrelated directory.
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

        log "Found an empty or partially created ${VENV_DIR} without pyvenv.cfg. Attempting an in-place repair..."
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi

    CONFIGURED_MINOR="$(sed -nE 's/^version = ([0-9]+\.[0-9]+)\..*/\1/p' "${VENV_DIR}/pyvenv.cfg" | head -n 1)"
    if [[ -n "$CONFIGURED_MINOR" && "$CONFIGURED_MINOR" != "3.12" ]]; then
        fail "${VENV_DIR} was created with Python ${CONFIGURED_MINOR}, not Python 3.12.
To avoid mixing packages from different Python versions, the script will not replace it automatically. Inspect the environment, remove it manually if you intend to rebuild it, and rerun."
    fi

    if [[ -x "$VENV_PYTHON" ]]; then
        if VENV_MINOR="$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"; then
            if [[ "$VENV_MINOR" == "3.12" ]]; then
                log "Reusing the existing Python ${VENV_MINOR} environment at ${VENV_DIR}."
            else
                fail "The interpreter in ${VENV_DIR} reports Python ${VENV_MINOR}, not Python 3.12. The script will not mix packages from different Python versions. Inspect the environment, remove it manually if you intend to rebuild it, and rerun."
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

if [[ ! -x "$VENV_PYTHON" || ! -f "$ACTIVATE_FILE" ]]; then
    fail "The virtual environment is still incomplete: expected '${VENV_PYTHON}' and '${ACTIVATE_FILE}'. Inspect the error above, correct the cause, and rerun."
fi

VENV_VERSION="$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ "$VENV_VERSION" != 3.12.* ]]; then
    fail "${VENV_DIR} is using Python ${VENV_VERSION}, not Python 3.12.x. The environment was not deleted or replaced."
fi

# Correct the prompt line only when the expected original pattern is present.
# Allow leading spaces/tabs and preserve the original indentation on replacement.
# If the line was already corrected, or is in an unexpected form, leave it alone.
if grep -Fq 'PS1="${VIRTUAL_ENV_PROMPT}${PS1:-}"' "$ACTIVATE_FILE"; then
    log "The activate prompt fix is already present."
elif grep -Eq '^[[:space:]]*PS1=.*\(\.venv-resnet\).*[$][{]PS1:-[}].*$' "$ACTIVATE_FILE"; then
    log "Applying the Python 3.12 activate prompt fix..."
    sed -i -E 's|^([[:space:]]*)PS1=.*\(\.venv-resnet\).*[$][{]PS1:-[}].*$|\1PS1="${VIRTUAL_ENV_PROMPT}${PS1:-}"|' "$ACTIVATE_FILE"
    if grep -Fq 'PS1="${VIRTUAL_ENV_PROMPT}${PS1:-}"' "$ACTIVATE_FILE"; then
        log "The activate prompt fix was applied."
    else
        warn "The original prompt line was detected, but the expected replacement could not be verified. The script will continue as requested."
    fi
else
    log "The expected original prompt line was not found in activate; leaving that file unchanged."
fi

# Ensure pip is available if this is a partially initialized environment.
if ! "$VENV_PYTHON" -m pip --version; then
    log "pip is not available in the virtual environment. Bootstrapping it with ensurepip..."
    "$VENV_PYTHON" -m ensurepip --upgrade
fi

log "Upgrading pip, setuptools, and wheel..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

log "Installing the pinned project dependencies..."
"$VENV_PYTHON" -m pip install \
    "rknn-toolkit-lite2==2.3.2" \
    "scikit-learn==1.7.1" \
    "Pillow==13.0.0" \
    "openpyxl==3.1.5" \
    "huggingface_hub==2.3.0"

log "Verifying required imports and installed package versions..."
"$VENV_PYTHON" - <<'PYTHON_CHECK'
from importlib import import_module, metadata

packages = (
    ("rknn-toolkit-lite2", "rknnlite.api", "2.3.2"),
    ("scikit-learn", "sklearn", "1.7.1"),
    ("Pillow", "PIL", "13.0.0"),
    ("openpyxl", "openpyxl", "3.1.5"),
    ("huggingface_hub", "huggingface_hub", "2.3.0"),
)

for distribution, module_name, expected_version in packages:
    print(f"Checking {distribution} ({module_name})...", flush=True)
    module = import_module(module_name)
    installed_version = metadata.version(distribution)
    if installed_version != expected_version:
        raise RuntimeError(
            f"{distribution}: expected {expected_version}, found {installed_version}"
        )
    print(f"  OK: version {installed_version}; module path: {module.__file__}", flush=True)

print("All required imports and pinned versions are valid.")
PYTHON_CHECK

log "Setup completed successfully."
printf 'Python environment: %s\n' "$VENV_DIR"
printf 'To activate it in your current terminal, run:\n  source %s/bin/activate\n' "$VENV_DIR"
