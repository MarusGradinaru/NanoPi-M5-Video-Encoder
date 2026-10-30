#!/usr/bin/env bash
# Process a source video, pause for review, then encode and mux its validated scenes.

set -o pipefail

# -----------------------------------------------------------------------------
# Configuration  (change this according to your needs)
# -----------------------------------------------------------------------------
TARGET_CLASS="andreea-esca"
PROJECT_DIR="/NAS/WDRed2/TV Rec/Pisde/Andreea Esca"
CANDIDATE_THRESHOLD="0.4"
ENCODER_QP="20"
ENCODER_TIER="high"
ENCODER_LEVEL="5.1"

# Keep both Python scripts alongside this wrapper.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROCESS_SCRIPT="$SCRIPT_DIR/process_video.py"
ENCODE_SCRIPT="$SCRIPT_DIR/encode_segments.py"

print_rule() {
    printf '%s\n' '=============================================================================='
}

format_elapsed() {
    local total_seconds="$1"
    printf '%02d:%02d:%02d' \
        "$((total_seconds / 3600))" \
        "$(((total_seconds % 3600) / 60))" \
        "$((total_seconds % 60))"
}

fail() {
    printf '\n[Error] %s\n' "$1" >&2
    exit "${2:-1}"
}

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s <video-file-path-or-name>\n' "${0##*/}" >&2
    printf 'Example: %s "/mnt/NAS/path/to/video.ts"\n' "${0##*/}" >&2
    exit 2
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
    fail "Project base directory does not exist: $PROJECT_DIR" 2
fi

# Accept an absolute/existing path, or a filename relative to PROJECT_DIR.
INPUT_ARGUMENT="$1"
if [[ "$INPUT_ARGUMENT" = /* ]]; then
    INPUT_CANDIDATE="$INPUT_ARGUMENT"
elif [[ -f "$PROJECT_DIR/$INPUT_ARGUMENT" ]]; then
    INPUT_CANDIDATE="$PROJECT_DIR/$INPUT_ARGUMENT"
elif [[ -f "$INPUT_ARGUMENT" ]]; then
    INPUT_CANDIDATE="$INPUT_ARGUMENT"
else
    fail "Input video was not found: $INPUT_ARGUMENT (also checked under $PROJECT_DIR)" 2
fi

[[ -f "$INPUT_CANDIDATE" ]] || fail "Input path is not a regular file: $INPUT_CANDIDATE" 2
command -v realpath >/dev/null 2>&1 || fail "Required command 'realpath' is not available." 2
INPUT_VIDEO="$(realpath -- "$INPUT_CANDIDATE")" || fail "Could not resolve input video path." 2

VIDEO_FILENAME="${INPUT_VIDEO##*/}"
if [[ "$VIDEO_FILENAME" != *.* || "$VIDEO_FILENAME" == .* ]]; then
    fail "Input filename must have a video extension: $VIDEO_FILENAME" 2
fi
VIDEO_STEM="${VIDEO_FILENAME%.*}"
PROCESSING_DIR="$PROJECT_DIR/$VIDEO_STEM"

[[ -f "$PROCESS_SCRIPT" ]] || fail "Processing script not found: $PROCESS_SCRIPT" 2
[[ -f "$ENCODE_SCRIPT" ]] || fail "Encoding script not found: $ENCODE_SCRIPT" 2

# Prefer the active environment, then the usual local .venv-resnet.
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -x "$SCRIPT_DIR/.venv-resnet/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv-resnet/bin/python"
else
    fail "Python was not found. Create or activate the required virtual environment." 2
fi

TOTAL_STARTED=$SECONDS

printf '\n'
print_rule
printf '                 VIDEO SCENE PROCESSING AND ENCODING\n'
print_rule
printf 'Input video       : %s\n' "$INPUT_VIDEO"
printf 'Project directory : %s\n' "$PROCESSING_DIR"
printf 'Target class      : %s\n' "$TARGET_CLASS"
printf 'Review threshold  : %s\n' "$CANDIDATE_THRESHOLD"
printf 'Encoder QP        : %s\n' "$ENCODER_QP"
printf 'HEVC tier / level : %s / %s\n' "$ENCODER_TIER" "$ENCODER_LEVEL"
printf 'Python            : %s\n' "$PYTHON_BIN"
printf '\n'

# -----------------------------------------------------------------------------
# Step 1: Extract and classify scenes.
# -----------------------------------------------------------------------------
print_rule
printf '[Step 1/2] Detecting scenes and classifying the video...\n'
print_rule

"$PYTHON_BIN" "$PROCESS_SCRIPT" \
  --target-class "$TARGET_CLASS" \
  --candidate-threshold "$CANDIDATE_THRESHOLD" \
  --out "$PROCESSING_DIR" \
  "$INPUT_VIDEO"
RC=$?
if (( RC != 0 )); then
  fail "Scene processing failed with exit code $RC. Encoding was not started." "$RC"
fi

printf '\nWhen ready, press any key to continue with the next step: Encode processed segments...\n\n'
if [[ -t 0 ]]; then
    IFS= read -r -n 1 -s _key
else
    fail "Interactive input is required before encoding; run this script from a terminal." 2
fi

# -----------------------------------------------------------------------------
# Step 2: Encode validated scenes and mux the final video.
# -----------------------------------------------------------------------------
print_rule
printf '[Step 2/2] Encoding processed segments and creating the final video...\n'
print_rule

"$PYTHON_BIN" "$ENCODE_SCRIPT" "$PROCESSING_DIR" \
  --qp "$ENCODER_QP" \
  --tier "$ENCODER_TIER" \
  --level "$ENCODER_LEVEL"
RC=$?
if (( RC != 0 )); then
  fail "Encoding or final muxing failed with exit code $RC." "$RC"
fi

printf '\n'
print_rule
printf '[Complete] The video processing workflow finished successfully.\n'
printf 'Total runtime: %s (hh:mm:ss)\n' "$(format_elapsed "$((SECONDS - TOTAL_STARTED))")"
print_rule
printf '\n'
