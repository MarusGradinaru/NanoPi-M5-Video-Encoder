#!/usr/bin/env bash
# Process a source video when needed, then encode and mux its validated scenes.

set -o pipefail

# -----------------------------------------------------------------------------
# Configuration (change this according to your needs)
# -----------------------------------------------------------------------------
TARGET_CLASS="andreea-esca"
PROJECT_DIR="/NAS/WDRed2/TV Rec/Pisde/Andreea Esca"
CANDIDATE_THRESHOLD="0.4"

# Native ResNet NPU configuration.
# The matching RKNN model is selected automatically from RESNET_BATCH_SIZE.
RESNET_BATCH_SIZE="4"
RESNET_UPDATE_SIZE="8"

ENCODER_QP="20"
ENCODER_TIER="high"
ENCODER_LEVEL="5.1"

# Keep both Python scripts alongside this wrapper.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROCESS_SCRIPT="$SCRIPT_DIR/process_video.py"
ENCODE_SCRIPT="$SCRIPT_DIR/encode_segments.py"
RESNET_MODELS_DIR="$SCRIPT_DIR/../resnet/models"

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

# RKNN models have a fixed batch dimension. Accept batch 1 or a positive even
# batch size, matching the native ResNetEngine and the exported model naming.
if ! [[ "$RESNET_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
    fail "RESNET_BATCH_SIZE must be a positive integer (1 or a positive even value)." 2
fi
RESNET_BATCH_SIZE=$((10#$RESNET_BATCH_SIZE))
if (( RESNET_BATCH_SIZE < 1 || (RESNET_BATCH_SIZE != 1 && RESNET_BATCH_SIZE % 2 != 0) )); then
    fail "RESNET_BATCH_SIZE must be 1 or a positive even integer (2, 4, 6, ...)." 2
fi

if ! [[ "$RESNET_UPDATE_SIZE" =~ ^[0-9]+$ ]]; then
    fail "RESNET_UPDATE_SIZE must be a positive integer." 2
fi
RESNET_UPDATE_SIZE=$((10#$RESNET_UPDATE_SIZE))
if (( RESNET_UPDATE_SIZE < 1 )); then
    fail "RESNET_UPDATE_SIZE must be greater than zero." 2
fi

# Select the RKNN model that matches the configured batch size. Batch 1 keeps
# the original filename; batch > 1 uses the _batchN suffix.
MODEL_STEM="resnet50v2_feature_extractor_rk3576_fp16_320x180"
if (( RESNET_BATCH_SIZE == 1 )); then
    EXTRACTOR_FILENAME="${MODEL_STEM}.rknn"
else
    EXTRACTOR_FILENAME="${MODEL_STEM}_batch${RESNET_BATCH_SIZE}.rknn"
fi
RESNET_EXTRACTOR="$RESNET_MODELS_DIR/$EXTRACTOR_FILENAME"

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
printf 'ResNet batch size : %s\n' "$RESNET_BATCH_SIZE"
printf 'Progress interval : %s inference calls (%s images)\n' \
    "$RESNET_UPDATE_SIZE" "$((RESNET_BATCH_SIZE * RESNET_UPDATE_SIZE))"
printf 'ResNet extractor  : %s\n' "$RESNET_EXTRACTOR"
printf 'Encoder QP        : %s\n' "$ENCODER_QP"
printf 'HEVC tier / level : %s / %s\n' "$ENCODER_TIER" "$ENCODER_LEVEL"
printf 'Python            : %s\n' "$PYTHON_BIN"
printf '\n'

# -----------------------------------------------------------------------------
# Step 1: Extract and classify scenes, unless completed results already exist.
# -----------------------------------------------------------------------------
PROCESSING_ALREADY_COMPLETE=0
if [[ -e "$PROCESSING_DIR" || -L "$PROCESSING_DIR" ]]; then
    # Never clean through a project-directory symlink. The project path must be
    # a real directory directly underneath PROJECT_DIR.
    if [[ -L "$PROCESSING_DIR" ]]; then
        fail "Processing path is a symbolic link; refusing to clean or use it: $PROCESSING_DIR" 2
    fi
    if [[ ! -d "$PROCESSING_DIR" ]]; then
        fail "Processing path exists but is not a directory: $PROCESSING_DIR" 2
    fi
    if [[ "$PROCESSING_DIR" != "$PROJECT_DIR/"* ]]; then
        fail "Refusing to clean a path outside the configured project directory: $PROCESSING_DIR" 2
    fi
    # process_video.py creates these outputs after classification succeeds.
    # Require real directory/file entries, not symlinks, before treating the
    # existing project as a completed processing run.
    if [[ -d "$PROCESSING_DIR/scenes_validated" && \
          ! -L "$PROCESSING_DIR/scenes_validated" && \
          -f "$PROCESSING_DIR/scene_predictions.xlsx" && \
          ! -L "$PROCESSING_DIR/scene_predictions.xlsx" ]]; then
        PROCESSING_ALREADY_COMPLETE=1
    fi
fi

# Show the Step 1 separator only when a new processing run is needed.
if (( PROCESSING_ALREADY_COMPLETE == 0 )); then
    print_rule
fi

if [[ -e "$PROCESSING_DIR" || -L "$PROCESSING_DIR" ]]; then
    if (( PROCESSING_ALREADY_COMPLETE == 1 )); then
        printf '[Step 1/2] Skipped: completed scene-processing results already exist.\n'
        printf '  Existing results: %s\n' "$PROCESSING_DIR"
    else
        printf '[Cleanup] Processing directory exists, but completed results were not found.\n'
        printf '[Cleanup] Removing its contents before starting a fresh processing run...\n'
        # Remove only entries inside PROCESSING_DIR, including hidden files.
        # find does not follow child symlinks, and rm -rf unlinks a symlink itself.
        find "$PROCESSING_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + || \
            fail "Could not completely clean the incomplete processing directory: $PROCESSING_DIR" 2
        printf '[Cleanup] Incomplete processing results removed.\n'
    fi
fi

if (( PROCESSING_ALREADY_COMPLETE == 0 )); then
    # Do not require the ResNet model when reusing an already completed project.
    [[ -f "$RESNET_EXTRACTOR" ]] || fail \
        "RKNN feature extractor not found: $RESNET_EXTRACTOR. Export/transfer the matching batch-size model first." 2

    printf '[Step 1/2] Detecting scenes and classifying the video...\n'
    print_rule

    "$PYTHON_BIN" "$PROCESS_SCRIPT" \
        --target-class "$TARGET_CLASS" \
        --candidate-threshold "$CANDIDATE_THRESHOLD" \
        --extractor "$RESNET_EXTRACTOR" \
        --batch-size "$RESNET_BATCH_SIZE" \
        --update-size "$RESNET_UPDATE_SIZE" \
        --out "$PROCESSING_DIR" \
        "$INPUT_VIDEO"
    RC=$?
    if (( RC != 0 )); then
        fail "Scene processing failed with exit code $RC. Encoding was not started." "$RC"
    fi
fi

# Review is needed only after a new processing run. Existing completed results
# have already been reviewed, so proceed directly to encoding in that case.
if (( PROCESSING_ALREADY_COMPLETE == 0 )); then
    printf '\nWhen ready, press any key to continue with the next step: Encode processed segments...\n\n'
    if [[ -t 0 ]]; then
        IFS= read -r -n 1 -s _key
    else
        fail "Interactive input is required before encoding; run this script from a terminal." 2
    fi
else
    printf '\n'
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
