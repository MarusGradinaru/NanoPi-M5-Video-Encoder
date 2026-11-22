#!/usr/bin/env bash
# Process a source video when needed, then encode and mux its validated segments.

set -o pipefail

# Keep both Python scripts and the local fallback configuration alongside this wrapper.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROCESS_SCRIPT="$SCRIPT_DIR/process_video.py"
ENCODE_SCRIPT="$SCRIPT_DIR/encode_segments.py"

# Temporary Windows-uploaded configuration takes priority over the local INI file.
SHM_CONFIG="/dev/shm/process-and-encode.ini"
LOCAL_CONFIG="$SCRIPT_DIR/process-and-encode.ini"
CONFIG_FILE=""
CONFIG_SOURCE=""

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

bool_text() {
    case "$1" in
        1) printf 'Yes' ;;
        0) printf 'No' ;;
        *) printf '%s' "$1" ;;
    esac
}

fail() {
    printf '\n[Error] %s\n' "$1" >&2
    exit "${2:-1}"
}

if [[ -e "$SHM_CONFIG" || -L "$SHM_CONFIG" ]]; then
    if [[ ! -f "$SHM_CONFIG" || -L "$SHM_CONFIG" ]]; then
        fail "RAM configuration path is not a regular file: $SHM_CONFIG" 2
    fi
    CONFIG_FILE="$SHM_CONFIG"
    CONFIG_SOURCE="RAM upload"

    cleanup_uploaded_config() {
        if ! rm -f -- "$SHM_CONFIG"; then
            printf '[Warning] Could not remove temporary configuration: %s\n' "$SHM_CONFIG" >&2
        fi
    }

    # Remove the uploaded INI on normal exit and after handled termination signals.
    trap cleanup_uploaded_config EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
elif [[ -f "$LOCAL_CONFIG" ]]; then
    CONFIG_FILE="$LOCAL_CONFIG"
    CONFIG_SOURCE="local file"
else
    fail "Configuration file not found in either location: $SHM_CONFIG or $LOCAL_CONFIG" 2
fi

# Parse the selected INI file without evaluating its contents as shell code.
CONFIG_KEYS=()
CONFIG_VALUES=()
SEEN_CONFIG_KEYS=()

trim_whitespace() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    TRIMMED_VALUE="$value"
}

config_get() {
    local requested_key="$1"
    local index
    CONFIG_FOUND=0
    CONFIG_VALUE=""
    for index in "${!CONFIG_KEYS[@]}"; do
        if [[ "${CONFIG_KEYS[$index]}" == "$requested_key" ]]; then
            CONFIG_FOUND=1
            CONFIG_VALUE="${CONFIG_VALUES[$index]}"
            return 0
        fi
    done
    return 1
}

is_known_config_key() {
    case "$1" in
        target-class|threshold|review-thr|batch-size|deint|verbose|log-class-work|log-class-frames|\
        qp|tier|level|edge-trim|scan-type|sel-begin|sel-end|offset|audio-delay|log-enc-work)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

load_config() {
    local line key value previous_key
    local line_number=0

    [[ -r "$CONFIG_FILE" ]] || fail "Configuration file is not readable: $CONFIG_FILE" 2

    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_number += 1))

        # Accept an optional UTF-8 BOM commonly added by Windows text editors.
        if (( line_number == 1 )); then
            line="${line#$'\xEF\xBB\xBF'}"
        fi

        # Remove the first comment marker and everything to its right, then trim.
        line="${line%%#*}"
        trim_whitespace "$line"
        line="$TRIMMED_VALUE"
        [[ -z "$line" ]] && continue

        [[ "$line" == *=* ]] || \
            fail "Invalid INI syntax at line $line_number: expected 'key = value'." 2

        key="${line%%=*}"
        value="${line#*=}"
        trim_whitespace "$key"
        key="$TRIMMED_VALUE"
        trim_whitespace "$value"
        value="$TRIMMED_VALUE"

        [[ -n "$key" ]] || fail "Empty parameter name at INI line $line_number." 2
        [[ -n "$value" ]] || fail "Empty value for '$key' at INI line $line_number." 2

        for previous_key in "${SEEN_CONFIG_KEYS[@]}"; do
            [[ "$previous_key" != "$key" ]] || \
                fail "Duplicate INI parameter '$key' at line $line_number." 2
        done
        SEEN_CONFIG_KEYS+=("$key")

        if ! is_known_config_key "$key"; then
            printf '[Warning] Unknown INI parameter at line %d: %s (ignored)\n' \
                "$line_number" "$key" >&2
            continue
        fi

        CONFIG_KEYS+=("$key")
        CONFIG_VALUES+=("$value")
    done < "$CONFIG_FILE"
}

parse_signed_integer() {
    local value="$1"
    local sign=1 digits

    if [[ "$value" == -* ]]; then
        sign=-1
        digits="${value#-}"
    elif [[ "$value" == +* ]]; then
        digits="${value#+}"
    else
        digits="$value"
    fi

    [[ "$digits" =~ ^[0-9]+$ ]] || return 1
    digits="${digits#"${digits%%[!0]*}"}"
    [[ -n "$digits" ]] || digits=0
    # Keep arithmetic safely within Bash's signed 64-bit integer range.
    (( ${#digits} <= 15 )) || return 1
    PARSED_INTEGER=$((sign * (10#$digits)))
}

validate_integer_parameter() {
    local key="$1" minimum="$2" maximum="${3:-}"
    config_get "$key" || return 0
    parse_signed_integer "$CONFIG_VALUE" || \
        fail "INI parameter '$key' must be a signed integer; received '$CONFIG_VALUE'." 2
    if (( PARSED_INTEGER < minimum )); then
        fail "INI parameter '$key' must be at least $minimum; received '$CONFIG_VALUE'." 2
    fi
    if [[ -n "$maximum" ]] && (( PARSED_INTEGER > maximum )); then
        fail "INI parameter '$key' must be at most $maximum; received '$CONFIG_VALUE'." 2
    fi
}

validate_boolean_parameter() {
    local key="$1"
    config_get "$key" || return 0
    [[ "$CONFIG_VALUE" == 0 || "$CONFIG_VALUE" == 1 ]] || \
        fail "INI parameter '$key' must be 0 or 1; received '$CONFIG_VALUE'." 2
}

validate_timestamp_parameter() {
    local key="$1" special_value="$2"
    config_get "$key" || return 0
    if [[ "$CONFIG_VALUE" == "$special_value" ]]; then
        return 0
    fi
    [[ "$CONFIG_VALUE" =~ ^[0-9]{2,}:[0-5][0-9]:[0-5][0-9]\.[0-9]{3}$ ]] || \
        fail "INI parameter '$key' must be '$special_value' or HH:MM:SS.mmm; received '$CONFIG_VALUE'." 2
}

# Read and validate all configured values before touching the video project.
load_config

config_get target-class || fail "Required INI parameter 'target-class' is missing." 2
TARGET_CLASS="$CONFIG_VALUE"

THRESHOLD="0.75"
config_get threshold && THRESHOLD="$CONFIG_VALUE"
[[ "$THRESHOLD" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || \
    fail "INI parameter 'threshold' must be a numeric probability; received '$THRESHOLD'." 2
awk -v value="$THRESHOLD" 'BEGIN { exit !(value > 0 && value < 1) }' || \
    fail "INI parameter 'threshold' must be greater than 0 and less than 1." 2

REVIEW_THR="0"
config_get review-thr && REVIEW_THR="$CONFIG_VALUE"
[[ "$REVIEW_THR" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || \
    fail "INI parameter 'review-thr' must be numeric; received '$REVIEW_THR'." 2
awk -v review="$REVIEW_THR" -v threshold="$THRESHOLD" \
    'BEGIN { exit !(review == 0 || (review > 0 && review < threshold)) }' || \
    fail "INI parameter 'review-thr' must be 0 or greater than 0 and less than 'threshold'." 2

validate_integer_parameter batch-size 1
validate_integer_parameter qp 0
validate_integer_parameter edge-trim 0 1000
validate_integer_parameter offset -999999999999999 999999999999999
validate_integer_parameter audio-delay -999999999999999 999999999999999

for boolean_key in deint verbose log-class-work log-class-frames log-enc-work; do
    validate_boolean_parameter "$boolean_key"
done

if config_get tier; then
    [[ "$CONFIG_VALUE" == main || "$CONFIG_VALUE" == high ]] || \
        fail "INI parameter 'tier' must be 'main' or 'high'; received '$CONFIG_VALUE'." 2
fi
if config_get level; then
    case "$CONFIG_VALUE" in
        1|2|2.1|3|3.1|4|4.1|5|5.1|5.2|6|6.1|6.2) ;;
        *) fail "Unsupported HEVC level in INI parameter 'level': $CONFIG_VALUE." 2 ;;
    esac
fi
if config_get scan-type; then
    case "$CONFIG_VALUE" in
        auto|detect|progressive|interlaced) ;;
        *) fail "Unsupported value for INI parameter 'scan-type': $CONFIG_VALUE." 2 ;;
    esac
fi
validate_timestamp_parameter sel-begin start
validate_timestamp_parameter sel-end end

# Build argument arrays. Every value remains one argument; no eval or shell splitting.
PROCESS_ARGS=(--target-class "$TARGET_CLASS")
config_get threshold && PROCESS_ARGS+=(--threshold "$CONFIG_VALUE")
config_get review-thr && PROCESS_ARGS+=(--review-thr "$CONFIG_VALUE")
config_get batch-size && PROCESS_ARGS+=(--batch-size "$CONFIG_VALUE")
config_get deint && PROCESS_ARGS+=(--deint "$CONFIG_VALUE")
config_get verbose && [[ "$CONFIG_VALUE" == 1 ]] && PROCESS_ARGS+=(--verbose)
config_get log-class-work && [[ "$CONFIG_VALUE" == 1 ]] && PROCESS_ARGS+=(--log-work)
config_get log-class-frames && [[ "$CONFIG_VALUE" == 1 ]] && PROCESS_ARGS+=(--log-frames)

ENCODE_ARGS=()
config_get qp && ENCODE_ARGS+=(--qp "$CONFIG_VALUE")
config_get tier && ENCODE_ARGS+=(--tier "$CONFIG_VALUE")
config_get level && ENCODE_ARGS+=(--level "$CONFIG_VALUE")
config_get edge-trim && ENCODE_ARGS+=(--edge-trim "$CONFIG_VALUE")
config_get scan-type && ENCODE_ARGS+=(--scan-type "$CONFIG_VALUE")
config_get sel-begin && [[ "$CONFIG_VALUE" != start ]] && ENCODE_ARGS+=(--sb "$CONFIG_VALUE")
config_get sel-end && [[ "$CONFIG_VALUE" != end ]] && ENCODE_ARGS+=(--se "$CONFIG_VALUE")
config_get offset && ENCODE_ARGS+=(--offset "$CONFIG_VALUE")
config_get audio-delay && ENCODE_ARGS+=(--audio-delay "$CONFIG_VALUE")
config_get log-enc-work && [[ "$CONFIG_VALUE" == 1 ]] && ENCODE_ARGS+=(--log-work)

# Values used in the console header when a setting is omitted from the INI.
BATCH_SIZE="4"; config_get batch-size && BATCH_SIZE="$CONFIG_VALUE"
DEINT="1"; config_get deint && DEINT="$CONFIG_VALUE"
VERBOSE="0"; config_get verbose && VERBOSE="$CONFIG_VALUE"
LOG_CLASS_WORK="0"; config_get log-class-work && LOG_CLASS_WORK="$CONFIG_VALUE"
LOG_CLASS_FRAMES="0"; config_get log-class-frames && LOG_CLASS_FRAMES="$CONFIG_VALUE"
QP="20"; config_get qp && QP="$CONFIG_VALUE"
TIER="high"; config_get tier && TIER="$CONFIG_VALUE"
LEVEL="5.1"; config_get level && LEVEL="$CONFIG_VALUE"
EDGE_TRIM="40"; config_get edge-trim && EDGE_TRIM="$CONFIG_VALUE"
SCAN_TYPE="auto"; config_get scan-type && SCAN_TYPE="$CONFIG_VALUE"
SEL_BEGIN="start"; config_get sel-begin && SEL_BEGIN="$CONFIG_VALUE"
SEL_END="end"; config_get sel-end && SEL_END="$CONFIG_VALUE"
OFFSET="0"; config_get offset && OFFSET="$CONFIG_VALUE"
AUDIO_DELAY="0"; config_get audio-delay && AUDIO_DELAY="$CONFIG_VALUE"
LOG_ENC_WORK="0"; config_get log-enc-work && LOG_ENC_WORK="$CONFIG_VALUE"

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s <absolute-path-to-video>\n' "${0##*/}" >&2
    printf 'Example: %s "/NAS/path/to/video.ts"\n' "${0##*/}" >&2
    exit 2
fi

# The input video must be given as an absolute Linux path.
INPUT_VIDEO="$1"
[[ "$INPUT_VIDEO" = /* ]] || fail "The input video must be given as an absolute path: $INPUT_VIDEO" 2

# Resolve all symbolic links and convert the input path to its canonical form.
INPUT_VIDEO="$(realpath -e -- "$INPUT_VIDEO" 2>/dev/null)" || \
    fail "Input video does not exist or its path could not be resolved: $1" 2

[[ -f "$INPUT_VIDEO" ]] || fail "Input path is not a regular file: $INPUT_VIDEO" 2

VIDEO_FILENAME="${INPUT_VIDEO##*/}"
if [[ "$VIDEO_FILENAME" != *.* || "$VIDEO_FILENAME" == .* ]]; then
    fail "Input filename must have a video extension: $VIDEO_FILENAME" 2
fi
VIDEO_STEM="${VIDEO_FILENAME%.*}"

# The project directory is the folder that contains the real video file.
# The output directory is created next to it, named after it without the extension.
PROJECT_DIR="$(dirname -- "$INPUT_VIDEO")"
OUTPUT_DIR="${PROJECT_DIR%/}/$VIDEO_STEM"
INPUT_VIDEO_NAME="$(realpath --relative-to="$PROJECT_DIR" -- "$INPUT_VIDEO")"
OUTPUT_DIR_NAME="$(realpath --relative-to="$PROJECT_DIR" -- "$OUTPUT_DIR")"

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
print_rule
printf '                    VIDEO PROCESSING AND ENCODING\n'
print_rule
printf 'Project directory   : %s\n' "$PROJECT_DIR"
printf 'Input video         : %s\n' "$INPUT_VIDEO_NAME"
printf 'Output folder       : %s\n' "$OUTPUT_DIR_NAME"
printf 'Python              : %s\n' "$PYTHON_BIN"
printf 'Config INI          : %s (%s)\n' "$CONFIG_FILE" "$CONFIG_SOURCE"
printf ' -target class      : %s\n' "$TARGET_CLASS"
printf ' -thresholds        : %s for validation, %s for review\n' "$THRESHOLD" "$REVIEW_THR"
printf ' -batch size        : %s images per inference\n' "$BATCH_SIZE"
printf ' -classifier deint  : %s\n' "$(bool_text "$DEINT")"
printf ' -classifier logs   : console: %s; frames: %s; verbose: %s\n' \
    "$(bool_text "$LOG_CLASS_WORK")" \
    "$(bool_text "$LOG_CLASS_FRAMES")" \
    "$(bool_text "$VERBOSE")"
printf ' -encoder setup     : QP: %s; tier: %s; level: %s; edge trim: %s ms\n' \
    "$QP" "$TIER" "$LEVEL" "$EDGE_TRIM"
printf ' -video scan type   : %s\n' "$SCAN_TYPE"
printf ' -timing correction : range: %s - %s, offset: %s ms, audio: %s ms\n' \
     "$SEL_BEGIN" "$SEL_END" "$OFFSET" "$AUDIO_DELAY"
printf ' -encoder logs      : console: %s\n' "$(bool_text "$LOG_ENC_WORK")"
printf '\n'

# -----------------------------------------------------------------------------
# Step 1: Extract and classify scenes, unless completed results already exist.
# -----------------------------------------------------------------------------
PROCESSING_ALREADY_COMPLETE=0
if [[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
    # Never clean through a project-directory symlink. The project path must be
    # a real directory directly underneath PROJECT_DIR.
    if [[ -L "$OUTPUT_DIR" ]]; then
        fail "Output path is a symbolic link; refusing to clean or use it: $OUTPUT_DIR" 2
    fi
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        fail "Output path exists but is not a directory: $OUTPUT_DIR" 2
    fi
    # process_video.py creates these outputs after classification succeeds.
    # Require real directory/file entries, not symlinks, before treating the
    # existing project as a completed processing run.
    if [[ -d "$OUTPUT_DIR/seg_validated" && \
          ! -L "$OUTPUT_DIR/seg_validated" && \
          -f "$OUTPUT_DIR/delete_source.sh" && \
          ! -L "$OUTPUT_DIR/delete_source.sh" ]]; then
        PROCESSING_ALREADY_COMPLETE=1
    fi
fi

# Show the Step 1 separator only when a new processing run is needed.
if (( PROCESSING_ALREADY_COMPLETE == 0 )); then
    print_rule
fi

if [[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
    if (( PROCESSING_ALREADY_COMPLETE == 1 )); then
        printf '[Step 1/2] Skipped: completed scene-processing results already exist.\n'
        printf '  Existing results: %s\n' "$OUTPUT_DIR"
    else
        printf '[Cleanup] Processing directory exists, but completed results were not found.\n'
        printf '[Cleanup] Removing its contents before starting a fresh processing run...\n'
        # Remove only entries inside OUTPUT_DIR, including hidden files.
        # find does not follow child symlinks, and rm -rf unlinks a symlink itself.
        find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + || \
            fail "Could not completely clean the incomplete output directory: $OUTPUT_DIR" 2
        printf '[Cleanup] Incomplete processing results removed.\n'
    fi
fi

if (( PROCESSING_ALREADY_COMPLETE == 0 )); then
    printf '[Step 1/2] Detecting scenes and classifying the video...\n'
    print_rule

    "$PYTHON_BIN" "$PROCESS_SCRIPT" \
        "${PROCESS_ARGS[@]}" \
        --out "$OUTPUT_DIR" \
        "$INPUT_VIDEO"
    RC=$?
    if (( RC != 0 )); then
        fail "Scene processing failed with exit code $RC. Encoding was not started." "$RC"
    fi
fi

# Review is needed only after a new processing run. Existing completed results
# have already been reviewed, so proceed directly to encoding in that case.
if (( PROCESSING_ALREADY_COMPLETE == 0 )); then
    printf '\nWhen ready, press any key to encode processed segments...\n\n'
    if [[ -t 0 ]]; then
        IFS= read -r -n 1 -s _key
    else
        fail "Interactive input is required before encoding; run this script from a terminal." 2
    fi
else
    printf '\n'
fi

# -----------------------------------------------------------------------------
# Step 2: Encode validated segments and mux the final video.
# -----------------------------------------------------------------------------
print_rule
printf '[Step 2/2] Encoding processed segments and creating the final video...\n'
print_rule
"$PYTHON_BIN" "$ENCODE_SCRIPT" "${ENCODE_ARGS[@]}" "$OUTPUT_DIR"
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
