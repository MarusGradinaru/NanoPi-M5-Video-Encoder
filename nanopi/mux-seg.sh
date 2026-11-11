#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

FFMPEG="${FFMPEG:-/opt/rkmpp-stack/bin/ffmpeg}"
OUTPUT_NAME="output_video.mkv"

usage() {
    cat <<'USAGE'
Usage:
  ./mux-seg.sh SEGMENT_FOLDER
  ./mux-seg.sh COMMON_PATH RELATIVE_FOLDER [RELATIVE_FOLDER ...]

Arguments:
  SEGMENT_FOLDER       One folder containing timestamp-named MP4 segments.
  COMMON_PATH          Common directory for all following relative folder paths.
  RELATIVE_FOLDER      A segment folder path relative to COMMON_PATH.

Behavior:
  - Finds MP4 segments directly inside each supplied folder (not recursively).
  - Sorts segments chronologically within each folder.
  - When multiple folders are supplied, concatenates folders in command-line order.
  - Rejects invalid segment filenames and overlapping time ranges within any one folder.
  - Creates output_video.mkv in the parent of SEGMENT_FOLDER (single-folder mode),
    or directly in COMMON_PATH (multi-folder mode).

Expected segment filename:
  HH-MM-SS-mmm - HH-MM-SS-mmm.mp4

FFMPEG may be set in the environment to override the default executable.
USAGE
}

error() {
    printf '[Error] %s\n' "$*" >&2
    exit 1
}

if (( $# == 0 )); then
    usage
    exit 2
fi

if [[ ! -x "$FFMPEG" ]]; then
    if command -v ffmpeg >/dev/null 2>&1; then
        FFMPEG="$(command -v ffmpeg)"
    else
        error "FFmpeg was not found at '$FFMPEG' and no ffmpeg executable is available in PATH. Set FFMPEG to its path."
    fi
fi

# Resolve input folders and choose where the combined output belongs.
declare -a SEGMENT_DIRS=()
if (( $# == 1 )); then
    SEGMENT_DIR="$(realpath -e -- "$1")" || error "Could not resolve segment folder: $1"
    [[ -d "$SEGMENT_DIR" ]] || error "Segment path is not a directory: $SEGMENT_DIR"
    SEGMENT_DIRS+=("$SEGMENT_DIR")
    OUTPUT_DIR="$(dirname -- "$SEGMENT_DIR")"
else
    COMMON_DIR="$(realpath -e -- "$1")" || error "Could not resolve common path: $1"
    [[ -d "$COMMON_DIR" ]] || error "Common path is not a directory: $COMMON_DIR"
    OUTPUT_DIR="$COMMON_DIR"
    shift
    for relative_dir in "$@"; do
        [[ "$relative_dir" != /* ]] || error "Expected a relative folder path, got an absolute path: $relative_dir"
        resolved_dir="$(realpath -e -- "$COMMON_DIR/$relative_dir")" \
            || error "Could not resolve segment folder: $COMMON_DIR/$relative_dir"
        [[ -d "$resolved_dir" ]] || error "Segment path is not a directory: $resolved_dir"
        SEGMENT_DIRS+=("$resolved_dir")
    done
fi

[[ -d "$OUTPUT_DIR" && -w "$OUTPUT_DIR" ]] \
    || error "Output directory does not exist or is not writable: $OUTPUT_DIR"

# Return timestamp milliseconds for HH-MM-SS-mmm. Hours may contain more than two digits.
timestamp_to_ms() {
    local timestamp="$1" hours minutes seconds milliseconds
    if [[ ! "$timestamp" =~ ^([0-9]{2,})-([0-9]{2})-([0-9]{2})-([0-9]{3})$ ]]; then
        printf '[Error] Invalid timestamp in segment name: %s\n' "$timestamp" >&2
        return 1
    fi
    hours=$((10#${BASH_REMATCH[1]}))
    minutes=$((10#${BASH_REMATCH[2]}))
    seconds=$((10#${BASH_REMATCH[3]}))
    milliseconds=$((10#${BASH_REMATCH[4]}))
    (( minutes < 60 && seconds < 60 )) || {
        printf '[Error] Invalid minutes or seconds in timestamp: %s\n' "$timestamp" >&2
        return 1
    }
    printf '%s\n' "$(( ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds ))"
}

# Escape a pathname using shell-style escapes understood by FFmpeg's concat demuxer.
# This handles common path characters such as spaces, parentheses, apostrophes and backslashes.
ffconcat_quote() {
    local value="$1" escaped
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] \
        || error "Newline characters in input paths are not supported: $value"
    printf -v escaped '%q' "$value"
    printf '%s' "$escaped"
}

# The global list preserves folder order; files are sorted within each folder only.
declare -a MUX_FILES=()
shopt -s nullglob nocaseglob

for segment_dir in "${SEGMENT_DIRS[@]}"; do
    declare -a folder_paths=() folder_starts=() folder_ends=() folder_start_labels=() folder_end_labels=()

    for segment_path in "$segment_dir"/*; do
        [[ -f "$segment_path" ]] || continue
        extension="${segment_path##*.}"
        [[ "${extension,,}" == "mp4" ]] || continue

        filename="${segment_path##*/}"
        stem="${filename%.*}"
        if [[ ! "$stem" =~ ^([0-9]{2,}-[0-9]{2}-[0-9]{2}-[0-9]{3})\ -\ ([0-9]{2,}-[0-9]{2}-[0-9]{2}-[0-9]{3})$ ]]; then
            error "Invalid segment filename in '$segment_dir': '$filename'. Expected 'HH-MM-SS-mmm - HH-MM-SS-mmm.mp4'."
        fi
        start_label="${BASH_REMATCH[1]}"
        end_label="${BASH_REMATCH[2]}"
        start_ms="$(timestamp_to_ms "$start_label")" || exit 1
        end_ms="$(timestamp_to_ms "$end_label")" || exit 1
        (( end_ms > start_ms )) || error "Segment end must be later than its start: '$filename'."

        folder_paths+=("$segment_path")
        folder_starts+=("$start_ms")
        folder_ends+=("$end_ms")
        folder_start_labels+=("$start_label")
        folder_end_labels+=("$end_label")
    done

    (( ${#folder_paths[@]} > 0 )) || error "No MP4 segments were found directly inside: $segment_dir"

    # Sort by numeric start time, then end time.
    for ((i = 1; i < ${#folder_paths[@]}; i++)); do
        key_path="${folder_paths[i]}"
        key_start="${folder_starts[i]}"
        key_end="${folder_ends[i]}"
        key_start_label="${folder_start_labels[i]}"
        key_end_label="${folder_end_labels[i]}"
        j=$((i - 1))
        while (( j >= 0 )); do
            if (( folder_starts[j] < key_start || (folder_starts[j] == key_start && folder_ends[j] <= key_end) )); then
                break
            fi
            folder_paths[j + 1]="${folder_paths[j]}"
            folder_starts[j + 1]="${folder_starts[j]}"
            folder_ends[j + 1]="${folder_ends[j]}"
            folder_start_labels[j + 1]="${folder_start_labels[j]}"
            folder_end_labels[j + 1]="${folder_end_labels[j]}"
            j=$((j - 1))
        done
        folder_paths[j + 1]="$key_path"
        folder_starts[j + 1]="$key_start"
        folder_ends[j + 1]="$key_end"
        folder_start_labels[j + 1]="$key_start_label"
        folder_end_labels[j + 1]="$key_end_label"
    done

    # Overlap checks are per folder because timestamps from different source folders
    # may refer to different source-video timelines. Folder order remains authoritative.
    for ((i = 1; i < ${#folder_paths[@]}; i++)); do
        previous=$((i - 1))
        if (( folder_starts[i] < folder_ends[previous] )); then
            error "Overlapping segment ranges in '$segment_dir': '${folder_paths[previous]##*/}' (${folder_start_labels[previous]} - ${folder_end_labels[previous]}) and '${folder_paths[i]##*/}' (${folder_start_labels[i]} - ${folder_end_labels[i]}). Correct or remove the overlapping segment files."
        fi
    done

    printf '[Info] %s: %d segment(s)\n' "$segment_dir" "${#folder_paths[@]}"
    MUX_FILES+=("${folder_paths[@]}")
    unset folder_paths folder_starts folder_ends folder_start_labels folder_end_labels
 done

(( ${#MUX_FILES[@]} > 0 )) || error "No segments were found."

# Temporary manifest and output are cleaned up on every exit; an existing output is preserved on failure.
CONCAT_LIST="$(mktemp "${TMPDIR:-/tmp}/mux-seg-list.XXXXXX")" \
    || error "Could not create a temporary FFmpeg concat list."
TEMP_OUTPUT=""
cleanup() {
    [[ -z "${CONCAT_LIST:-}" || ! -e "$CONCAT_LIST" ]] || rm -f -- "$CONCAT_LIST"
    [[ -z "${TEMP_OUTPUT:-}" || ! -e "$TEMP_OUTPUT" ]] || rm -f -- "$TEMP_OUTPUT"
}
trap cleanup EXIT

for segment_path in "${MUX_FILES[@]}"; do
    printf 'file %s\n' "$(ffconcat_quote "$segment_path")" >> "$CONCAT_LIST"
done

TEMP_OUTPUT="$(mktemp --tmpdir="$OUTPUT_DIR" --suffix=.mkv .output_video.XXXXXX)" \
    || error "Could not create a temporary output file in: $OUTPUT_DIR"

printf '\n[Mux] Muxing %d encoded segment(s) in the requested folder order and chronological order within each folder...\n' "${#MUX_FILES[@]}"
if ! "$FFMPEG" -hide_banner -loglevel error -nostats \
    -f concat -safe 0 -i "$CONCAT_LIST" -map 0 -c copy -y "$TEMP_OUTPUT"; then
    error "FFmpeg failed while muxing segments. The existing '$OUTPUT_DIR/$OUTPUT_NAME' was preserved."
fi

chmod 664 -- "$TEMP_OUTPUT" || error "Could not set permissions on the muxed output file."
mv -f -- "$TEMP_OUTPUT" "$OUTPUT_DIR/$OUTPUT_NAME" \
    || error "Could not save the final output to: $OUTPUT_DIR/$OUTPUT_NAME"
TEMP_OUTPUT=""
printf '[Mux] The segments were successfully muxed and saved to: %s\n\n' "$OUTPUT_NAME"
printf '[Done] Muxed %d segment(s) into: %s/%s\n' "${#MUX_FILES[@]}" "$OUTPUT_DIR" "$OUTPUT_NAME"
