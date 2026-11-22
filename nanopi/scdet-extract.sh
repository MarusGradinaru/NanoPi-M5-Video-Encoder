#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_NAME="${0##*/}"
DEFAULT_WIDTH=320
SCENE_THRESHOLD=10

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS] /path/to/video.ts

Extract scene-change thumbnails using the Rockchip MPP/RGA pipeline.

Options:
  --width PIXELS       Output thumbnail width; height keeps the source aspect ratio
                       (default: ${DEFAULT_WIDTH}; width must be an even number).
  --scene-threshold N  FFmpeg scene-change sensitivity from 0 to 100 (default: 10).
  --offset SECONDS     Override the automatically calculated timestamp offset.
  --out DIRECTORY      Output directory (default: <video_name>_scenes beside the video).
  -h, --help           Show this help message and exit.
EOF
}

fail() {
    printf '[Error] %s\n' "$*" >&2
    exit 1
}

WIDTH="$DEFAULT_WIDTH"
SCENE_THRESHOLD="10"
OFFSET_OVERRIDE=""
OUTPUT_DIR=""
VIDEO=""

while (($#)); do
    case "$1" in
        --width)
            (($# >= 2)) || fail "--width requires a value."
            WIDTH="$2"
            shift 2
            ;;
        --width=*)
            WIDTH="${1#*=}"
            shift
            ;;
        --scene-threshold)
            (($# >= 2)) || fail "--scene-threshold requires a value."
            SCENE_THRESHOLD="$2"
            shift 2
            ;;
        --scene-threshold=*)
            SCENE_THRESHOLD="${1#*=}"
            shift
            ;;
        --offset)
            (($# >= 2)) || fail "--offset requires a value."
            OFFSET_OVERRIDE="$2"
            shift 2
            ;;
        --offset=*)
            OFFSET_OVERRIDE="${1#*=}"
            shift
            ;;
        --out)
            (($# >= 2)) || fail "--out requires a directory."
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --out=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            if (($# != 1)) || [[ -n "$VIDEO" ]]; then
                usage >&2
                exit 2
            fi
            VIDEO="$1"
            shift
            ;;
        -* )
            usage >&2
            fail "Unknown option: $1"
            ;;
        *)
            if [[ -n "$VIDEO" ]]; then
                usage >&2
                fail "Only one input video may be specified."
            fi
            VIDEO="$1"
            shift
            ;;
    esac
done

[[ -n "$VIDEO" ]] || { usage >&2; fail "An input video is required."; }
[[ -f "$VIDEO" ]] || fail "Input video not found: $VIDEO"
[[ "$WIDTH" =~ ^[0-9]+$ ]] || fail "--width must be a positive integer."
WIDTH=$((10#$WIDTH))
((WIDTH >= 2)) || fail "--width must be at least 2 pixels."
((WIDTH % 2 == 0)) || fail "--width must be even when using RGA YUV420 output."
[[ "$SCENE_THRESHOLD" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
    fail "--scene-threshold must be a number between 0 and 100."
awk -v threshold="$SCENE_THRESHOLD" 'BEGIN { exit !(threshold >= 0 && threshold <= 100) }' || \
    fail "--scene-threshold must be between 0 and 100."
if [[ -n "$OFFSET_OVERRIDE" ]]; then
    [[ "$OFFSET_OVERRIDE" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
        fail "--offset must be a valid number of seconds."
fi

VIDEO="$(realpath -- "$VIDEO")"
VIDEO_DIR="$(dirname -- "$VIDEO")"
VIDEO_FILE="$(basename -- "$VIDEO")"
VIDEO_STEM="${VIDEO_FILE%.*}"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${VIDEO_DIR}/${VIDEO_STEM}_scenes"
fi
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath -- "$OUTPUT_DIR")"
if [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    fail "Output directory is not empty: $OUTPUT_DIR"
fi

ROCKCHIP_FFMPEG="/opt/rkmpp-stack/bin/ffmpeg"
ROCKCHIP_FFPROBE="/opt/rkmpp-stack/bin/ffprobe"
if [[ -x "$ROCKCHIP_FFMPEG" ]]; then
    FFMPEG="$ROCKCHIP_FFMPEG"
else
    FFMPEG="$(command -v ffmpeg || true)"
fi
if [[ -x "$ROCKCHIP_FFPROBE" ]]; then
    FFPROBE="$ROCKCHIP_FFPROBE"
else
    FFPROBE="$(command -v ffprobe || true)"
fi
[[ -n "$FFMPEG" ]] || fail "FFmpeg was not found."
[[ -n "$FFPROBE" ]] || fail "FFprobe was not found."

ENCODER_LISTING="$("$FFMPEG" -hide_banner -encoders 2>&1)"
if ! grep -Eq '(^|[[:space:]])mjpeg_rkmpp([[:space:]]|$)' <<< "$ENCODER_LISTING"; then
    fail "FFmpeg '$FFMPEG' does not provide the mjpeg_rkmpp encoder."
fi

probe_stream_value() {
    local entries="$1"
    "$FFPROBE" -v quiet -select_streams v:0 -show_entries "$entries" \
        -of default=noprint_wrappers=1:nokey=1 "$VIDEO" | sed -n '1p'
}

probe_format_value() {
    local entries="$1"
    "$FFPROBE" -v quiet -show_entries "$entries" \
        -of default=noprint_wrappers=1:nokey=1 "$VIDEO" | sed -n '1p'
}

SOURCE_WIDTH="$(probe_stream_value 'stream=width')"
SOURCE_HEIGHT="$(probe_stream_value 'stream=height')"
VIDEO_START="$(probe_stream_value 'stream=start_time')"
VIDEO_DURATION="$(probe_stream_value 'stream=duration')"
FORMAT_START="$(probe_format_value 'format=start_time')"
DURATION="$(probe_format_value 'format=duration')"

[[ "$SOURCE_WIDTH" =~ ^[0-9]+$ ]] && ((SOURCE_WIDTH > 0)) || fail "Could not determine source video width."
[[ "$SOURCE_HEIGHT" =~ ^[0-9]+$ ]] && ((SOURCE_HEIGHT > 0)) || fail "Could not determine source video height."
[[ "$VIDEO_START" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || fail "Could not determine video stream start_time."

if [[ ! "$FORMAT_START" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    STREAM_STARTS="$("$FFPROBE" -v quiet -show_entries stream=start_time \
        -of csv=p=0 "$VIDEO" || true)"
    FORMAT_START="$(awk '
        $1 ~ /^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$/ {
            if (!found || $1 < minimum) minimum = $1
            found = 1
        }
        END { if (found) print minimum }
    ' <<< "$STREAM_STARTS")"
    [[ -n "$FORMAT_START" ]] || FORMAT_START="$VIDEO_START"
fi
if [[ ! "$DURATION" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    DURATION="$VIDEO_DURATION"
fi
[[ "$DURATION" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || fail "Could not determine a valid video duration."

# Match the former Python implementation: round the computed height to the nearest even value.
OUTPUT_HEIGHT=$(( (WIDTH * SOURCE_HEIGHT + SOURCE_WIDTH) / (2 * SOURCE_WIDTH) * 2 ))
((OUTPUT_HEIGHT >= 2)) || OUTPUT_HEIGHT=2

if [[ -n "$OFFSET_OVERRIDE" ]]; then
    OFFSET="$(awk -v offset="$OFFSET_OVERRIDE" 'BEGIN { printf "%.9f", offset }')"
else
    OFFSET="$(awk -v video_start="$VIDEO_START" -v format_start="$FORMAT_START" \
        'BEGIN { printf "%.9f", video_start - format_start }')"
fi
SETPTS_EXPR="$(awk -v offset="$OFFSET" 'BEGIN {
    if (offset > 0) printf "PTS+%.9f/TB", offset
    else if (offset < 0) printf "PTS-%.9f/TB", -offset
    else printf "PTS"
}')"

seconds_to_ms() {
    awk -v seconds="$1" 'BEGIN { printf "%.0f", seconds * 1000 }'
}

format_timestamp() {
    local total_ms="$1"
    ((total_ms >= 0)) || fail "Cannot format a negative timestamp: ${total_ms} ms."
    local hours=$((total_ms / 3600000))
    local remainder=$((total_ms % 3600000))
    local minutes=$((remainder / 60000))
    remainder=$((remainder % 60000))
    local seconds=$((remainder / 1000))
    local milliseconds=$((remainder % 1000))
    printf '%02d-%02d-%02d-%03d' "$hours" "$minutes" "$seconds" "$milliseconds"
}

format_progress_time() {
    local total_ms="$1"
    local hours=$((total_ms / 3600000))
    local remainder=$((total_ms % 3600000))
    local minutes=$((remainder / 60000))
    local seconds=$(((remainder % 60000) / 1000))
    printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
}

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/extract_scene_thumbnails.XXXXXXXX")"
cleanup() {
    rm -rf -- "$TEMP_DIR"
}
trap cleanup EXIT

SCENE_FILTER="scale_rkrga=w=${WIDTH}:h=${OUTPUT_HEIGHT}:format=yuv420p:force_yuv=8bit:force_chroma=420p,hwmap=mode=read,format=yuv420p,setpts=${SETPTS_EXPR},scdet=t=${SCENE_THRESHOLD}:s=1,metadata=mode=print:key=lavfi.scd.time:file=scene_times_raw.txt"
INITIAL_FILTER="scale_rkrga=w=${WIDTH}:h=${OUTPUT_HEIGHT}:format=yuv420p:force_yuv=8bit:force_chroma=420p,hwmap=mode=read,format=yuv420p,setpts=${SETPTS_EXPR},showinfo"

printf '[Video] %s\n' "$VIDEO"
printf '[Video] Source: %sx%s; thumbnails: %sx%s; duration: %s s\n' \
    "$SOURCE_WIDTH" "$SOURCE_HEIGHT" "$WIDTH" "$OUTPUT_HEIGHT" "$DURATION"
printf '[Timeline] Applied offset: %s s\n' "$OFFSET"
DURATION_MS="$(seconds_to_ms "$DURATION")"
((DURATION_MS > 0)) || fail "Video duration is too short or invalid."

# Extract the first decoded frame before the scene-change pass. Keep FFmpeg/RGA
# diagnostics in a temporary log so successful runs remain clean.
printf '[FFmpeg] Extracting thumbnails with MPP/RGA...\n'
if ! (
    cd "$TEMP_DIR"
    "$FFMPEG" -hide_banner -nostdin -loglevel info \
        -hwaccel rkmpp -hwaccel_output_format drm_prime -deint 0 \
        -i "$VIDEO" -map 0:v:0 \
        -vf "$INITIAL_FILTER" -frames:v 1 -an -fps_mode vfr \
        -c:v mjpeg_rkmpp -qp_init 90 -f image2 initial_frame.jpg \
        > initial_frame.log 2>&1
); then
    tail -n 20 "$TEMP_DIR/initial_frame.log" >&2 || true
    fail "FFmpeg failed to extract the initial frame."
fi

INITIAL_PTS="$(grep -oE 'pts_time:[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)' "$TEMP_DIR/initial_frame.log" \
    | head -n 1 | sed 's/^pts_time://' || true)"
[[ -n "$INITIAL_PTS" ]] || fail "Could not determine the initial frame timestamp from showinfo."
INITIAL_MS="$(seconds_to_ms "$INITIAL_PTS")"

# FFmpeg reports progress independently of its regular log output. A FIFO lets
# the script update one progress line without exposing FFmpeg/RGA diagnostics.
PROGRESS_FIFO="$TEMP_DIR/ffmpeg_progress.fifo"
mkfifo "$PROGRESS_FIFO"

(
    cd "$TEMP_DIR"
    "$FFMPEG" -hide_banner -nostdin -loglevel error -nostats \
        -stats_period 0.5 -progress "$PROGRESS_FIFO" \
        -hwaccel rkmpp -hwaccel_output_format drm_prime -deint 0 \
        -i "$VIDEO" -map 0:v:0 \
        -vf "$SCENE_FILTER" \
        -an -fps_mode vfr -c:v mjpeg_rkmpp -qp_init 90 \
        -f image2 'frame_%06d.jpg' \
        > "$TEMP_DIR/scene_extraction.log" 2>&1
) &
FFMPEG_PID=$!

PROGRESS_FRAME=0
PROGRESS_OUT_US=0
LAST_PERCENT=0
while IFS='=' read -r key value; do
    case "$key" in
        frame)
            [[ "$value" =~ ^[0-9]+$ ]] && PROGRESS_FRAME="$value"
            ;;
        out_time_us|out_time_ms)
            [[ "$value" =~ ^-?[0-9]+$ ]] && PROGRESS_OUT_US="$value"
            ;;
        progress)
            CURRENT_MS=$((PROGRESS_OUT_US / 1000))
            ((CURRENT_MS < 0)) && CURRENT_MS=0
            ((CURRENT_MS > DURATION_MS)) && CURRENT_MS="$DURATION_MS"
            PERCENT=$((CURRENT_MS * 100 / DURATION_MS))
            LAST_PERCENT="$PERCENT"
            printf '\r[Progress] %3d%% | %s / %s | %s thumbnails' \
                "$PERCENT" "$(format_progress_time "$CURRENT_MS")" \
                "$(format_progress_time "$DURATION_MS")" "$PROGRESS_FRAME"
            ;;
    esac
done < "$PROGRESS_FIFO"

if wait "$FFMPEG_PID"; then
    if ((LAST_PERCENT < 100)); then
        printf '\r[Progress] 100%% | %s / %s | %s thumbnails' \
            "$(format_progress_time "$DURATION_MS")" \
            "$(format_progress_time "$DURATION_MS")" "$PROGRESS_FRAME"
    fi
    printf '\n'
else
    FFMPEG_STATUS=$?
    printf '\n' >&2
    tail -n 30 "$TEMP_DIR/scene_extraction.log" >&2 || true
    fail "FFmpeg failed while extracting scene-change thumbnails (exit code ${FFMPEG_STATUS})."
fi

mapfile -t SCENE_TIMES < <(
    grep -oE 'lavfi\.scd\.time=[0-9]+([.][0-9]+)?' "$TEMP_DIR/scene_times_raw.txt" 2>/dev/null \
        | sed 's/^lavfi\.scd\.time=//' || true
)
mapfile -t FRAME_FILES < <(find "$TEMP_DIR" -maxdepth 1 -type f -name 'frame_*.jpg' -printf '%f\n' | sort -V)

if ((${#FRAME_FILES[@]} != ${#SCENE_TIMES[@]})); then
    fail "Frame/timestamp mismatch: ${#FRAME_FILES[@]} images but ${#SCENE_TIMES[@]} timestamps."
fi

: > "$TEMP_DIR/timestamps.tsv"
declare -A SEEN_TIMESTAMPS=()
for index in "${!FRAME_FILES[@]}"; do
    timestamp_ms="$(seconds_to_ms "${SCENE_TIMES[$index]}")"
    [[ "$timestamp_ms" =~ ^[0-9]+$ ]] || fail "Invalid scene timestamp: ${SCENE_TIMES[$index]}"
    [[ -z "${SEEN_TIMESTAMPS[$timestamp_ms]+x}" ]] || \
        fail "Two scene timestamps round to the same millisecond: $(format_timestamp "$timestamp_ms")."
    SEEN_TIMESTAMPS[$timestamp_ms]=1
    timestamp_name="$(format_timestamp "$timestamp_ms")"
    mv -- "$TEMP_DIR/${FRAME_FILES[$index]}" "$TEMP_DIR/${timestamp_name}.jpg"
    printf '%s\t%s\n' "$timestamp_ms" "$timestamp_name" >> "$TEMP_DIR/timestamps.tsv"
done

if [[ -z "${SEEN_TIMESTAMPS[$INITIAL_MS]+x}" ]]; then
    INITIAL_NAME="$(format_timestamp "$INITIAL_MS")"
    mv -- "$TEMP_DIR/initial_frame.jpg" "$TEMP_DIR/${INITIAL_NAME}.jpg"
    printf '%s\t%s\n' "$INITIAL_MS" "$INITIAL_NAME" >> "$TEMP_DIR/timestamps.tsv"
else
    rm -f -- "$TEMP_DIR/initial_frame.jpg"
fi

mapfile -t SORTED_RECORDS < <(sort -n -k1,1 "$TEMP_DIR/timestamps.tsv")
if ((${#SORTED_RECORDS[@]} == 0)); then
    fail "No thumbnails were extracted."
fi
for index in "${!SORTED_RECORDS[@]}"; do
    IFS=$'\t' read -r START_MS START_NAME <<< "${SORTED_RECORDS[$index]}"
    if ((index + 1 < ${#SORTED_RECORDS[@]})); then
        IFS=$'\t' read -r END_MS _ <<< "${SORTED_RECORDS[$((index + 1))]}"
    else
        END_MS="$DURATION_MS"
    fi
    ((END_MS > START_MS)) || fail "Invalid scene interval: $(format_timestamp "$START_MS") to $(format_timestamp "$END_MS")."
    OUTPUT_NAME="$(format_timestamp "$START_MS") - $(format_timestamp "$END_MS").jpg"
    mv -- "$TEMP_DIR/${START_NAME}.jpg" "$OUTPUT_DIR/$OUTPUT_NAME"
done

printf '[Done] Extracted %s thumbnails into: %s\n' "${#SORTED_RECORDS[@]}" "$OUTPUT_DIR"
