#!/usr/bin/env bash
# adj_audio.sh - adjust audio/video sync without re-encoding
# Usage: ./adj_audio.sh "/path/to/video.mkv" <offset_ms>
#   offset_ms > 0 : audio delayed
#   offset_ms < 0 : audio earlier (video is delayed instead)

# Path to ffmpeg: edit this line, or override it from the shell,
# e.g. FFMPEG=/opt/ffmpeg/bin/ffmpeg ./adj_audio.sh ...
FFMPEG="${FFMPEG:-/opt/rkmpp-stack/bin/ffmpeg}"

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 \"/path/to/video.mkv\" <offset_ms>" >&2
    exit 1
fi

in="$1"
ms="$2"

if ! command -v "$FFMPEG" >/dev/null 2>&1; then
    echo "Error: ffmpeg not found or not executable: $FFMPEG" >&2
    exit 1
fi

if [ ! -f "$in" ]; then
    echo "Error: input file does not exist: $in" >&2
    exit 1
fi

if ! [[ "$ms" =~ ^-?[0-9]+$ ]]; then
    echo "Error: offset must be an integer in milliseconds (e.g. -60)" >&2
    exit 1
fi

out="${in%.*}_synced.${in##*.}"

if [ -e "$out" ]; then
    echo "Error: output file already exists: $out" >&2
    exit 1
fi

abs_ms="${ms#-}"
sec=$(awk -v ms="$abs_ms" 'BEGIN { printf "%.3f", ms/1000 }')

if (( ms < 0 )); then
    # Audio earlier => delay video by the absolute value
    echo "Delaying video by ${sec}s (equivalent to audio -${abs_ms}ms)"
    "$FFMPEG" -hide_banner -itsoffset "$sec" -i "$in" -i "$in" \
        -map 0:v -map 1:a? -map 0:s? -map 0:t? \
        -map_metadata 0 -map_chapters 0 \
        -c copy "$out"
else
    # Audio delayed (or no change)
    echo "Delaying audio by ${sec}s"
    "$FFMPEG" -hide_banner -i "$in" -itsoffset "$sec" -i "$in" \
        -map 0:v -map 1:a? -map 0:s? -map 0:t? \
        -map_metadata 0 -map_chapters 0 \
        -c copy "$out"
fi

echo "Done: $out"
