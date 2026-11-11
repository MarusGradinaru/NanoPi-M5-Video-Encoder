#!/usr/bin/env bash
set -Eeuo pipefail

# Build and install a pinned Rockchip-enabled FFmpeg stack on ARM64 Ubuntu.
# Source repositories are stored beside this script and are expected to be
# excluded from Git by the parent repository's .gitignore.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_PREFIX="/opt/rkmpp-stack"
INSTALL_PREFIX="$DEFAULT_PREFIX"
BUILD_JOBS="$(nproc)"

RKMpp_URL="https://gitee.com/nyanmisaka/mpp.git"
RKMpp_BRANCH="jellyfin-mpp"
RKMpp_COMMIT="a9380ef333102ac318628f83b5f7a460d377749e"

RKRGA_URL="https://gitee.com/nyanmisaka/rga.git"
RKRGA_BRANCH="jellyfin-rga"
RKRGA_COMMIT="1d330cc28551943bed3380261a5a9c6fbd58ff53"

FFMPEG_URL="https://github.com/nyanmisaka/ffmpeg-rockchip.git"
FFMPEG_COMMIT="d90e3a1c18d7929383cf88c1b3da2e2d1c966cbf"

RKMpp_DIR="$SCRIPT_DIR/rkmpp"
RKRGA_DIR="$SCRIPT_DIR/rkrga"
FFMPEG_DIR="$SCRIPT_DIR/ffmpeg"
RKMpp_BUILD_DIR="$RKMpp_DIR/rkmpp_build"
RKRGA_BUILD_DIR="$RKRGA_DIR/build"

MARKER_NAME=".nanopi-video-encoder-ffmpeg-build"
MARKER_PATH=""

info() { printf '\n[INFO] %s\n' "$*"; }
warn() { printf '\n[WARN] %s\n' "$*" >&2; }
die()  { printf '\n[ERROR] %s\n' "$*" >&2; exit 1; }

on_error() {
    local status=$?
    local line="${1:-unknown}"
    printf '\n[ERROR] Command failed at line %s (exit status %s).\n' "$line" "$status" >&2
    printf '[INFO] Fix the reported problem and run this script again to resume.\n' >&2
    exit "$status"
}
trap 'on_error "$LINENO"' ERR

usage() {
    cat <<'USAGE'
Usage: bash setup-ffmpeg.sh [options]

Build and install the pinned Rockchip MPP + RGA + FFmpeg stack.

Options:
  --dir <path>      Installation prefix (default: /opt/rkmpp-stack).
                    Relative paths are resolved from the current directory.
  --jobs <count>    Number of parallel build jobs (default: nproc).
  -j <count>        Short form of --jobs.
  -h, --help        Show this help.

Examples:
  bash setup-ffmpeg.sh
  bash setup-ffmpeg.sh --jobs 4
  bash setup-ffmpeg.sh --dir /opt/rkmpp-stack-test --jobs 3
USAGE
}

while (($#)); do
    case "$1" in
        --dir)
            (($# >= 2)) || die "Option --dir requires a path."
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --dir=*)
            INSTALL_PREFIX="${1#*=}"
            [[ -n "$INSTALL_PREFIX" ]] || die "Option --dir requires a path."
            shift
            ;;
        --jobs|-j)
            (($# >= 2)) || die "Option $1 requires a positive integer."
            BUILD_JOBS="$2"
            shift 2
            ;;
        --jobs=*|-j=*)
            BUILD_JOBS="${1#*=}"
            [[ -n "$BUILD_JOBS" ]] || die "Option --jobs requires a positive integer."
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1. Use --help to see available options."
            ;;
    esac
done

[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || die "Build job count must be a positive integer; got '$BUILD_JOBS'."
[[ -n "$INSTALL_PREFIX" ]] || die "The install prefix cannot be empty."

command -v realpath >/dev/null 2>&1 || die "The 'realpath' command is required (install coreutils)."
INSTALL_PREFIX="$(realpath -m -- "$INSTALL_PREFIX")"
[[ "$INSTALL_PREFIX" != "/" ]] || die "The installation prefix cannot be '/'."
[[ ! "$INSTALL_PREFIX" =~ [[:space:]:] ]] || die "The installation path must not contain whitespace or ':' because it is passed through FFmpeg and pkg-config build flags."

MARKER_PATH="$INSTALL_PREFIX/$MARKER_NAME"

if [[ "$(basename -- "$SCRIPT_DIR")" != "ffmpeg" ]]; then
    die "This script must remain in the project's ffmpeg/ directory. Current directory: $SCRIPT_DIR"
fi

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
    die "This build is intended for native ARM64/aarch64 Ubuntu on the NanoPi. Detected architecture: $(uname -m)."
fi

# Keep the parent project's source tree clean if the installation path is set
# accidentally to the directory containing this script or one of its children.
case "$INSTALL_PREFIX/" in
    "$SCRIPT_DIR/"|"$SCRIPT_DIR/"*)
        die "The install prefix must be outside the script/source directory '$SCRIPT_DIR'. Use --dir with a separate path."
        ;;
esac

marker_has_line() {
    local expected="$1"
    [[ -f "$MARKER_PATH" ]] && grep -Fqx -- "$expected" "$MARKER_PATH"
}

validate_marker_identity() {
    [[ -f "$MARKER_PATH" ]] || return 1
    marker_has_line "format_version=1" || return 1
    marker_has_line "install_prefix=$INSTALL_PREFIX" || return 1
    marker_has_line "rkmpp_commit=$RKMpp_COMMIT" || return 1
    marker_has_line "rkrga_commit=$RKRGA_COMMIT" || return 1
    marker_has_line "ffmpeg_commit=$FFMPEG_COMMIT" || return 1
}

get_marker_status() {
    sed -n 's/^status=//p' "$MARKER_PATH" | head -n 1
}

# Verify binaries without starting an encode/decode job. This is used both for
# the newly built binaries before installation and for an existing installation.
verify_ffmpeg_pair() {
    local ffmpeg_bin="$1"
    local ffprobe_bin="$2"
    local label="$3"
    local version encoders decoders filters buildconf ldd_out
    local codec filter flag

    printf '\n[CHECK] %s\n' "$label"
    [[ -x "$ffmpeg_bin" ]] || { warn "Missing executable: $ffmpeg_bin"; return 1; }
    [[ -x "$ffprobe_bin" ]] || { warn "Missing executable: $ffprobe_bin"; return 1; }

    version="$("$ffmpeg_bin" -hide_banner -version 2>&1)" || {
        warn "FFmpeg failed to start: $ffmpeg_bin"
        printf '%s\n' "$version" >&2
        return 1
    }
    printf '%s\n' "$version" | head -n 1
    "$ffprobe_bin" -hide_banner -version >/dev/null 2>&1 || {
        warn "FFprobe failed to start: $ffprobe_bin"
        return 1
    }

    encoders="$("$ffmpeg_bin" -hide_banner -encoders 2>&1)" || return 1
    decoders="$("$ffmpeg_bin" -hide_banner -decoders 2>&1)" || return 1
    filters="$("$ffmpeg_bin" -hide_banner -filters 2>&1)" || return 1
    buildconf="$("$ffmpeg_bin" -hide_banner -buildconf 2>&1)" || return 1

    # Require the encoders used by the project. The software "mjpeg" encoder is optional.
    for codec in h264_rkmpp hevc_rkmpp mjpeg_rkmpp; do
        if ! grep -Eq "[[:space:]]${codec}[[:space:]]" <<<"$encoders"; then
            warn "Required encoder '$codec' was not found in $ffmpeg_bin."
            return 1
        fi
    done
    for codec in h264_rkmpp hevc_rkmpp mjpeg_rkmpp; do
        if ! grep -Eq "[[:space:]]${codec}[[:space:]]" <<<"$decoders"; then
            warn "Required decoder '$codec' was not found in $ffmpeg_bin."
            return 1
        fi
    done
    for filter in scale_rkrga vpp_rkrga overlay_rkrga; do
        if ! grep -Eq "[[:space:]]${filter}[[:space:]]" <<<"$filters"; then
            warn "Required filter '$filter' was not found in $ffmpeg_bin."
            return 1
        fi
    done
    for flag in --enable-gpl --enable-version3 --enable-libdrm --enable-rkmpp --enable-rkrga; do
        if ! grep -Fq -- "$flag" <<<"$buildconf"; then
            warn "Expected configure option '$flag' was not reported by $ffmpeg_bin."
            return 1
        fi
    done

    printf '%s\n' "$encoders" | grep -E '[[:space:]](h264_rkmpp|hevc_rkmpp|mjpeg_rkmpp)[[:space:]]' || true
    printf '%s\n' "$decoders" | grep -E '[[:space:]](h264_rkmpp|hevc_rkmpp|mjpeg_rkmpp)[[:space:]]' || true
    printf '%s\n' "$filters" | grep -E '[[:space:]](scale_rkrga|vpp_rkrga|overlay_rkrga)[[:space:]]' || true

    ldd_out="$(ldd "$ffmpeg_bin" 2>&1)" || {
        warn "ldd failed for $ffmpeg_bin"
        printf '%s\n' "$ldd_out" >&2
        return 1
    }
    if grep -Fq 'not found' <<<"$ldd_out"; then
        warn "The FFmpeg executable has unresolved shared-library dependencies:"
        printf '%s\n' "$ldd_out" >&2
        return 1
    fi
    if ! grep -q 'rockchip_mpp' <<<"$ldd_out"; then
        warn "The FFmpeg executable does not show a dynamically resolved Rockchip MPP library in ldd output."
        printf '%s\n' "$ldd_out" >&2
        return 1
    fi
    if ! grep -q 'librga' <<<"$ldd_out"; then
        warn "The FFmpeg executable does not show a dynamically resolved RGA library in ldd output."
        printf '%s\n' "$ldd_out" >&2
        return 1
    fi
    if ! grep -F "$INSTALL_PREFIX/lib/" <<<"$ldd_out" | grep -q 'rockchip_mpp'; then
        warn "FFmpeg is not resolving Rockchip MPP from this installation prefix: $INSTALL_PREFIX/lib"
        printf '%s\n' "$ldd_out" >&2
        return 1
    fi
    if ! grep -F "$INSTALL_PREFIX/lib/" <<<"$ldd_out" | grep -q 'librga'; then
        warn "FFmpeg is not resolving RGA from this installation prefix: $INSTALL_PREFIX/lib"
        printf '%s\n' "$ldd_out" >&2
        return 1
    fi

    printf '[CHECK] Shared-library dependencies for %s:\n' "$ffmpeg_bin"
    printf '%s\n' "$ldd_out" | grep -E 'rockchip_mpp|librga|not found' || true

    local probe_ldd
    probe_ldd="$(ldd "$ffprobe_bin" 2>&1)" || {
        warn "ldd failed for $ffprobe_bin"
        printf '%s\n' "$probe_ldd" >&2
        return 1
    }
    if grep -Fq 'not found' <<<"$probe_ldd"; then
        warn "FFprobe has unresolved shared-library dependencies:"
        printf '%s\n' "$probe_ldd" >&2
        return 1
    fi

    printf '[OK] %s passed.\n' "$label"
    return 0
}

# Preflight the destination before doing any work. The marker prevents accidental
# overwrites of unrelated installations and permits resuming an interrupted run.
if [[ -e "$INSTALL_PREFIX" || -L "$INSTALL_PREFIX" ]]; then
    [[ -d "$INSTALL_PREFIX" && ! -L "$INSTALL_PREFIX" ]] || die "Install prefix exists but is not a normal directory: $INSTALL_PREFIX"

    if [[ -e "$MARKER_PATH" || -L "$MARKER_PATH" ]]; then
        [[ -f "$MARKER_PATH" && ! -L "$MARKER_PATH" ]] || die "The installation marker is not a regular file: $MARKER_PATH"
        validate_marker_identity || die "The marker at '$MARKER_PATH' does not match this script's prefix/source pins. Refusing to modify this installation."

        case "$(get_marker_status)" in
            complete)
                if verify_ffmpeg_pair "$INSTALL_PREFIX/bin/ffmpeg" "$INSTALL_PREFIX/bin/ffprobe" "Existing installation"; then
                    info "The Rockchip FFmpeg stack is already installed and passed verification at: $INSTALL_PREFIX"
                    exit 0
                fi
                warn "The installation marker says 'complete', but one or more checks failed. The script will attempt to repair this script-managed installation using the pinned sources."
                    ;;
            in_progress)
                info "A previous installation is marked as incomplete. The script will resume using the same pinned sources."
                ;;
            *)
                die "Unknown installation state in '$MARKER_PATH'. Refusing to modify the prefix."
                ;;
        esac
    else
        prefix_contents="$(sudo find "$INSTALL_PREFIX" -mindepth 1 -maxdepth 1 -print -quit)"
        [[ -z "$prefix_contents" ]] || die "Install prefix already exists and is non-empty but is not managed by this script: $INSTALL_PREFIX. Inspect it and remove or choose another path with --dir."
        info "The existing installation prefix is empty and can be used: $INSTALL_PREFIX"
    fi
else
    info "Installation prefix does not exist yet: $INSTALL_PREFIX"
fi

info "Build directory: $SCRIPT_DIR"
info "Install prefix:  $INSTALL_PREFIX"
info "Parallel jobs:   $BUILD_JOBS"
info "Pinned MPP:      $RKMpp_COMMIT"
info "Pinned RGA:      $RKRGA_COMMIT"
info "Pinned FFmpeg:   $FFMPEG_COMMIT"

info "Installing build dependencies (if not already installed)..."
sudo apt-get install -y git cmake meson ninja-build pkg-config build-essential libdrm-dev

for tool in git cmake meson ninja pkg-config make gcc g++ pkg-config ldd grep sed; do
    command -v "$tool" >/dev/null 2>&1 || die "Required command '$tool' is not available after dependency installation."
done

ensure_source_repo() {
    local name="$1"
    local url="$2"
    local branch="$3"
    local commit="$4"
    local path="$5"
    local current remote shallow

    if [[ ! -e "$path" && ! -L "$path" ]]; then
        info "Cloning $name source repository..."
        if [[ -n "$branch" ]]; then
            git clone --depth=1 --no-checkout --single-branch --branch "$branch" "$url" "$path"
        else
            git clone --depth=1 --no-checkout "$url" "$path"
        fi
    else
        [[ -d "$path" && ! -L "$path" ]] || die "Source path exists but is not a normal directory: $path"
        git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Source path '$path' exists but is not a valid Git repository. Inspect it and remove or rename it manually before retrying."
        remote="$(git -C "$path" remote get-url origin 2>/dev/null || true)"
        [[ "$remote" == "$url" ]] || die "The origin URL in '$path' is '$remote', but this script expects '$url'. Refusing to alter the repository."
        if ! git -C "$path" diff --quiet || ! git -C "$path" diff --cached --quiet; then
            die "Tracked source files in '$path' have local modifications. Commit or revert them before rerunning this script. Untracked build artifacts are permitted."
        fi
    fi

    if ! git -C "$path" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        info "Fetching pinned commit for $name..."
        if ! git -C "$path" fetch --depth=1 origin "$commit"; then
            warn "Direct shallow fetch of $commit was not accepted; fetching the relevant branch history as a fallback."
            if [[ -n "$branch" ]]; then
                shallow="$(git -C "$path" rev-parse --is-shallow-repository 2>/dev/null || printf 'false')"
                if [[ "$shallow" == "true" ]]; then
                    git -C "$path" fetch --unshallow origin "$branch" || git -C "$path" fetch origin "$branch"
                else
                    git -C "$path" fetch origin "$branch"
                fi
            else
                shallow="$(git -C "$path" rev-parse --is-shallow-repository 2>/dev/null || printf 'false')"
                if [[ "$shallow" == "true" ]]; then
                    git -C "$path" fetch --unshallow origin || git -C "$path" fetch origin
                else
                    git -C "$path" fetch origin
                fi
            fi
        fi
    fi

    git -C "$path" cat-file -e "${commit}^{commit}" 2>/dev/null || die "Pinned commit '$commit' is not available in repository '$path'. Check remote availability and retry."
    current="$(git -C "$path" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$current" != "$commit" ]]; then
        info "Checking out pinned $name commit $commit..."
        git -C "$path" checkout --detach "$commit"
    else
        # Still detach HEAD to avoid accidental changes of the pinned revision.
        git -C "$path" checkout --detach "$commit"
    fi

    current="$(git -C "$path" rev-parse HEAD)"
    [[ "$current" == "$commit" ]] || die "Failed to select the expected $name commit in '$path'."
    info "$name source revision: $(git -C "$path" log -1 --format='%h | %ad | %s' --date=short)"
}

ensure_source_repo "MPP" "$RKMpp_URL" "$RKMpp_BRANCH" "$RKMpp_COMMIT" "$RKMpp_DIR"
ensure_source_repo "RGA" "$RKRGA_URL" "$RKRGA_BRANCH" "$RKRGA_COMMIT" "$RKRGA_DIR"
ensure_source_repo "FFmpeg Rockchip" "$FFMPEG_URL" "" "$FFMPEG_COMMIT" "$FFMPEG_DIR"

info "Configuring MPP with CMake..."
if [[ -e "$RKMpp_BUILD_DIR" && ! -f "$RKMpp_BUILD_DIR/CMakeCache.txt" ]]; then
    mpp_build_contents="$(find "$RKMpp_BUILD_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)"
    [[ -z "$mpp_build_contents" ]] || die "MPP build directory exists but has no CMakeCache.txt and is not empty: $RKMpp_BUILD_DIR. Inspect it and remove the incomplete build directory manually before retrying."
fi
cmake -S "$RKMpp_DIR" -B "$RKMpp_BUILD_DIR" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TEST=OFF

info "Building MPP with $BUILD_JOBS parallel job(s)..."
cmake --build "$RKMpp_BUILD_DIR" --parallel "$BUILD_JOBS"

# A marker is written before the first install operation so interruptions during
# any of the three install stages can be recognized and resumed safely.
write_marker() {
    local status="$1"
    sudo install -d -m 755 -- "$INSTALL_PREFIX"
    {
        printf 'format_version=1\n'
        printf 'status=%s\n' "$status"
        printf 'install_prefix=%s\n' "$INSTALL_PREFIX"
        printf 'rkmpp_commit=%s\n' "$RKMpp_COMMIT"
        printf 'rkrga_commit=%s\n' "$RKRGA_COMMIT"
        printf 'ffmpeg_commit=%s\n' "$FFMPEG_COMMIT"
    } | sudo tee "$MARKER_PATH" >/dev/null
    sudo chmod 644 "$MARKER_PATH"
}

info "Marking the installation as in progress, then installing MPP..."
write_marker "in_progress"
sudo cmake --install "$RKMpp_BUILD_DIR"
# The install target may create a root-owned install_manifest.txt in the build tree.
sudo chown -R "$(id -u):$(id -g)" "$RKMpp_BUILD_DIR"

info "Configuring RGA with Meson (build directory inside rkrga/build)..."
if [[ -e "$RKRGA_BUILD_DIR" ]]; then
    [[ -d "$RKRGA_BUILD_DIR" && ! -L "$RKRGA_BUILD_DIR" ]] || die "RGA build path is not a normal directory: $RKRGA_BUILD_DIR"
    if [[ ! -f "$RKRGA_BUILD_DIR/meson-private/coredata.dat" ]]; then
        rga_build_contents="$(find "$RKRGA_BUILD_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)"
        [[ -z "$rga_build_contents" ]] || die "RGA build directory exists but is not a valid Meson build directory: $RKRGA_BUILD_DIR. Inspect it and remove the incomplete build directory manually before retrying."
        meson setup "$RKRGA_BUILD_DIR" "$RKRGA_DIR" \
            --prefix="$INSTALL_PREFIX" \
            --libdir=lib \
            --buildtype=release \
            --default-library=shared \
            -Dcpp_args=-fpermissive \
            -Dlibdrm=false \
            -Dlibrga_demo=false
    else
        meson setup --reconfigure "$RKRGA_BUILD_DIR" "$RKRGA_DIR" \
            --prefix="$INSTALL_PREFIX" \
            --libdir=lib \
            --buildtype=release \
            --default-library=shared \
            -Dcpp_args=-fpermissive \
            -Dlibdrm=false \
            -Dlibrga_demo=false
    fi
else
    meson setup "$RKRGA_BUILD_DIR" "$RKRGA_DIR" \
        --prefix="$INSTALL_PREFIX" \
        --libdir=lib \
        --buildtype=release \
        --default-library=shared \
        -Dcpp_args=-fpermissive \
        -Dlibdrm=false \
        -Dlibrga_demo=false
fi

info "Building RGA with $BUILD_JOBS parallel job(s)..."
meson compile -C "$RKRGA_BUILD_DIR" --jobs "$BUILD_JOBS"
info "Installing RGA..."
sudo meson install -C "$RKRGA_BUILD_DIR"
sudo chown -R "$(id -u):$(id -g)" "$RKRGA_BUILD_DIR"

export PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
info "Checking pkg-config metadata for libdrm, MPP, and RGA..."
for package in libdrm rockchip_mpp librga; do
    if ! pkg-config --exists "$package"; then
        die "pkg-config cannot find '$package' after installing MPP and RGA. Check files in '$INSTALL_PREFIX/lib/pkgconfig'."
    fi
    printf '[INFO] %-14s %s\n' "$package" "$(pkg-config --modversion "$package")"
done

info "Configuring FFmpeg Rockchip..."
cd "$FFMPEG_DIR"
PKG_CONFIG_PATH="$PKG_CONFIG_PATH" ./configure \
    --prefix="$INSTALL_PREFIX" \
    --extra-cflags="-I$INSTALL_PREFIX/include" \
    --extra-ldflags="-L$INSTALL_PREFIX/lib -Wl,-rpath,$INSTALL_PREFIX/lib" \
    --enable-gpl \
    --enable-version3 \
    --enable-libdrm \
    --enable-rkmpp \
    --enable-rkrga

info "Building FFmpeg with $BUILD_JOBS parallel job(s)..."
make -j "$BUILD_JOBS"

if ! verify_ffmpeg_pair "$FFMPEG_DIR/ffmpeg" "$FFMPEG_DIR/ffprobe" "Build-tree FFmpeg verification"; then
    die "The locally built FFmpeg did not pass verification. It will not be installed. Review the output above, fix the problem, and rerun the script."
fi

info "Installing FFmpeg into $INSTALL_PREFIX..."
sudo make install

if ! verify_ffmpeg_pair "$INSTALL_PREFIX/bin/ffmpeg" "$INSTALL_PREFIX/bin/ffprobe" "Installed FFmpeg verification"; then
    die "The installed FFmpeg stack did not pass verification. The installation marker remains 'in_progress'; fix the issue and rerun this script to resume."
fi

write_marker "complete"
info "Installation completed successfully."
info "FFmpeg:  $INSTALL_PREFIX/bin/ffmpeg"
info "FFprobe: $INSTALL_PREFIX/bin/ffprobe"
info "Sources and build directories are retained under: $SCRIPT_DIR"
