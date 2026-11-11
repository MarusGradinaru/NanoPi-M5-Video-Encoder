#!/usr/bin/env bash
#
# Install SSH aliases for the NanoPi and the NAS server.
# Usage: install-ssh-alias.sh <nanopi_user> <nanopi_host_ip> <nanopi_key> <nas_user> <nas_host_ip> <nas_key>
# Use a dash (-) when no private key filename is specified.
# Private-key files, when specified, must be located in ~/.ssh/

set -u

SSH_DIR="$HOME/.ssh"
CONFIG="$SSH_DIR/config"
SCRIPT_NAME="$(basename "$0")"

usage() {
    cat <<EOF

Install the NanoPi and NAS SSH aliases in the current user's OpenSSH config.

Usage:
    $SCRIPT_NAME <nanopi_user> <nanopi_host_ip> <nanopi_key> <nas_user> <nas_host_ip> <nas_key>

Examples:
    $SCRIPT_NAME marus 192.168.0.xx nanopi_key marus 192.168.0.xx nas_key
    $SCRIPT_NAME marus 192.168.0.xx - marus 192.168.0.xx -
    $SCRIPT_NAME marus 192.168.0.xx nanopi_key marus 192.168.0.xx -

Use a dash (-) when no private key is specified.
Key filenames must refer to files inside: ~/.ssh/
Existing alias entries are left untouched; the other alias is still processed.

EOF
}

fail() {
    printf '\n[Error] %s\n' "$1" >&2
    shift
    for line in "$@"; do
        printf '%s\n' "$line" >&2
    done
    echo >&2
    exit 1
}

usage_fail() {
    printf '\n[Error] %s\n' "$1" >&2
    usage >&2
    exit 2
}

# --- Argument validation -----------------------------------------------------

if [ "$#" -ne 6 ]; then
    usage >&2
    exit 2
fi

NANOPI_USER="$1"
NANOPI_HOST="$2"
NANOPI_KEY="$3"
NAS_USER="$4"
NAS_HOST="$5"
NAS_KEY="$6"

for arg in "$NANOPI_USER" "$NANOPI_HOST" "$NANOPI_KEY" "$NAS_USER" "$NAS_HOST" "$NAS_KEY"; do
    if [ -z "$arg" ]; then
        usage >&2
        exit 2
    fi
done

# Keep arguments simple: no whitespace allowed.
has_space() { case "$1" in *[[:space:]]*) return 0 ;; *) return 1 ;; esac; }

has_space "$NANOPI_USER" && usage_fail "The NanoPi SSH username must not contain spaces."
has_space "$NANOPI_HOST" && usage_fail "The NanoPi host IP/name must not contain spaces."
has_space "$NANOPI_KEY"  && usage_fail "The NanoPi private-key filename must not contain spaces."
has_space "$NAS_USER"    && usage_fail "The NAS SSH username must not contain spaces."
has_space "$NAS_HOST"    && usage_fail "The NAS host IP/name must not contain spaces."
has_space "$NAS_KEY"     && usage_fail "The NAS private-key filename must not contain spaces."

# --- Prepare ~/.ssh and config -----------------------------------------------

if [ ! -d "$SSH_DIR" ]; then
    mkdir -p "$SSH_DIR" 2>/dev/null && chmod 700 "$SSH_DIR"
fi
[ -d "$SSH_DIR" ] || fail "Could not create or access \"$SSH_DIR\"."

if [ ! -e "$CONFIG" ]; then
    : > "$CONFIG" 2>/dev/null && chmod 600 "$CONFIG"
fi
[ -f "$CONFIG" ] || fail "Could not create or access \"$CONFIG\"."

# --- Resolve optional private keys -------------------------------------------

NANOPI_IDENTITY_FILE=""
if [ "$NANOPI_KEY" != "-" ]; then
    [ -f "$SSH_DIR/$NANOPI_KEY" ] || fail \
        "NanoPi private-key file was not found:" \
        "    \"$SSH_DIR/$NANOPI_KEY\"" \
        "Use - as the third argument if no dedicated key is required."
    NANOPI_IDENTITY_FILE="$SSH_DIR/$NANOPI_KEY"
fi

NAS_IDENTITY_FILE=""
if [ "$NAS_KEY" != "-" ]; then
    [ -f "$SSH_DIR/$NAS_KEY" ] || fail \
        "NAS private-key file was not found:" \
        "    \"$SSH_DIR/$NAS_KEY\"" \
        "Use - as the sixth argument if no dedicated key is required."
    NAS_IDENTITY_FILE="$SSH_DIR/$NAS_KEY"
fi

# Warn (don't fail) if a key has permissions that ssh will reject.
check_key_perms() {
    local f="$1" perms
    [ -n "$f" ] || return 0
    perms="$(stat -c '%a' "$f" 2>/dev/null)" || return 0
    case "$perms" in
        600|400) ;;
        *) echo "[Warning] \"$f\" has permissions $perms; ssh requires 600 or 400 (chmod 600 \"$f\")." ;;
    esac
}

# --- Helpers -----------------------------------------------------------------

# True if a "Host" line in the config already lists this exact alias.
alias_exists() {
    grep -Eiq "^[[:space:]]*Host[[:space:]]+([^#]*[[:space:]])?$1([[:space:]].*)?$" "$CONFIG"
}

# Arguments: alias name, host, user, optional identity-file path.
write_alias() {
    {
        echo
        echo "# Added by $SCRIPT_NAME"
        echo "Host $1"
        echo "    HostName $2"
        echo "    User $3"
        if [ -n "$4" ]; then
            echo "    IdentityFile $4"
            echo "    IdentitiesOnly yes"
        fi
    } >> "$CONFIG"
}

install_alias() {
    local name="$1" host="$2" user="$3" identity="$4"
    if alias_exists "$name"; then
        echo "[Info] SSH alias \"$name\" already exists. It was left unchanged."
        return 0
    fi
    if ! write_alias "$name" "$host" "$user" "$identity"; then
        fail "Could not write the SSH aliases to \"$CONFIG\"." \
             "Check the file permissions and available disk space."
    fi
    echo "[OK] Installed SSH alias \"$name\" for $user@$host."
}

# --- Install -----------------------------------------------------------------

echo
echo "OpenSSH config: \"$CONFIG\""
echo

install_alias "rkvenc-nanopi" "$NANOPI_HOST" "$NANOPI_USER" "$NANOPI_IDENTITY_FILE"
install_alias "rkvenc-nas"    "$NAS_HOST"    "$NAS_USER"    "$NAS_IDENTITY_FILE"

check_key_perms "$NANOPI_IDENTITY_FILE"
check_key_perms "$NAS_IDENTITY_FILE"

echo
echo "SSH aliases are ready."
echo
echo "Test the connections with:"
echo "    ssh rkvenc-nanopi"
echo "    ssh rkvenc-nas"
echo

exit 0
