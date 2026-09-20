#!/bin/bash
# Fetch the PowerGraph cascades dataset archive used by this validation.
#
# Default source is the official figshare object referenced by the
# PowerGraph-Graph README (article 22820534, file 46619158, dataset_cascades.zip).
#
# Environment overrides:
#   POWERGRAPH_DATA_URL       archive URL (default: figshare ndownloader for file 46619158)
#   POWERGRAPH_DATA_ARCHIVE   output path (default: <package>/data_download/dataset_cascades.zip)
#   POWERGRAPH_DATA_MD5       expected md5 (default: the checksum verified for the
#                             original run); set to "skip" to disable verification
#   POWERGRAPH_FIGSHARE_FILE_ID   figshare file id used by the proxy fallback (default 46619158)
#   POWERGRAPH_FIGSHARE_PROXY     optional HTTP proxy, e.g. http://host:port
#
# The proxy fallback exists because figshare.com answered HTTP 403 for every path
# in the *original validation environment*. It is only used when the direct
# download fails; it never relies on a stored presigned URL (those expire in
# seconds) but re-derives one from the official figshare redirect each time.
#
# Nothing here is committed to git: the archive lands in a git-ignored directory.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VALIDATION_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"

FILE_ID="${POWERGRAPH_FIGSHARE_FILE_ID:-46619158}"
DATA_URL="${POWERGRAPH_DATA_URL:-https://figshare.com/ndownloader/files/${FILE_ID}}"
ARCHIVE="${POWERGRAPH_DATA_ARCHIVE:-$_VALIDATION_ROOT/data_download/dataset_cascades.zip}"
EXPECTED_MD5="${POWERGRAPH_DATA_MD5:-70b677416d2f377ccfee9f51d8369867}"
PROXY="${POWERGRAPH_FIGSHARE_PROXY:-}"

mkdir -p "$(dirname "$ARCHIVE")"

verify() {
    [ "$EXPECTED_MD5" = "skip" ] && return 0
    local got
    got="$(md5sum "$1" | awk '{print $1}')"
    if [ "$got" != "$EXPECTED_MD5" ]; then
        echo "ERROR: md5 mismatch for $1" >&2
        echo "  expected $EXPECTED_MD5" >&2
        echo "  actual   $got" >&2
        return 1
    fi
    echo "md5 OK: $got"
}

if [ -s "$ARCHIVE" ]; then
    if verify "$ARCHIVE"; then
        echo "archive already present and verified: $ARCHIVE"
        exit 0
    fi
    echo "existing archive failed verification; re-downloading" >&2
fi

echo "[1/2] direct download from $DATA_URL"
if curl -fL --retry 2 --connect-timeout 20 --max-time 1800 \
        -o "$ARCHIVE.part" "$DATA_URL" 2>/dev/null; then
    mv "$ARCHIVE.part" "$ARCHIVE"
    if verify "$ARCHIVE"; then
        echo "saved $ARCHIVE"
        exit 0
    fi
    echo "direct download produced an unexpected payload" >&2
else
    rm -f "$ARCHIVE.part"
    echo "direct download failed (this is expected on networks where figshare returns HTTP 403)" >&2
fi

if [ -z "$PROXY" ]; then
    echo "ERROR: direct download failed and POWERGRAPH_FIGSHARE_PROXY is not set." >&2
    echo "       Set POWERGRAPH_FIGSHARE_PROXY=http://host:port, or download" >&2
    echo "       https://figshare.com/ndownloader/files/${FILE_ID} manually into:" >&2
    echo "       $ARCHIVE" >&2
    exit 1
fi

echo "[2/2] proxy fallback: resolving the official figshare redirect through $PROXY"
LOC="$(timeout 60 curl -x "$PROXY" -sI --max-time 55 \
        "https://ndownloader.figshare.com/files/${FILE_ID}" \
        | grep -i '^location:' | sed 's/^[Ll]ocation: *//' | tr -d '\r')"
[ -n "$LOC" ] || { echo "ERROR: no redirect for file id ${FILE_ID}" >&2; exit 1; }
case "$LOC" in
    https://s3-*.amazonaws.com/*) ;;
    *) echo "ERROR: unexpected redirect target: $LOC" >&2; exit 1 ;;
esac

echo "downloading payload directly from S3 (no proxy for the payload)"
curl -s --max-time 1800 -o "$ARCHIVE.part" \
     -w "code=%{http_code} size=%{size_download} time=%{time_total}s speed=%{speed_download}\n" \
     "$LOC"
mv "$ARCHIVE.part" "$ARCHIVE"
verify "$ARCHIVE"
echo "saved $ARCHIVE"
