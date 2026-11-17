#!/bin/sh
# fetch_watcom.sh - obtain a verified OpenWatcom snapshot tarball.
#
# Hard-won details, each one cost a failed CI run:
#
#  1. The release host drops this ~148 MB transfer near the end.  curl
#     reports success often enough on a truncated transfer that its exit
#     code is worthless; the only trustworthy check is xz -t.
#  2. Every attempt must write to its OWN file.  Resuming into a shared
#     path while another attempt is running produces a file that is longer
#     than the real asset and corrupt in the middle - which is exactly what
#     happened when this was first parallelised.
#  3. Once a verified file exists it is moved into place, so a later run
#     finds it already valid and skips the download entirely.
#
# Usage: sh tools/fetch_watcom.sh [workdir]
set -e

WORK=${1:-${WORK:-/tmp/owtest}}
URL="https://github.com/open-watcom/open-watcom-v2/releases/download/Current-build/ow-snapshot.tar.xz"
FINAL="$WORK/ow.tar.xz"

mkdir -p "$WORK"

if xz -t "$FINAL" 2>/dev/null; then
    echo "cached archive is valid: $(wc -c < "$FINAL") bytes"
    exit 0
fi
rm -f "$FINAL"

i=1
while [ "$i" -le 6 ]; do
    TMP="$WORK/ow.$i.part"
    rm -f "$TMP"

    # Three passes per attempt: start, resume once, resume twice.  The
    # host tends to drop at 30-99%, and each resume continues from where
    # the previous one stopped.
    p=1
    while [ "$p" -le 3 ]; do
        curl -fsL -C - --max-time 300 --retry 2 --retry-delay 1 \
             -o "$TMP" "$URL" 2>/dev/null || true
        if xz -t "$TMP" 2>/dev/null; then
            break
        fi
        p=$((p + 1))
    done

    if xz -t "$TMP" 2>/dev/null; then
        SIZE=$(wc -c < "$TMP")
        echo "VALID on attempt $i: $SIZE bytes"
        mv "$TMP" "$FINAL"
        exit 0
    fi

    echo "attempt $i failed: $(wc -c < "$TMP" 2>/dev/null || echo 0) bytes, discarded"
    rm -f "$TMP"
    i=$((i + 1))
done

echo "FAIL: could not obtain a verified archive"
echo "      The DOS build is best-effort by design; see docs/02-ci.md."
exit 1
