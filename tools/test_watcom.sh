#!/bin/sh
# test_watcom.sh - verify the exact steps the dos-build CI job runs.
#
# The job is continue-on-error, so a broken URL or a wrong directory layout
# would leave a permanently yellow job that nobody reads.  This script is
# how that was checked by hand before the job was trusted.
set -e

WORK=${WORK:-/tmp/owtest}
URL="https://github.com/open-watcom/open-watcom-v2/releases/download/Current-build/ow-snapshot.tar.xz"

echo "== download"
if [ ! -f "$WORK/ow.tar.xz" ]; then
    mkdir -p "$WORK"
    curl -fL --retry 3 -o "$WORK/ow.tar.xz" "$URL"
fi
ls -la "$WORK/ow.tar.xz"

echo "== verify archive integrity before extracting"
xz -t "$WORK/ow.tar.xz" && echo "archive is valid"

echo "== extract"
rm -rf "$WORK/opt"
mkdir -p "$WORK/opt"
tar xf "$WORK/ow.tar.xz" -C "$WORK/opt"
echo "top level:"
ls "$WORK/opt"

echo "== locate wcc"
WCC=$(find "$WORK/opt" -maxdepth 4 -name wcc -type f | head -1)
echo "wcc: $WCC"
if [ -z "$WCC" ]; then
    echo "FAIL: wcc not found in the archive"
    exit 1
fi

echo "== derive WATCOM root"
# wcc lives in <root>/binl64 or <root>/binl
WATCOM=$(dirname "$(dirname "$WCC")")
echo "WATCOM=$WATCOM"
ls "$WATCOM" | head -10

echo "== compile the core for 16-bit DOS"
cd "$(dirname "$0")/.."
export WATCOM
export PATH="$WATCOM/binl64:$WATCOM/binl:$PATH"
export INCLUDE="$WATCOM/h"
rm -rf build_dos && mkdir -p build_dos
wcc -ml -zq -Isrc -fo=build_dos/ src/main.c src/sk_math.c \
    src/sk_arena.c src/sk_clock.c src/sk_vault.c src/sk_sched.c \
    src/sk_io.c src/sk_selftest.c

echo "== link"
wlink system dos name build_dos/skynet.exe file build_dos/*.obj
ls -la build_dos/skynet.exe

echo "== DOS BUILD OK"
