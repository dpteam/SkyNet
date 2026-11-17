#!/bin/sh
# test_watcom.sh - verify the exact steps the dos-build CI job runs.
#
# The job is continue-on-error, so a broken URL or a wrong directory layout
# would leave a permanently yellow job that nobody reads.  This script is
# how that was checked by hand before the job was trusted.
#
# Two things about OpenWatcom that cost real debugging time:
#   1. wcc compiles ONE file per invocation.  Passing eight sources gives
#      "E1139: Command line contains more than one file to compile" seven
#      times and no objects.  Unlike gcc, it needs a loop.
#   2. The snapshot ships host binaries for many platforms (binl64 for
#      Linux x64, bino64 for OS/2, binnt for Windows).  'find ... -name wcc'
#      can return any of them, so the Linux one must be selected explicitly.
set -e

WORK=${WORK:-/tmp/owtest}
URL="https://github.com/open-watcom/open-watcom-v2/releases/download/Current-build/ow-snapshot.tar.xz"
EXPECTED=155531784

echo "== download"
if [ ! -f "$WORK/ow.tar.xz" ] || [ "$(stat -c%s "$WORK/ow.tar.xz")" -ne "$EXPECTED" ]; then
    mkdir -p "$WORK"
    rm -f "$WORK/ow.tar.xz"
    for i in 1 2 3 4 5 6 7 8; do
        curl -fL -C - --retry 3 --retry-delay 2 -o "$WORK/ow.tar.xz" "$URL" || true
        if xz -t "$WORK/ow.tar.xz" 2>/dev/null; then
            echo "archive verified on attempt $i"
            break
        fi
        echo "attempt $i incomplete ($(stat -c%s "$WORK/ow.tar.xz" 2>/dev/null || echo 0) bytes)"
        sleep 2
    done
fi
xz -t "$WORK/ow.tar.xz" || { echo "FAIL: archive still corrupt"; exit 1; }
echo "archive is valid ($(stat -c%s "$WORK/ow.tar.xz") bytes)"

echo "== extract"
rm -rf "$WORK/opt"
mkdir -p "$WORK/opt"
tar xf "$WORK/ow.tar.xz" -C "$WORK/opt"

echo "== locate the LINUX host compiler"
WATCOM="$WORK/opt"
WCC="$WATCOM/binl64/wcc"
if [ ! -x "$WCC" ]; then
    WCC="$WATCOM/binl/wcc"
fi
if [ ! -x "$WCC" ]; then
    echo "FAIL: no Linux host compiler in the snapshot"
    ls "$WATCOM" | head -20
    exit 1
fi
echo "wcc: $WCC"

echo "== compile the core for 16-bit DOS, one file at a time"
cd "$(dirname "$0")/.."
export WATCOM
export PATH="$WATCOM/binl64:$WATCOM/binl:$PATH"
export INCLUDE="$WATCOM/h"
rm -rf build_dos && mkdir -p build_dos

for f in main sk_math sk_arena sk_clock sk_vault sk_sched sk_io sk_selftest; do
    echo "  wcc $f.c"
    wcc -ml -zq -Isrc -fo=build_dos/ "src/$f.c"
done

echo "== link"
ls -la build_dos/*.o
wlink system dos name build_dos/skynet.exe file build_dos/main.o \
    build_dos/sk_math.o build_dos/sk_arena.o build_dos/sk_clock.o \
    build_dos/sk_vault.o build_dos/sk_sched.o build_dos/sk_io.o \
    build_dos/sk_selftest.o

ls -la build_dos/skynet.exe
echo
echo "== DOS BUILD OK"
