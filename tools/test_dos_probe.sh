#!/bin/sh
# test_dos_probe.sh - build and inspect the DOS capability probe.
#
# The full program died silently on DOS. This probe answers, one line at a
# time, which capability is missing, so the fix is based on evidence rather
# than on a guess about stack size or file handles.
set -e

WORK=${WORK:-/tmp/ow}
cd "$(dirname "$0")/.."

sh tools/fetch_watcom.sh "$WORK"

rm -rf "$WORK/opt"
mkdir -p "$WORK/opt"
xz -t "$WORK/ow.tar.xz" >/dev/null 2>&1 || { echo "archive corrupt"; exit 1; }
tar xf "$WORK/ow.tar.xz" -C "$WORK/opt"

WATCOM="$WORK/opt"
WCC="$WATCOM/binl64/wcc"
[ -x "$WCC" ] || WCC=$(find "$WATCOM" -maxdepth 3 -type f -name wcc | head -1)
WATCOM=$(dirname "$(dirname "$WCC")")
export WATCOM
export PATH="$WATCOM/binl64:$WATCOM/binl:$PATH"
export INCLUDE="$WATCOM/h"

echo "wcc: $(command -v wcc)"
echo "WATCOM=$WATCOM"

mkdir -p build_dos
echo "== compile"
wcc -ml -zq -Itests -fo=build_dos/ tests/probe_dos.c
echo "== objects"
ls -la build_dos/*.o
echo "== link"
wlink system dos name build_dos/probe.exe option stack=8192 file build_dos/probe_dos.o
ls -la build_dos/probe.exe
echo "== probe built"
