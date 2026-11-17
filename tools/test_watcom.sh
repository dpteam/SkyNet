#!/bin/sh
# test_watcom.sh - verify the exact steps the dos-build CI job runs.
#
# The job is continue-on-error, so a broken URL or a wrong directory layout
# would leave a permanently yellow job that nobody reads.  This script is
# how that was checked by hand before the job was trusted.
#
# Three things about OpenWatcom that cost real debugging time:
#   1. wcc compiles ONE file per invocation.  Passing eight sources gives
#      "E1139: Command line contains more than one file to compile" seven
#      times and no objects.  Unlike gcc, it needs a loop.
#   2. The snapshot ships host binaries for many platforms (binl64 for
#      Linux x64, bino64 for OS/2, binnt for Windows).  'find ... -name wcc'
#      can return any of them, so binl64 is selected explicitly.
#   3. The download is flaky and must be verified with xz -t.  That logic
#      lives in tools/fetch_watcom.sh and is NOT duplicated here: the first
#      version of this script had its own copy, which happily overwrote a
#      verified 148 MB archive with a fresh truncated one.
set -e

WORK=${WORK:-/tmp/owtest}
cd "$(dirname "$0")/.."

echo "== obtain the toolchain"
sh tools/fetch_watcom.sh "$WORK"

echo
echo "== extract"
rm -rf "$WORK/opt"
mkdir -p "$WORK/opt"
tar xf "$WORK/ow.tar.xz" -C "$WORK/opt"

echo "== locate the LINUX host compiler"
WATCOM="$WORK/opt"
WCC="$WATCOM/binl64/wcc"
[ -x "$WCC" ] || WCC="$WATCOM/binl/wcc"
if [ ! -x "$WCC" ]; then
    echo "FAIL: no Linux host compiler in the snapshot"
    ls "$WATCOM" | head -20
    exit 1
fi
echo "wcc: $WCC"

echo
echo "== compile the core for 16-bit DOS, one file at a time"
export WATCOM
export PATH="$WATCOM/binl64:$WATCOM/binl:$PATH"
export INCLUDE="$WATCOM/h"
rm -rf build_dos && mkdir -p build_dos

fail=0
for f in main sk_math sk_arena sk_clock sk_vault sk_sched sk_io sk_selftest; do
    printf '  wcc %s.c ... ' "$f"
    if wcc -ml -zq -Isrc -fo=build_dos/ "src/$f.c" >"build_dos/$f.log" 2>&1; then
        echo ok
    else
        echo FAIL
        grep -E 'Error|Warning' "build_dos/$f.log" | head -8
        fail=$((fail + 1))
    fi
done

if [ "$fail" -ne 0 ]; then
    echo
    echo "== $fail file(s) failed to compile for 16-bit"
    exit 1
fi

echo
echo "== link"
# wlink does not take a bare list of object files: each needs the `file`
# directive, and the syntax is one directive per argument.
LINK_ARGS="system dos name build_dos/skynet.exe"
for o in main sk_math sk_arena sk_clock sk_vault sk_sched sk_io sk_selftest; do
    LINK_ARGS="$LINK_ARGS file build_dos/$o.o"
done
# shellcheck disable=SC2086
wlink $LINK_ARGS

ls -la build_dos/skynet.exe
echo
echo "== DOS BUILD OK"
