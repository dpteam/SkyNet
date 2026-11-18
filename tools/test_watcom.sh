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
# Verify before extracting: a truncated archive fails inside tar with a
# message that looks like a code problem, which wasted a debugging cycle.
if ! xz -t "$WORK/ow.tar.xz" 2>/dev/null; then
    echo "::error::archive at $WORK/ow.tar.xz is not intact"
    exit 1
fi
tar xf "$WORK/ow.tar.xz" -C "$WORK/opt" || {
    echo "::error::extraction failed"
    exit 1
}

echo "== locate the LINUX host compiler"
WATCOM="$WORK/opt"

# The snapshot ships host binaries for many platforms (binl64 for Linux x64,
# bino64 for OS/2, binnt for Windows).  Prefer the Linux ones, but fall back
# to any wcc in the tree: the layout has changed between snapshots, and a
# hard-coded path turns a cosmetic change upstream into a red job.
WCC=""
for cand in "$WATCOM/binl64/wcc" "$WATCOM/binl/wcc"; do
    if [ -x "$cand" ]; then WCC="$cand"; break; fi
done
if [ -z "$WCC" ]; then
    WCC=$(find "$WATCOM" -maxdepth 3 -type f -name wcc 2>/dev/null | head -1)
fi
if [ -z "$WCC" ] || [ ! -x "$WCC" ]; then
    echo "::error::no host compiler found in the OpenWatcom snapshot"
    echo "archive: $(wc -c < "$WORK/ow.tar.xz" 2>/dev/null || echo 0) bytes"
    echo "top level of the extracted tree:"
    ls "$WATCOM" 2>/dev/null | head -30
    echo "any wcc in the tree:"
    find "$WATCOM" -name 'wcc*' 2>/dev/null | head -10
    exit 1
fi
WATCOM=$(dirname "$(dirname "$WCC")")
echo "wcc: $WCC"
echo "WATCOM: $WATCOM"

echo
echo "== compile the core for 16-bit DOS, one file at a time"
export WATCOM
export PATH="$WATCOM/binl64:$WATCOM/binl:$PATH"
export INCLUDE="$WATCOM/h"
rm -rf build_dos && mkdir -p build_dos

fail=0
for f in main sk_math sk_arena sk_clock sk_vault sk_sched sk_io sk_mem sk_tensor sk_forward sk_selftest; do
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
#
# stack=8192 is not cosmetic.  The default DOS stack is 2 KB, and the vault
# layer puts a sk_vault (about 900 bytes: two 192-byte slot paths, a 128-byte
# directory, a journal path and a scratch path) on the stack in every caller.
# With selftest frames on top of that, 2 KB overflows and the program dies
# with no output at all - which is exactly how the first DOS run failed.
LINK_ARGS="system dos name build_dos/skynet.exe option stack=8192"
for o in main sk_math sk_arena sk_clock sk_vault sk_sched sk_io sk_mem sk_tensor sk_forward sk_selftest; do
    LINK_ARGS="$LINK_ARGS file build_dos/$o.o"
done
# shellcheck disable=SC2086
wlink $LINK_ARGS

ls -la build_dos/skynet.exe
echo
echo "== DOS BUILD OK"
