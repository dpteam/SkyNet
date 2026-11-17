#!/bin/sh
# crash_test.sh - kill SkyNet mid-run, restart, verify the output is still
# a contiguous prime sequence with no gaps and no duplicates.
#
# POSIX sh so it runs on any CI runner without bash-specific features.
#
# Two things make this test honest:
#   - the killed rounds are UNBOUNDED (--max 0), so the process can never
#     finish on its own and every kill really lands in the middle of work;
#   - the final round is bounded to "a little more than we already have",
#     so it must resume from the vault rather than start over.
#
# Usage: sh tests/crash_test.sh [rounds]

set -e

ROUNDS=${1:-15}
EXTRA=${EXTRA:-20000}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXE="$ROOT/skynet"
VER="$ROOT/build/verify"
WORK="$ROOT/build/crashtest"

if [ ! -x "$EXE" ] || [ ! -x "$VER" ]; then
    echo "build first: make (or see build.ps1 / .github/workflows/ci.yml)" >&2
    exit 2
fi

rm -rf "$WORK"
mkdir -p "$WORK"

STATE="$WORK/state"
OUT="$WORK/primes.txt"

echo "crash test: $ROUNDS unbounded kills, then resume"

i=1
while [ "$i" -le "$ROUNDS" ]; do
    "$EXE" run --quiet --max 0 --dir "$STATE" --out "$OUT" &
    pid=$!
    # a different delay each round, deterministic but varied
    ms=$(( (i * 37 + 41) % 360 + 40 ))
    sleep "$(echo "$ms" | awk '{printf "%.3f", $1/1000}')"
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    if [ -f "$OUT" ]; then
        n=$(wc -l < "$OUT" | tr -d ' ')
    else
        n=0
    fi
    echo "  round $i: killed after ${ms}ms, $n primes durable"
    i=$((i + 1))
done

if [ -f "$OUT" ]; then
    before=$(wc -l < "$OUT" | tr -d ' ')
else
    before=0
fi
target=$((before + EXTRA))
echo "resuming from $before, running to $target"

"$EXE" run --quiet --max "$target" --dir "$STATE" --out "$OUT" >/dev/null

"$VER" "$OUT"
