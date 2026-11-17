#!/bin/sh
# wsl_ci_check.sh - reproduce the CI steps locally inside WSL.
#
# This exists because the first CI run found two real bugs (fileno/ftruncate
# hidden by __STRICT_ANSI__, and an unused variable under -Werror) that the
# Windows build could never have caught.  Keeping the CI steps runnable
# locally means the next one is found before it is pushed.
set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)
echo "== root: $ROOT"

CFLAGS="-std=c89 -pedantic -Wall -Wextra -Werror -O2"

echo
echo "== 1. hygiene"
python3 tools/hygiene.py

echo
echo "== 2. build (strict, -Werror)"
rm -f src/*.o skynet
make all CFLAGS="$CFLAGS"

echo
echo "== 3. selftest"
./skynet selftest

echo
echo "== 4. crash test"
sh tests/crash_test.sh 6

echo
echo "== 5. durable mode"
rm -rf build/dur && mkdir -p build/dur
./skynet run --quiet --durable --max 5000 --dir build/dur/state --out build/dur/p.txt >/dev/null
./build/verify build/dur/p.txt

echo
echo "== 6. recovery after a damaged slot"
rm -rf build/rec && mkdir -p build/rec
./skynet run --quiet --max 3000 --dir build/rec/state --out build/rec/p.txt >/dev/null
./skynet run --quiet --max 6000 --dir build/rec/state --out build/rec/p.txt >/dev/null
f=$(ls build/rec/state/STATE.* | head -1)
printf '\336\255\276\357' | dd of="$f" bs=1 seek=30 conv=notrunc status=none
echo "damaged $f"
./skynet recover --dir build/rec/state | tail -2

echo
echo "== 7. logo reproducibility"
python3 tools/make_logo.py >/dev/null
if ! git diff --quiet -- assets/; then
    echo "FAIL: assets/ changed after regeneration"
    git diff --stat -- assets/
    exit 1
fi
echo "logo is byte-identical after regeneration"

echo
echo "ALL CHECKS PASSED"
