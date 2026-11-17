#!/bin/sh
# dos_test.sh - acceptance test on the real target: 16-bit DOS in DOSBox.
#
# POSIX twin of tests/dos_test.ps1 so CI can run it.  Every check here is
# something the host builds cannot prove:
#   - the program loads in real mode (69 KB in 640 KB conventional memory);
#   - a 16-bit size_t does not silently truncate an 8 MB arena request;
#   - the vault works without fsync;
#   - recovery resumes from state written by a PREVIOUS process on a real
#     DOS filesystem, not from a file this script just created.
#
# Requires: build_dos/skynet.exe (tools/test_watcom.sh), build/verify,
# and dosbox on PATH.
set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)
WORK="$ROOT/build/doswork"
EXE_SRC="$ROOT/build_dos/skynet.exe"
VER="$ROOT/build/verify"

[ -f "$EXE_SRC" ] || { echo "missing $EXE_SRC - run tools/test_watcom.sh"; exit 2; }
[ -x "$VER" ] || { echo "missing $VER - run make all"; exit 2; }
command -v dosbox >/dev/null || { echo "dosbox not installed"; exit 2; }

fails=0
check() {
    if [ "$2" = "1" ]; then
        echo "  PASS  $1"
    else
        echo "  FAIL  $1"
        fails=$((fails + 1))
    fi
}

matches() { case "$1" in *"$2"*) echo 1 ;; *) echo 0 ;; esac; }

reset_work() {
    rm -rf "$WORK"
    mkdir -p "$WORK"
    cp "$EXE_SRC" "$WORK/SKYNET.EXE"
}

echo "=== DOS acceptance test ==="

echo "[1] selftest"
reset_work
OUT=$(sh tools/dos_run.sh "$WORK" "skynet selftest" SELF.TXT || true)
# Deliberately not a hardcoded count: it changes every time a check is
# added, and a test that must be edited after each improvement is a test
# that will eventually be edited to pass.
check "selftest reports zero failures" "$(matches "$OUT" "passed, 0 failed")"
check "selftest reached the end"       "$(matches "$OUT" "vault 9: close")"

echo "[2] arena degradation (8 MB requested on a 640 KB machine)"
OUT=$(sh tools/dos_run.sh "$WORK" "skynet run --quiet --max 100 --dir S --out P.TXT" RUN1.TXT || true)
check "arena degraded instead of failing" "$(matches "$OUT" "of 8388608 bytes granted")"
check "the run still completed"           "$(matches "$OUT" "stopped at prime #100")"

echo "[3] checkpoints on every super-prime"
OUT=$(sh tools/dos_run.sh "$WORK" "skynet run --quiet --max 20000 --ckpt 1 --dir S --out P.TXT" RUN2.TXT || true)
check "checkpoints were written" "$(matches "$OUT" "checkpoints")"

echo "[4] recovery from a previous process"
OUT=$(sh tools/dos_run.sh "$WORK" "skynet run --quiet --max 25000 --dir S --out P.TXT" RUN3.TXT || true)
check "second process resumed"        "$(matches "$OUT" "resumed at prime #")"
case "$OUT" in
    *"fresh start"*) check "second process did not restart" 0 ;;
    *)               check "second process did not restart" 1 ;;
esac

echo "[5] verify the primes file the DOS program wrote"
if [ -f "$WORK/P.TXT" ]; then
    V=$("$VER" "$WORK/P.TXT" || true)
    check "primes file is an exact prefix" "$(matches "$V" "PASS")"
    echo "$V" | grep -E "lines|last value|mismatches" | sed 's/^/        /'
else
    check "primes file exists" 0
fi

echo "[6] EMS memory report"
OUT=$(sh tools/dos_run.sh "$WORK" "skynet mem" MEM.TXT || true)
check "EMS driver detected" "$(matches "$OUT" "ems   : v")"
check "EMS reports free pages" "$(matches "$OUT" "pages")"
check "the 64 KB ceiling is lifted" "$(matches "$OUT" "no longer applies")"
echo "$OUT" | grep -E "ems|usable" | sed 's/^/        /'

echo "[7] paged store round-trip across page boundaries"
# The selftest writes 3 pages of a pattern derived from the offset, then
# reads them back in 173-byte chunks so every page boundary is crossed at
# two different alignments. A forgotten map() shows up as a byte mismatch.
OUT=$(sh tools/dos_run.sh "$WORK" "skynet selftest" SELF2.TXT || true)
check "pool opened in EMS"          "$(matches "$OUT" "pool: 48 KB in 3 EMS pages")"
check "pool round-trip passed"      "$(matches "$OUT" "passed, 0 failed")"
echo "$OUT" | grep -E "pool:|maps" | sed 's/^/        /'

echo
if [ "$fails" -eq 0 ]; then
    echo "=== DOS ACCEPTANCE PASSED ==="
    exit 0
fi
echo "=== $fails CHECK(S) FAILED ==="
exit 1
