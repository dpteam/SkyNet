#!/bin/sh
# verify_model.sh - the model checks that CI runs, runnable by hand.
#
# Written because every attempt to inline this in a shell command through
# PowerShell mangled the quoting, and the CI YAML heredoc broke the YAML.
# A file has neither problem.
set -e

cd "$(dirname "$0")/.."
MODEL=${SKYNET_MODEL:-}
if [ -z "$MODEL" ]; then
    if [ -f refs/Tiny-2.1-main/dos/TINY.BIN ]; then
        MODEL=refs/Tiny-2.1-main/dos/TINY.BIN
    else
        MODEL=tests/fixtures/TINY.BIN
    fi
fi
if [ ! -f "$MODEL" ]; then
    echo "missing baseline: $MODEL" >&2
    echo "tests/fixtures/TINY.BIN should be in the clone; refs/ is not" >&2
    exit 2
fi

echo "== build"
make all CFLAGS="-std=c89 -pedantic -Wall -Wextra -Werror -O2" >/dev/null
cc -O2 -o build/ref_tiny tests/ref_tiny.c -lm
echo "ok"

echo
echo "== every single letter matches the reference"
fail=0
for c in a b c d e f g h i j k l m n o p q r s t u v w x y z; do
    ./skynet predict "$MODEL" "$c" > /tmp/fp.txt
    ./build/ref_tiny "$MODEL" "$c" > /tmp/ref.txt
    if ! python3 tools/compare_scores.py /tmp/ref.txt /tmp/fp.txt > /tmp/cmp.txt; then
        echo "  FAIL letter $c"
        sed 's/^/    /' /tmp/cmp.txt
        fail=1
    fi
done
if [ "$fail" = 0 ]; then
    echo "  all 26 letters match"
else
    exit 1
fi

echo
echo "== answers on the documented questions"
python3 tools/ask_accuracy.py "$MODEL" tests/expected.txt > /tmp/acc.txt
tail -4 /tmp/acc.txt
RATE=$(grep -oE '[0-9]+\.[0-9]%' /tmp/acc.txt | head -1)
python3 tools/check_rate.py "$RATE" 90 "accuracy"

echo
echo "== decoding matches the reference on a question set"
python3 tools/ask_compare.py "$MODEL" tests/questions.txt > /tmp/dec.txt
tail -6 /tmp/dec.txt
RATE=$(grep -oE '[0-9]+\.[0-9]%' /tmp/dec.txt | head -1)
python3 tools/check_rate.py "$RATE" 90 "exact agreement"

echo
echo "MODEL CHECKS PASSED"
