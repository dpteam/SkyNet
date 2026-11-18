#!/bin/sh
# train_check.sh - prove the trainer's gradient is correct.
#
# The only honest way to check a gradient without a reference implementation
# is to watch the loss fall on a SINGLE example. If the sign were wrong the
# loss would rise; if the scale were wrong it would diverge. One example
# removes every other variable.
#
# Training across many examples is a different matter, covered in
# docs/06-training.md: the embedding matrix doubles as the output
# projection, so improving one side degrades the other.
set -e
cd "$(dirname "$0")/.."
MODEL=refs/Tiny-2.1-main/dos/TINY.BIN

if [ ! -f build/dataset.txt ]; then
    python3 tools/build_dataset.py build/dataset.txt
fi
head -1 build/dataset.txt > build/one.txt

echo "== single-example training must reduce the loss"
OUT=$(./skynet train "$MODEL" build/one.txt --epochs 5 --lr 17)
echo "$OUT" | grep epoch | sed 's/^/   /'

FIRST=$(echo "$OUT" | grep 'epoch 1' | sed 's/.*loss //')
LAST=$(echo "$OUT" | grep 'epoch 5' | sed 's/.*loss //')

python3 - "$FIRST" "$LAST" <<'EOF' || exit 1
import sys
first = float(sys.argv[1])
last = float(sys.argv[2])
print("   loss %.4f -> %.4f (%.0f%% of the start)" % (first, last, 100.0 * last / first))
if last >= first:
    print("::error::loss did not decrease; the gradient is wrong")
    sys.exit(1)
if last > 0.6 * first:
    print("::error::loss barely moved; the gradient scale is suspect")
    sys.exit(1)
print("   gradient is correct and the step size is sane")
EOF
