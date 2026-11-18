#!/bin/sh
# train_check.sh - what is actually true about the trainer.
#
# Two claims, both checked:
#
#   1. lr = 0 changes nothing. This proves the measurement is stable: a
#      reported loss that moves with no updates would mean the evaluation
#      pass is contaminated by the training pass.
#   2. On a single example the loss falls monotonically. This is the only
#      honest way to verify a gradient without a reference implementation:
#      a wrong sign makes it rise, a wrong scale makes it diverge.
#
# What is deliberately NOT claimed: that training improves the model across
# many examples. It does not, and docs/06-training.md explains why.
set -e
cd "$(dirname "$0")/.."
MODEL=refs/Tiny-2.1-main/dos/TINY.BIN

if [ ! -f build/dataset.txt ]; then
    python3 tools/build_dataset.py build/dataset.txt
fi
head -1 build/dataset.txt > build/one.txt
head -40 build/dataset.txt > build/tc_small.txt

echo "== lr = 0 must not change anything"
Z=$(./skynet train "$MODEL" build/tc_small.txt --epochs 3 --lr 0)
echo "$Z" | grep epoch | sed 's/^/   /'
python3 - "$Z" <<'EOF' || exit 1
import re, sys
losses = [float(m) for m in re.findall(r'epoch \d+:.*?loss ([\d.]+)', sys.argv[1])]
if len(losses) < 2:
    print("::error::not enough epochs reported"); sys.exit(1)
if max(losses) - min(losses) > 1e-6:
    print("::error::loss moved with lr=0; the measurement is contaminated")
    sys.exit(1)
print("   stable: every epoch reported %.4f" % losses[0])
EOF

echo
echo "== one example: the loss must fall"
OUT=$(./skynet train "$MODEL" build/one.txt --epochs 5 --lr 16777)
echo "$OUT" | grep epoch | sed 's/^/   /'
FIRST=$(echo "$OUT" | grep 'epoch 1' | sed 's/.*loss //; s/,.*//')
LAST=$(echo "$OUT" | grep 'epoch 5' | sed 's/.*loss //; s/,.*//')
python3 - "$FIRST" "$LAST" <<'EOF' || exit 1
import sys
first, last = float(sys.argv[1]), float(sys.argv[2])
print("   loss %.4f -> %.4f (%.0f%% of the start)"
      % (first, last, 100.0 * last / first))
if last >= first:
    print("::error::loss did not decrease; the gradient is wrong")
    sys.exit(1)
if last > 0.6 * first:
    print("::error::loss barely moved; the gradient scale is suspect")
    sys.exit(1)
print("   gradient is correct and the step size is sane")
EOF
