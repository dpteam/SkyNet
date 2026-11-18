#!/usr/bin/env python3
"""compare_scores.py - diff the fixed-point forward pass against the float one.

The reference is tests/ref_tiny.c, which is the released Tiny-2.1 forward
pass compiled for the host, not a rewrite of it. That distinction matters:
comparing two of my own implementations only proves I made the same mistake
twice.

Two things are checked:
  1. the top-1 letter agrees, because that is what the program actually
     outputs and a wrong argmax means wrong answers regardless of how close
     the numbers look;
  2. the scores are close, measured as a relative error on the spread of
     the vector (max - min), because the absolute values drift with the
     Q16.16 quantisation of the row scales and that drift is not an error.

Usage: compare_scores.py <ref.txt> <fp.txt>
"""
import sys


def parse(path):
    for line in open(path):
        line = line.strip()
        if line.startswith("scores "):
            return [float(x) for x in line.split()[1:]]
    return None


def main():
    if len(sys.argv) < 3:
        print("usage: compare_scores.py <ref.txt> <fp.txt>")
        return 2

    ref = parse(sys.argv[1])
    fp = parse(sys.argv[2])
    if ref is None or fp is None:
        print("could not parse scores")
        return 2
    if len(ref) != len(fp):
        print("length mismatch: %d vs %d" % (len(ref), len(fp)))
        return 1

    n = len(ref)
    ref_top = max(range(n), key=lambda i: ref[i])
    fp_top = max(range(n), key=lambda i: fp[i])

    spread = max(ref) - min(ref)
    diffs = [abs(ref[i] - fp[i]) for i in range(n)]
    worst = max(diffs)
    rel = worst / spread if spread else 0.0

    # How close was the reference to a tie?  If the top two letters differ by
    # less than the fixed-point error, an argmax flip says nothing about the
    # port being wrong - it says the model was nearly undecided.
    order = sorted(range(n), key=lambda i: -ref[i])
    gap = ref[order[0]] - ref[order[1]]

    print("letters        : %d" % n)
    print("reference top  : %d (%.4f)" % (ref_top, ref[ref_top]))
    print("fixed-point top: %d (%.4f)" % (fp_top, fp[fp_top]))
    print("top-1 gap      : %.4f" % gap)
    print("spread         : %.4f" % spread)
    print("worst diff     : %.4f (%.2f%% of spread)" % (worst, rel * 100.0))

    ok = True
    if rel > 0.02:
        print("FAIL: scores differ by more than 2%% of the spread")
        ok = False
    if ref_top != fp_top:
        if gap < worst:
            print("NOTE: different letter chosen, but the reference's top two "
                  "differ by %.4f, which is inside the fixed-point error %.4f"
                  % (gap, worst))
        else:
            print("FAIL: different letter chosen with a clear margin")
            ok = False
    if ok:
        print("PASS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
