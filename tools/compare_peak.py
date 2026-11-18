#!/usr/bin/env python3
"""compare_peak.py - check the fixed-point matvec against the float reference.

Two implementations of the same wrong idea agree with each other, so the
Q16.16 result is compared against Python floats, never against itself.

The tolerance is not arbitrary: a row scale is stored as one Q16.16 value,
so it carries at most 1/65536 of absolute error, which is a relative error
of 1/(65536*scale). For the released model's scales (around 0.004) that is
about 0.4%, so 1% is the honest budget. Anything larger means a real bug.

Usage: compare_peak.py <reference-q16> <our-value-as-printed>
"""
import sys


def main():
    if len(sys.argv) < 3:
        print("usage: compare_peak.py <ref_q16> <ours>")
        return 2

    ref_q16 = int(sys.argv[1])
    ours = float(sys.argv[2])
    ref = ref_q16 / 65536.0

    if ref == 0.0:
        print("reference peak is zero; nothing to compare")
        return 1

    diff = abs(ref - ours) / ref
    print("reference peak: %.6f (Q16.16 %d)" % (ref, ref_q16))
    print("our peak      : %.6f" % ours)
    print("relative difference: %.4f%%" % (diff * 100.0))

    if diff > 0.01:
        print("::error::matvec differs from the float reference by more than 1%")
        return 1
    print("within the one-LSB-per-row budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
