#!/usr/bin/env python3
"""check_matvec.py - float reference for one weight table.

The C core has no floating point, so its matvec is Q16.16. This script
computes the same product in float and prints the peak, so the fixed-point
result can be compared against an independent implementation. If they
disagree, the bug is in the int8 path or in the scale conversion, not in
the file format.

Usage: check_matvec.py [model] [table] [row]
"""
import struct
import os, sys

VOCAB = 97
CTX = 128


def baseline():
    """The released weight file: refs/ when the full tree is present,
    otherwise the tracked fixture. Never a silent fallback to something else
    - if neither exists this returns a path that does not open, and the
    caller says so."""
    full = os.path.join('refs', 'Tiny-2.1-main', 'dos', 'TINY.BIN')
    if os.path.exists(full):
        return full
    return os.path.join('tests', 'fixtures', 'TINY.BIN')

def read_tables(path):
    f = open(path, "rb")
    assert f.read(4) == b"TNY2"
    D, L, H, HID = struct.unpack("<4H", f.read(8))

    tables = {}

    def table(name, rows, cols):
        scales = []
        weights = []
        for _ in range(rows):
            s = struct.unpack("<f", f.read(4))[0]
            w = struct.unpack("<%db" % cols, f.read(cols))
            scales.append(s)
            weights.append(w)
        tables[name] = (scales, weights)

    def gains(n):
        f.read(4 * n)

    table("emb", VOCAB, D)
    table("pos", CTX, D)
    for l in range(L):
        gains(D)
        table("wq%d" % l, D, D)
        table("wk%d" % l, D, D)
        table("wv%d" % l, D, D)
        table("wo%d" % l, D, D)
        gains(D)
        table("f1_%d" % l, HID, D)
        table("f2_%d" % l, D, HID)
    gains(D)
    f.close()
    return D, L, HID, tables


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else baseline()
    name = sys.argv[2] if len(sys.argv) > 2 else "wq0"
    D, L, HID, tables = read_tables(path)
    scales, weights = tables[name]

    print("table %s: %d rows x %d cols" % (name, len(weights), len(weights[0])))
    print("scale[0] = %r  scale[1] = %r" % (scales[0], scales[1]))
    print("first 8 weights of row 0: %s" % (weights[0][:8],))

    # y[i] = scale[i] * sum_j w[i][j] * x[j], with x = all ones
    y = [scales[i] * sum(weights[i]) for i in range(len(weights))]
    peak = max(abs(v) for v in y)
    nz = sum(1 for v in y if v != 0.0)
    print("x = all ones")
    print("  non-zero: %d/%d" % (nz, len(y)))
    print("  peak |y|: %.9f" % peak)
    print("  y[0]    : %.9f" % y[0])
    print("  y[1]    : %.9f" % y[1])
    print("  peak as Q16.16: %d" % int(round(peak * 65536)))

    # What the C code should print, allowing one LSB of Q16.16 per row.
    print()
    print("expected in C: non-zero %d, peak ~%d" % (nz, int(round(peak * 65536))))


if __name__ == "__main__":
    main()
