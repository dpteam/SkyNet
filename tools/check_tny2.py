#!/usr/bin/env python3
"""check_tny2.py - independent parse of a TNY2 file.

Written because the C loader and the C verifier disagreed, and two
implementations of the same wrong idea still agree with each other. This
one is written from the format description alone:

  "TNY2", D, L, H, HID as 16-bit words, then for each table: rows of
  (float32 scale, cols signed bytes).  Order: emb, pos, then per layer
  g1 gains (D float32), wq wk wv wo, g2 gains, f1, f2, then gf gains.

If this script and the C loader produce the same CRC, the format reading
is right and the verifier is wrong.
"""
import struct
import os, sys
import zlib

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

def crc32c(data, crc=0):
    """CRC-32C (Castagnoli), the one the C code uses.

    zlib.crc32 is the older CRC-32 with polynomial 0xEDB88320, so its
    output is not comparable. Using it here was the first mistake in this
    script: two different checksums were compared and reported as a
    mismatch.
    """
    poly = 0x82F63B78
    table = crc32c.table
    if table is None:
        table = []
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ poly if c & 1 else c >> 1
            table.append(c)
        crc32c.table = table
    c = crc ^ 0xFFFFFFFF
    for b in data:
        c = table[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


crc32c.table = None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else baseline()
    f = open(path, "rb")

    magic = f.read(4)
    assert magic == b"TNY2", magic
    D, L, H, HID = struct.unpack("<4H", f.read(8))
    HD = D // H
    print("D=%d L=%d H=%d HID=%d HD=%d" % (D, L, H, HID, HD))

    weights = bytearray()
    order = []

    def table(name, rows, cols):
        start = len(weights)
        for r in range(rows):
            scale = f.read(4)
            assert len(scale) == 4, "short scale in %s row %d" % (name, r)
            w = f.read(cols)
            assert len(w) == cols, "short row in %s row %d" % (name, r)
            weights.extend(w)
        order.append((name, rows, cols, start, len(weights) - start))

    def gains(name, n):
        d = f.read(4 * n)
        assert len(d) == 4 * n, "short gains %s" % name

    table("emb", VOCAB, D)
    table("pos", CTX, D)
    for l in range(L):
        gains("g1[%d]" % l, D)
        table("wq[%d]" % l, D, D)
        table("wk[%d]" % l, D, D)
        table("wv[%d]" % l, D, D)
        table("wo[%d]" % l, D, D)
        gains("g2[%d]" % l, D)
        table("f1[%d]" % l, HID, D)
        table("f2[%d]" % l, D, HID)
    gains("gf", D)

    rest = f.read()
    f.close()

    print("weight bytes: %d" % len(weights))
    print("trailing bytes: %d" % len(rest))
    print("crc32c of weights: %08X" % crc32c(bytes(weights)))
    print("crc32 (zlib)     : %08X" % zlib.crc32(bytes(weights)))

    print()
    print("%-10s %5s %5s %9s %9s" % ("table", "rows", "cols", "offset", "bytes"))
    for name, rows, cols, off, n in order:
        print("%-10s %5d %5d %9d %9d" % (name, rows, cols, off, n))


if __name__ == "__main__":
    main()
