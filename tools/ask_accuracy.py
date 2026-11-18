#!/usr/bin/env python3
"""ask_accuracy.py - does the port actually answer correctly?

Comparing against the reference implementation proves the port is faithful.
This proves something different and more useful: that the answers are right.
The expected substrings come from the "Things to try" list in the released
TINY 2.1 README, which states every one of those questions is answered
correctly by the shipped TINY.BIN.

A port can be perfectly faithful and still be useless if it loaded the
weights wrong; this is the check that catches that.

Usage: ask_accuracy.py <model> <expected.txt>
"""
import subprocess
import sys


def ask(model, question):
    try:
        out = subprocess.run(["skynet.exe", "ask", model, question],
                             capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "<timeout>"
    for line in out.stdout.splitlines():
        if line.startswith("answer "):
            return line[7:]
    return "<no answer>"


def normalize(s):
    """Lower case and drop separators, so 'E-L-E-P-H-A-N-T.' matches
    'elephant'. The model spells with hyphens; that is correct output and a
    substring check on the raw string wrongly calls it a miss."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def main():
    if len(sys.argv) < 3:
        print("usage: ask_accuracy.py <model> <expected.txt>")
        return 2
    model, path = sys.argv[1], sys.argv[2]

    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        q, want = line.split("|", 1)
        rows.append((q.strip(), normalize(want.strip())))

    good = 0
    bad = []
    for q, want in rows:
        got = ask(model, q)
        ok = want in normalize(got)
        if ok:
            good += 1
        else:
            bad.append((q, want, got))
        print("%-4s %-45s %s" % ("ok" if ok else "MISS", q, got))

    n = len(rows)
    print()
    print("%d/%d correct (%.1f%%)" % (good, n, 100.0 * good / n if n else 0.0))
    if bad:
        print()
        print("misses:")
        for q, want, got in bad:
            print("  %-45s wanted %-12s got %s" % (q, want, got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
