#!/usr/bin/env python3
"""compression_curve.py - the M3 measurement: quality against size.

Every row is a real configuration the program can load, measured on the
documented question set. Nothing here is estimated.

The question M3 has to answer is "what do I lose if I only have N bytes".
Truncating layers and truncating feed-forward width were the obvious
answers; both turned out to be cliffs. Four-bit weights are the only
compression that degrades gradually, and this measures how far it goes.

Usage: compression_curve.py <model> <expected.txt>
"""
import os
import subprocess
import sys

BIN = "skynet.exe" if os.name == "nt" else "./skynet"

# (label, extra args)
CONFIGS = [
    ("8-bit full",        []),
    ("8-bit hid=224",     ["--hid", "224"]),
    ("8-bit hid=192",     ["--hid", "192"]),
    ("8-bit 3 layers",    ["--layers", "3"]),
    ("4-bit full",        ["--bits", "4"]),
    ("4-bit hid=224",     ["--bits", "4", "--hid", "224"]),
    ("4-bit hid=192",     ["--bits", "4", "--hid", "192"]),
    ("4-bit 3 layers",    ["--bits", "4", "--layers", "3"]),
]


def normalize(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def load_expected(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        q, want = line.split("|", 1)
        rows.append((q.strip(), normalize(want.strip())))
    return rows


def ask(model, question, extra):
    cmd = [BIN, "ask", model, question] + extra
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "<timeout>"
    for line in out.stdout.splitlines():
        if line.startswith("answer "):
            return line[7:]
    return "<no answer>"


def used_bytes(model, extra):
    cmd = [BIN, "model", model] + extra
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return 0
    for line in out.stdout.splitlines():
        if "weights" in line and "used" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "bytes,":
                    try:
                        return int(parts[i + 1])
                    except (ValueError, IndexError):
                        pass
    return 0


def main():
    if len(sys.argv) < 3:
        print("usage: compression_curve.py <model> <expected.txt>")
        return 2
    model, expected = sys.argv[1], sys.argv[2]
    rows = load_expected(expected)

    print("%-16s %-8s %-9s %s" % ("config", "KB", "accuracy", "sample"))
    print("-" * 78)

    for label, extra in CONFIGS:
        good = 0
        sample = ""
        for q, want in rows:
            got = ask(model, q, extra)
            if want in normalize(got):
                good += 1
            if not sample and got.strip():
                sample = got[:30]
        kb = used_bytes(model, extra) / 1024.0
        pct = 100.0 * good / len(rows) if rows else 0.0
        print("%-16s %-8.0f %-9.1f%% %s" % (label, kb, pct, sample))

    return 0


if __name__ == "__main__":
    sys.exit(main())
