#!/usr/bin/env python3
"""width_curve.py - quality against feed-forward width.

Layer truncation is a cliff: three of four blocks give 5.7% and four give
100%, so it is not a usable degradation path. The feed-forward width is the
other shrinkable dimension and it is continuous, so this is the curve that
decides whether a progressive format is worth building and where to cut.

Reports bytes, accuracy, and a sample answer per width so the failure mode
is visible (gibberish versus wrong-but-structured versus right).

Usage: width_curve.py <model> <expected.txt> [widths...]
"""
import os
import subprocess
import sys

BIN = "skynet.exe" if os.name == "nt" else "./skynet"


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


def ask(model, question, hid):
    cmd = [BIN, "ask", model, question, "--hid", str(hid)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "<timeout>"
    for line in out.stdout.splitlines():
        if line.startswith("answer "):
            return line[7:]
    return "<no answer>"


def used_bytes(model, hid):
    cmd = [BIN, "model", model, "--hid", str(hid)]
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
        print("usage: width_curve.py <model> <expected.txt> [widths...]")
        return 2
    model, expected = sys.argv[1], sys.argv[2]
    widths = [int(x) for x in sys.argv[3:]] or [32, 64, 96, 128, 160, 192, 224, 256]

    rows = load_expected(expected)
    print("%-8s %-9s %-9s %s" % ("width", "KB", "accuracy", "sample answer"))

    for w in widths:
        good = 0
        sample = ""
        for q, want in rows:
            got = ask(model, q, w)
            if want in normalize(got):
                good += 1
            if not sample and got.strip():
                sample = got[:34]
        kb = used_bytes(model, w) / 1024.0
        pct = 100.0 * good / len(rows) if rows else 0.0
        print("%-8d %-9.0f %-9.1f%% %s" % (w, kb, pct, sample))
    return 0


if __name__ == "__main__":
    sys.exit(main())
