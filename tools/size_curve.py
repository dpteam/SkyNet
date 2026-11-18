#!/usr/bin/env python3
"""size_curve.py - quality against size, measured instead of guessed.

The question M3 has to answer is "what do I lose if I only have N bytes".
This answers it by truncating the released model to its first N layers - a
shallower network is still a valid network - and measuring two things:

  bytes     weight storage the truncated model actually occupies
  accuracy  fraction of the documented questions answered correctly

The point is to find the knee: the smallest model that is still useful. That
number decides whether a progressive format is worth building and where the
cut has to be.

Usage: size_curve.py <model> <expected.txt> [max-layers]
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


def ask(model, question, layers):
    cmd = [BIN, "ask", model, question]
    if layers:
        cmd += ["--layers", str(layers)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "<timeout>"
    for line in out.stdout.splitlines():
        if line.startswith("answer "):
            return line[7:]
    return "<no answer>"


def model_bytes(model, layers):
    """Weight storage for the truncated model, from the program itself."""
    cmd = [BIN, "model", model]
    if layers:
        cmd += ["--layers", str(layers)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return 0
    for line in out.stdout.splitlines():
        if "weights" in line and "used" in line:
            # "[sky] weights 381796 bytes, 381796 used, att_div=..."
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
        print("usage: size_curve.py <model> <expected.txt> [max-layers]")
        return 2
    model, expected = sys.argv[1], sys.argv[2]
    max_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    rows = load_expected(expected)
    print("%-7s %-9s %-9s %s" % ("layers", "KB", "accuracy", "sample answer"))

    for n in range(1, max_layers + 1):
        good = 0
        sample = ""
        for q, want in rows:
            got = ask(model, q, n)
            if want in normalize(got):
                good += 1
            if not sample and got.strip():
                sample = got[:36]
        kb = model_bytes(model, n) / 1024.0
        pct = 100.0 * good / len(rows) if rows else 0.0
        print("%-7d %-9.0f %-9.1f%% %s" % (n, kb, pct, sample))

    return 0


if __name__ == "__main__":
    sys.exit(main())
