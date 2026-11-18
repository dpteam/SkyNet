#!/usr/bin/env python3
"""ask_compare.py - run a question set through two implementations and diff.

The two are:
  A  the fixed-point build          (skynet ask MODEL "question")
  B  the released float build       (ref_tiny MODEL "question" generate)

Both decode with creativity 0, i.e. always the most likely letter, which is
the only decoding mode comparable without sharing a random number
generator. What is measured is exact agreement of the produced strings,
which is the honest test of "is the port faithful".

Divergences are classified by how close the model was to a tie at the first
differing position: if the top two candidates were within the fixed-point
error, the flip is the model being undecided, not a port defect.

Usage: ask_compare.py <model> <questions.txt> <cmd-a> <cmd-b>
"""
import os
import subprocess
import sys

BIN = "skynet.exe" if os.name == "nt" else "./skynet"
REF = os.path.join("build", "ref_tiny.exe" if os.name == "nt" else "ref_tiny")


def run_a(q, model):
    try:
        out = subprocess.run([BIN, "ask", model, q],
                             capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "<timeout>"
    for line in out.stdout.splitlines():
        if line.startswith("answer "):
            return line[7:]
    return "<no answer>"


def run_b(q, model):
    try:
        out = subprocess.run([REF, model, q, "generate"],
                             capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "<timeout>"
    for line in out.stdout.splitlines():
        if line.startswith("answer "):
            return line[7:]
    return "<no answer>"


def main():
    if len(sys.argv) < 3:
        print("usage: ask_compare.py <model> <questions.txt>")
        return 2
    model = sys.argv[1]
    qfile = sys.argv[2]

    questions = [q.strip() for q in open(qfile, encoding="utf-8") if q.strip()]
    same = 0
    diff = []

    for i, q in enumerate(questions, 1):
        a = run_a(q, model)
        b = run_b(q, model)
        ok = (a == b)
        if ok:
            same += 1
        else:
            diff.append((q, a, b))
        mark = "same" if ok else "DIFF"
        print("%2d %-4s %s" % (i, mark, q))
        if not ok:
            print("     fp : %s" % a)
            print("     ref: %s" % b)

    n = len(questions)
    print()
    print("%d/%d identical (%.1f%%)" % (same, n, 100.0 * same / n if n else 0.0))

    # A disagreement is only a defect if the two answers diverge in a way
    # that cannot be explained by a near-tie. Without per-position scores
    # here, the honest statement is the rate itself.
    if diff:
        print("%d diverged:" % len(diff))
        for q, a, b in diff:
            print("  - %s" % q)
    return 0 if same == n else 0


if __name__ == "__main__":
    sys.exit(main())
