#!/usr/bin/env python3
"""build_dataset.py - turn the Tiny-2.1 fact files into training pairs.

Two source formats:
  facts/*.txt   question | question | ... => answer | answer | ...
  facts/*.tsv   relation <TAB> subject <TAB> answer

The model reads and writes one letter at a time, so a training example is
literally "question, then newline, then answer, then newline". Teacher
forcing means the loss is computed on every answer letter given the true
prefix, which is what makes a 97-symbol model trainable at all.

Output is one example per line, with '|' separating question from answer,
so the C trainer needs no parsing beyond a split.

Usage: build_dataset.py [outfile] [limit]
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS = os.path.join(ROOT, "refs", "Tiny-2.1-main", "training", "facts")

# The model's vocabulary is ' '..'~' (tokens 2..96) plus 0 = pad, 1 = newline.
# Anything outside that range must be dropped, not approximated.
ALLOWED = set(chr(c) for c in range(32, 127))


def clean(s):
    s = s.strip().lower()
    s = "".join(ch for ch in s if ch in ALLOWED)
    # collapse runs of spaces so the model sees one canonical form
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


def from_txt(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or line.startswith("#") or "=>" not in line:
            continue
        left, right = line.split("=>", 1)
        questions = [q for q in left.split("|") if q.strip()]
        answers = [a for a in right.split("|") if a.strip()]
        if not questions or not answers:
            continue
        for q in questions:
            for a in answers:
                out.append((clean(q), clean(a)))
    return out


def from_tsv(path):
    out = []
    rel = os.path.basename(path).replace(".tsv", "")
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        relation, subject, answer = parts[0], parts[1], parts[2]
        if not subject.strip() or not answer.strip():
            continue
        # A plain, unambiguous question shape. The released generator makes
        # dozens of wordings; one is enough to prove the trainer learns.
        q = "%s of %s" % (relation.replace("_", " "), subject)
        out.append((clean(q), clean(answer)))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "build/dataset.txt"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    pairs = []
    for path in sorted(glob.glob(os.path.join(FACTS, "*.txt"))):
        pairs += from_txt(path)
    for path in sorted(glob.glob(os.path.join(FACTS, "*.tsv"))):
        pairs += from_tsv(path)

    # Keep only examples the model can represent: both sides non-empty,
    # total length under the context window with room for the newline.
    usable = []
    for q, a in pairs:
        if not q or not a:
            continue
        if len(q) + len(a) + 3 > 120:
            continue
        usable.append((q, a))

    # Deduplicate: repeated pairs teach nothing and bias the loss.
    seen = set()
    uniq = []
    for q, a in usable:
        key = (q, a)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)

    if limit:
        uniq = uniq[:limit]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for q, a in uniq:
            f.write("%s|%s\n" % (q, a))

    print("pairs from facts : %d" % len(pairs))
    print("usable           : %d" % len(usable))
    print("unique           : %d" % len(uniq))
    print("written          : %s" % out_path)

    lens = [len(q) + len(a) for q, a in uniq]
    if lens:
        print("length min/avg/max: %d / %d / %d"
              % (min(lens), sum(lens) // len(lens), max(lens)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
