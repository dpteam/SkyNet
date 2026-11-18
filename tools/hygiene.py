#!/usr/bin/env python3
"""hygiene.py - enforce the core invariants of SkyNet on the source tree.

Why this exists: grep-based checks flag the words "float" and "malloc" when
they appear in a comment explaining that they are banned, so they get
disabled, and then they protect nothing.  This strips comments and string
literals first, then looks at actual tokens.

Checks (each maps to an invariant in docs/01-architecture.md):
  I1  no float/double          - determinism across targets
  I2  no malloc outside arena  - bounded memory, OOM returns NULL
  I2  no exit/abort in core    - never skip the checkpoint path
  I1  no rand/srand            - not portable, not reproducible
  --  no // comments, no long long, no stdint.h, no math.h  (C89)
  --  platform headers only in sk_io.c

Run:  python tools/hygiene.py [--verbose]
Exit: 0 clean, 1 violations
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

# the one file allowed to allocate directly, and the one allowed to know
# which operating system it is running on
ARENA_FILE = "sk_arena.c"

# Files allowed to include platform headers.  Both are platform layers by
# design: sk_io is the file/OS boundary, sk_mem is the BIOS/memory
# boundary (INT 67h for EMS).  Everything else must be portable C89, and
# this list is the whole list.
PLATFORM_FILES = {"sk_io.c", "sk_mem.c", "sk_plat.h"}
IO_FILE = "sk_io.c"

BANNED_KEYWORDS = {
    "float": "I1: no floating point in the core",
    "double": "I1: no floating point in the core",
}

BANNED_CALLS = {
    "rand": "I1: not reproducible across targets",
    "srand": "I1: not reproducible across targets",
    "malloc": "I2: all memory comes from the arena",
    "calloc": "I2: all memory comes from the arena",
    "realloc": "I2: all memory comes from the arena",
    "free": "I2: the arena uses mark/release, not free",
    "exit": "I2: exit skips the checkpoint path",
    "abort": "I2: abort loses the run",
    "longjmp": "I2: control flow must stay visible",
}

BANNED_INCLUDES = {
    "stdint.h": "not available in Turbo C 2.0",
    "math.h": "I1: floating point",
    "time.h": "I1: wall clock is not a valid time source",
    "pthread.h": "no threads on DOS",
    "threads.h": "no threads on DOS",
}

PLATFORM_INCLUDES = {
    "windows.h", "direct.h", "io.h", "dir.h", "unistd.h",
    "sys/stat.h", "sys/types.h", "conio.h", "dos.h",
}


def strip_comments_and_strings(text, keep_line_comments=False):
    """Replace comments and literals with spaces, preserving line numbers.

    With keep_line_comments=True, block comments are still removed but //
    comments are left intact, so a caller can search for // occurrences
    without matching the "//" that appears inside a block comment.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "*":
            out.append("  ")
            i += 2
            while i < n:
                if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                    out.append("  ")
                    i += 2
                    break
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
        elif c == "/" and nxt == "/":
            if keep_line_comments:
                out.append("//")
                i += 2
                while i < n and text[i] != "\n":
                    out.append(text[i])
                    i += 1
            else:
                # a // comment is itself a C89 violation, reported separately
                out.append("  ")
                i += 2
                while i < n and text[i] != "\n":
                    out.append(" ")
                    i += 1
        elif c in "\"'":
            quote = c
            out.append(" ")
            i += 1
            while i < n:
                if text[i] == "\\":
                    out.append("  ")
                    i += 2
                    continue
                if text[i] == quote:
                    out.append(" ")
                    i += 1
                    break
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def check_file(path, verbose):
    name = os.path.basename(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # // comments must be found after block comments are blanked, otherwise
    # the "//" inside a block comment that explains the ban trips the check
    problems = []
    for m in re.finditer(r"//", strip_comments_and_strings(raw, keep_line_comments=True)):
        problems.append((line_of(raw, m.start()), "C89: '//' comment"))

    code = strip_comments_and_strings(raw)

    for kw, why in BANNED_KEYWORDS.items():
        for m in re.finditer(r"\b%s\b" % kw, code):
            problems.append((line_of(code, m.start()), "%s: '%s'" % (why, kw)))

    if name != ARENA_FILE:
        for call, why in BANNED_CALLS.items():
            if name == ARENA_FILE and call == "malloc":
                continue
            for m in re.finditer(r"\b%s\s*\(" % call, code):
                problems.append((line_of(code, m.start()),
                                 "%s: '%s()'" % (why, call)))

    if re.search(r"\blong\s+long\b", code):
        for m in re.finditer(r"\blong\s+long\b", code):
            problems.append((line_of(code, m.start()), "C89: 'long long'"))

    for inc, why in BANNED_INCLUDES.items():
        pat = r'#\s*include\s*[<"]%s[>"]' % re.escape(inc)
        for m in re.finditer(pat, code):
            problems.append((line_of(code, m.start()),
                             "banned include <%s>: %s" % (inc, why)))

    for inc in PLATFORM_INCLUDES:
        pat = r'#\s*include\s*[<"]%s[>"]' % re.escape(inc)
        for m in re.finditer(pat, code):
            if name not in PLATFORM_FILES:
                problems.append((line_of(code, m.start()),
                                 "platform header <%s> belongs in %s only"
                                 % (inc, " or ".join(sorted(PLATFORM_FILES)))))

    if verbose and not problems:
        print("  ok   %s" % name)
    return problems


def check_modes():
    """Every mode in sk_modes needs a branch, and every branch a mode.

    This check exists because the two lists DID drift, silently: "ingest" and
    "shadow" were handled by the dispatcher and missing from the argument
    parser, so `skynet shadow model.bin` parsed "shadow" as a filename, fell
    through to the default mode and started the prime clock. The command
    looked like it ran - it printed plausible numbers - and only the wrong
    ones.

    Parsing the C here is crude on purpose: this is a consistency check on
    literal strings, not a compiler.
    """
    path = os.path.join(SRC, "main.c")
    text = io.open(path, encoding="utf-8").read()

    listed = set()
    m = re.search(r"sk_modes\[\]\s*=\s*\{(.*?)\};", text, re.S)
    if not m:
        return [(0, "main.c: sk_modes[] not found")]

    for s in re.findall(r'"([a-z]+)"', m.group(1)):
        if s in listed:
            return [(0, "main.c: mode %r listed twice" % s)]
        listed.add(s)

    dispatched = set(re.findall(r'strcmp\(o\.mode, "([a-z]+)"\)', text))

    problems = []
    for name in sorted(listed - dispatched):
        problems.append((0, 'main.c: mode "%s" is listed but has no branch'
                         % name))
    for name in sorted(dispatched - listed):
        problems.append((0, 'main.c: branch for "%s" is not in sk_modes[]'
                         % name))
    return problems


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    files = sorted(
        os.path.join(SRC, f) for f in os.listdir(SRC)
        if f.endswith(".c") or f.endswith(".h")
    )
    if not files:
        print("hygiene: no sources found in %s" % SRC)
        return 1

    total = 0
    for path in files:
        problems = check_file(path, verbose)
        for line, msg in problems:
            print("%s:%d: %s" % (os.path.basename(path), line, msg))
            total += 1

    for line, msg in check_modes():
        print(msg)
        total += 1

    if total:
        print("\nhygiene: %d violation(s). See docs/01-architecture.md" % total)
        return 1

    print("hygiene: clean (%d files)" % len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
