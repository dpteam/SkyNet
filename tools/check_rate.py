#!/usr/bin/env python3
"""check_rate.py - fail if a reported rate is below a threshold.

Used by CI to turn "35/35 correct (100.0%)" into a pass/fail without
hardcoding the count. The count changes whenever a question is added, and a
test that must be edited after every addition is a test that eventually gets
edited to pass.

Usage: check_rate.py <rate like 94.3%> <minimum percent> [label]
"""
import sys


def main():
    if len(sys.argv) < 3:
        print("usage: check_rate.py <rate> <min-percent> [label]")
        return 2
    rate = float(sys.argv[1].rstrip("%"))
    floor = float(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else "rate"

    print("%s: %.1f%% (floor %.1f%%)" % (label, rate, floor))
    if rate < floor:
        print("::error::%s dropped below %.1f%%" % (label, floor))
        return 1
    print("%s within budget" % label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
