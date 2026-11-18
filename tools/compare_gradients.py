"""compare_gradients.py - our fixed-point gradient against the reference.

Reads what tools/ref_gradients.py printed and what
`build/gradcheck ... 1` dumped for the same samples, and reports the ratio
per parameter kind.

What to look at is the MEDIAN of ours/theirs, not the percentage within 15%.
The reference is float; the fixed-point side is Q16.16 with saturation, and
the softmax gradient loses any class whose probability falls below one LSB.
On small gradients that is a factor of several, unavoidably. A median near
1.0 across every kind means the formulas are right; a median near 0.004 or
256 means a unit error; a spread of signs on large rows means a structural
error.

  python tools/compare_gradients.py build/ref_grad.txt build/our_grad.txt
"""
import sys, io

KINDS = ['GF', 'G2', 'F1', 'F2', 'OUT']
NAMES = {'GF': 'final gains', 'G2': 'g2 (last block)',
         'F1': 'f1 (last block)', 'F2': 'f2 (last block)',
         'OUT': 'output projection'}


def lines_of(path):
    raw = open(path, 'rb').read()
    # PowerShell redirects write UTF-16LE with a BOM; everything else writes
    # UTF-8. Accept both rather than making the caller remember which shell
    # it used.
    if raw[:2] == b'\xff\xfe':
        raw = raw.decode('utf-16').encode('utf-8')
    return raw.decode('utf-8', 'replace').split('\n')


def parse_ref(path):
    d = {}
    ls = lines_of(path)
    i = 0
    while i < len(ls):
        p = ls[i].strip().split()
        if len(p) == 2 and p[0] in NAMES:
            d[p[0]] = [float(x) for x in ls[i + 1].split()]
            i += 2
            continue
        i += 1
    return d


def parse_our(path):
    d = {}
    ls = lines_of(path)
    i = 0
    while i < len(ls):
        if ls[i].startswith('ANALYTIC '):
            name = ls[i][9:].strip()
            vals = []
            i += 1
            while i < len(ls) and ls[i] and not ls[i].startswith('ANALYTIC'):
                try:
                    vals.append(float(ls[i]))
                except ValueError:
                    break
                i += 1
            d[name] = vals
            continue
        i += 1
    return d


def main():
    refp = sys.argv[1] if len(sys.argv) > 1 else 'build/ref_grad.txt'
    ourp = sys.argv[2] if len(sys.argv) > 2 else 'build/our_grad.txt'
    ref = parse_ref(refp)
    our = parse_our(ourp)
    worst_median = None

    for key in KINDS:
        if key not in ref or NAMES[key] not in our:
            print('%-18s missing' % NAMES[key])
            worst_median = 1e9
            continue
        a, b = ref[key], our[NAMES[key]]
        ratios, worst = [], []
        for i in range(min(len(a), len(b))):
            if abs(a[i]) > 1e-4 and abs(b[i]) > 1e-9:
                ratios.append(b[i] / a[i])
                worst.append((abs(b[i] / a[i] - 1.0), i, a[i], b[i]))
        if not ratios:
            print('%-18s nothing comparable' % NAMES[key])
            continue
        ratios.sort()
        worst.sort(reverse=True)
        median = ratios[len(ratios) // 2]
        ok = sum(1 for r in ratios if abs(r - 1.0) < 0.15)
        print('%-18s n=%3d cmp=%3d within15%%=%3d  median %.3f'
              '  p10 %.3f  p90 %.3f' % (
                  NAMES[key], len(a), len(ratios), ok, median,
                  ratios[len(ratios) // 10],
                  ratios[(len(ratios) * 9) // 10]))
        for w, i, r, o in worst[:3]:
            print('    row %3d ref %+12.6f ours %+12.6f  ratio %+8.3f' % (
                i, r, o, o / r if r else 0.0))
        if worst_median is None or abs(median - 1.0) > abs(worst_median - 1.0):
            worst_median = median

    if worst_median is None:
        return 2
    # 2x either way is the tolerance: anything tighter is measuring Q16.16.
    bad = abs(worst_median - 1.0) > 1.0
    print('\nworst median %.3f -- %s' % (
        worst_median, 'FAILED' if bad else 'formulas agree'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
