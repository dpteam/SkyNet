"""verify_corpus.py - check every line of the corpus against an independent
evaluator.

Why "independent" is the whole point, and why this does not import
gen_corpus.py:

  The claim this corpus makes is that every answer is computed, so a wrong
  line is a bug rather than a bad source. That claim is worthless if the
  checker is the generator - it would confirm that the generator agrees with
  itself. So the answers here are worked out a second time, from the question
  text alone, using different code:

    - the arithmetic is parsed from the question and evaluated with Python's
      own operators, not read out of the record;
    - primality is a sieve over the whole range, where the generator used
      trial division per number;
    - primality, nth prime and super-prime are all re-derived from that sieve;
    - number words are parsed back to an integer and compared with the answer.

  If the two disagree, the line is printed and the exit status is non-zero.

This is the same principle as tests/ref_tiny.c: comparing an implementation
with its own rewrite proves only that the same mistake was made twice.

Usage:
  python3 tools/verify_corpus.py build/corpus.txt
  python3 tools/verify_corpus.py build/corpus.txt --sample 5000
"""
import io
import random
import re
import sys

MAXV = 20000          # sieve covers every number any recipe can ask about


def build_sieve(limit):
    ok = bytearray([1]) * (limit + 1)
    ok[0] = ok[1] = 0
    i = 2
    while i * i <= limit:
        if ok[i]:
            step = i
            start = i * i
            ok[start:limit + 1:step] = bytearray(len(range(start, limit + 1,
                                                             step)))
        i += 1
    return ok


SIEVE = build_sieve(MAXV)
PRIMES = [i for i in range(MAXV + 1) if SIEVE[i]]


def next_prime(n):
    k = n + 1
    while k <= MAXV:
        if SIEVE[k]:
            return k
        k += 1
    return None


def nth_prime(k):
    return PRIMES[k - 1] if 1 <= k <= len(PRIMES) else None


def super_prime(k):
    i = nth_prime(k)
    return nth_prime(i) if i else None


# ---------------------------------------------------------------- words
ONES = ('zero one two three four five six seven eight nine ten eleven twelve '
        'thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
        'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}


def parse_words(text):
    """Integer from English words. Written from the words themselves rather
    than by inverting the generator's formatter, so a formatter bug shows up
    as a disagreement instead of being mirrored."""
    text = text.strip().lower()
    if text.startswith('minus '):
        return -parse_words(text[6:])
    total, current = 0, 0
    for tok in re.split(r'[-\s]+', text):
        if tok in ('and', ''):
            continue
        if tok in ONES:
            current += ONES.index(tok)
        elif tok in TENS:
            current += TENS[tok]
        elif tok == 'hundred':
            current *= 100
        else:
            raise ValueError('unknown word %r' % tok)
    return total + current


def words(n):
    """English for 0..999, written from the rules of English rather than by
    importing the generator's formatter."""
    if n < 0:
        return 'minus ' + words(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        head = [w for w, v in TENS.items() if v == t * 10][0]
        return head if r == 0 else head + '-' + ONES[r]
    h, r = divmod(n, 100)
    out = ONES[h] + ' hundred'
    return out if r == 0 else out + ' and ' + words(r)


# ---------------------------------------------------------------- answers
def expect(question):
    """The answer the question should have, or None if unrecognised."""
    q = question.strip().lower()

    m = re.match(r'^(?:what is|how much is|add|sum of) (\d+) (?:plus|\+|and)'
                 r' (\d+)$', q)
    if m:
        return str(int(m.group(1)) + int(m.group(2)))
    m = re.match(r'^(?:what is|how much is) (\d+) - (\d+)$', q)
    if m:
        return str(int(m.group(1)) - int(m.group(2)))
    m = re.match(r'^subtract (\d+) from (\d+)$', q)
    if m:
        return str(int(m.group(2)) - int(m.group(1)))
    m = re.match(r'^what is (\d+) minus (\d+)$', q)
    if m:
        return str(int(m.group(1)) - int(m.group(2)))
    m = re.match(r'^(?:what is|how much is) (\d+) (?:times|x) (\d+)$', q)
    if m:
        return str(int(m.group(1)) * int(m.group(2)))
    m = re.match(r'^multiply (\d+) by (\d+)$', q)
    if m:
        return str(int(m.group(1)) * int(m.group(2)))
    m = re.match(r'^what is (\d+) divided by (\d+)$', q)
    if m:
        b = int(m.group(2))
        return str(int(m.group(1)) // b) if b else None
    m = re.match(r'^(\d+) / (\d+)$', q)
    if m:
        b = int(m.group(2))
        return str(int(m.group(1)) // b) if b else None
    m = re.match(r'^divide (\d+) by (\d+)$', q)
    if m:
        b = int(m.group(2))
        return str(int(m.group(1)) // b) if b else None

    # bare forms: "33 - 31", "54 + 38", "7 times 8"
    m = re.match(r'^(\d+) \+ (\d+)$', q)
    if m:
        return str(int(m.group(1)) + int(m.group(2)))
    m = re.match(r'^(\d+) - (\d+)$', q)
    if m:
        return str(int(m.group(1)) - int(m.group(2)))
    m = re.match(r'^(\d+) times (\d+)$', q)
    if m:
        return str(int(m.group(1)) * int(m.group(2)))

    # arithmetic asked in words, which is half the point of the corpus
    m = re.match(r'^what is (.+?) plus (.+)$', q)
    if m:
        try:
            return str(parse_words(m.group(1)) + parse_words(m.group(2)))
        except ValueError:
            return None

    m = re.match(r'^what is (\d+) squared$', q)
    if m:
        return str(int(m.group(1)) ** 2)
    m = re.match(r'^what is (\d+) cubed$', q)
    if m:
        return str(int(m.group(1)) ** 3)
    m = re.match(r'^what is (\d+) to the (\d+)$', q)
    if m:
        return str(int(m.group(1)) ** int(m.group(2)))

    m = re.match(r'^is (\d+) (?:a )?prime(?: number)?$', q)
    if m:
        return 'yes' if SIEVE[int(m.group(1))] else 'no'
    m = re.match(r'^what is the next prime after (\d+)$', q)
    if m:
        v = next_prime(int(m.group(1)))
        return str(v) if v is not None else None

    m = re.match(r'^which prime is number (\d+)$', q)
    if m:
        v = nth_prime(int(m.group(1)))
        return str(v) if v is not None else None
    m = re.match(r'^what is prime number (\d+)$', q)
    if m:
        v = nth_prime(int(m.group(1)))
        return str(v) if v is not None else None
    m = re.match(r'^what is super prime (\d+)$', q)
    if m:
        v = super_prime(int(m.group(1)))
        return str(v) if v is not None else None
    m = re.match(r'^which super prime is number (\d+)$', q)
    if m:
        v = super_prime(int(m.group(1)))
        return str(v) if v is not None else None

    m = re.match(r'^what is (\d+) in hex$', q)
    if m:
        return format(int(m.group(1)), 'x')
    m = re.match(r'^what is (\d+) in binary$', q)
    if m:
        return format(int(m.group(1)), 'b')
    m = re.match(r'^what is 0x([0-9a-f]+) in decimal$', q)
    if m:
        return str(int(m.group(1), 16))

    m = re.match(r'^how many bytes in a kilobyte$', q)
    if m:
        return '1024'
    m = re.match(r'^how many kilobytes in a megabyte$', q)
    if m:
        return '1024'
    m = re.match(r'^how many seconds in an hour$', q)
    if m:
        return '3600'
    m = re.match(r'^how many bits in (\d+) bytes$', q)
    if m:
        return str(int(m.group(1)) * 8)
    m = re.match(r'^how many bits in (\d+) kilobytes$', q)
    if m:
        return str(int(m.group(1)) * 1024 * 8)

    m = re.match(r'^(?:which is bigger|which number is larger|what is larger)'
                 r' (\d+) or (\d+)$', q)
    if m:
        return str(max(int(m.group(1)), int(m.group(2))))
    m = re.match(r'^is (\d+) (?:bigger|larger|greater) than (\d+)$', q)
    if m:
        return 'yes' if int(m.group(1)) > int(m.group(2)) else 'no'

    m = re.match(r'^how do you write (.+?) in digits$', q)
    if m:
        try:
            return str(parse_words(m.group(1)))
        except ValueError:
            return None
    m = re.match(r'^what is (\d+) in words$', q)
    if m:
        return words(int(m.group(1)))
    m = re.match(r'^spell (\d+) digit by digit$', q)
    if m:
        names = [ONES[int(c)] for c in m.group(1)]
        return ' '.join(names)

    m = re.match(r'^what is the first letter of (\w+)$', q)
    if m:
        return m.group(1)[0]
    m = re.match(r'^what is the last letter of (\w+)$', q)
    if m:
        return m.group(1)[-1]
    m = re.match(r'^how many letters in (\w+)$', q)
    if m:
        return str(len(m.group(1)))
    m = re.match(r'^spell (\w+) backwards$', q)
    if m:
        return m.group(1)[::-1]
    m = re.match(r'^what letter is number (\d+) in (\w+)$', q)
    if m:
        i = int(m.group(1)) - 1
        w = m.group(2)
        return w[i] if 0 <= i < len(w) else None
    m = re.match(r'^what comes after (\w) in the alphabet$', q)
    if m:
        return chr(ord(m.group(1)) + 1) if m.group(1) != 'z' else 'a'

    return None


def main(argv):
    path = argv[1] if len(argv) > 1 else 'build/corpus.txt'
    sample = None
    for i, a in enumerate(argv):
        if a == '--sample' and i + 1 < len(argv):
            sample = int(argv[i + 1])

    lines = io.open(path, encoding='utf-8').read().split('\n')
    lines = [l for l in lines if l.strip()]
    if sample and sample < len(lines):
        rng = random.Random(0)
        lines = rng.sample(lines, sample)

    ok = 0
    unrecognised = []
    wrong = []
    for line in lines:
        if '|' not in line:
            wrong.append((line, 'no separator'))
            continue
        q, a = line.split('|', 1)
        want = expect(q)
        if want is None:
            unrecognised.append(line)
            continue
        if want == a:
            ok += 1
        else:
            wrong.append((line, want))

    total = len(lines)
    print('checked %d lines: %d agree, %d wrong, %d not understood' % (
        total, ok, len(wrong), len(unrecognised)))
    for line, want in wrong[:20]:
        print('  WRONG  %s   expected %r' % (line[:70], want))
    for line in unrecognised[:10]:
        print('  ?      %s' % line[:70])
    if wrong:
        return 1
    if unrecognised:
        # Not a failure: these are shapes this checker does not parse yet.
        # Reported so the coverage figure is honest rather than implied.
        print('\nevery parsed line is correct; %d lines are outside the '
              'checker' % len(unrecognised))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
