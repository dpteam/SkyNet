"""gen_corpus.py - our own training corpus, generated rather than borrowed.

Why not someone else's corpus:

  The released TINY.BIN was trained on 3,500 facts multiplied by a 57 KB
  generator into about 15 million examples. The facts are not the data set -
  the generator is. Copying facts buys 3,500 facts; writing the generator
  buys an unbounded corpus, and this one has a property no borrowed corpus
  has: every answer is CHECKED BY COMPUTATION, so a wrong answer is a bug in
  the generator, not a bad line nobody noticed.

  That is also why the subject matter is mathematics. "What is the capital
  of Burkina Faso" needs a source and a licence; "what is the next prime
  after 97" needs `is_prime` and nothing else. The model can be graded
  exactly, automatically, forever, by code that has no opinion.

Wording matters as much as facts: a model that only ever sees
"97 + 41" will not answer "please add ninety-seven and forty-one". So every
fact is emitted through several templates and two number spellings.

Usage:
  python3 tools/gen_corpus.py build/corpus.txt            # default volume
  python3 tools/gen_corpus.py build/corpus.txt 200000     # cap the lines
  python3 tools/gen_corpus.py --stats                     # what is in it

Output is one example per line, `question|answer`, the same format
tools/build_dataset.py writes and the trainer reads.
"""
import io
import os
import random
import sys

MAXLEN = 126          # question + separator + answer must fit SK_TRAIN_MAXT
SEED = 20261118       # fixed: the corpus must be reproducible byte for byte


# ------------------------------------------------------------------ numbers
ONES = ('zero one two three four five six seven eight nine ten eleven twelve '
        'thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
TENS = ('_ _ twenty thirty forty fifty sixty seventy eighty ninety').split()


def word(n):
    """English for 0..999. Deterministic, no library, no locale."""
    if n < 0:
        return 'minus ' + word(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return TENS[t] if r == 0 else TENS[t] + '-' + ONES[r]
    h, r = divmod(n, 100)
    head = ONES[h] + ' hundred'
    return head if r == 0 else head + ' and ' + word(r)


def ordinal(n):
    if 10 <= n % 100 <= 20:
        suf = 'th'
    else:
        suf = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return word(n) + suf


def digits(n):
    """Spelled out digit by digit: '97' -> 'nine seven'."""
    return ' '.join(ONES[int(c)] for c in str(n))


# ------------------------------------------------------------------ maths
def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def next_prime(n):
    n += 1
    while not is_prime(n):
        n += 1
    return n


def nth_prime(k):
    """The k-th prime, 1-based. The prime clock in sk_clock.c does this with a
    segmented sieve; here the volumes are small enough to do it directly, and
    the point is that the ANSWER is arithmetic, not that the method matches."""
    n, count = 2, 0
    while True:
        if is_prime(n):
            count += 1
            if count == k:
                return n
        n += 1


def super_prime(k):
    """A006450: the prime whose index is prime. The same sequence the
    checkpoint clock uses, so the model is being taught the project's own
    notion of time."""
    return nth_prime(nth_prime(k))


# ------------------------------------------------------------------ recipes
# Each recipe yields (question, answer) strings. Keeping them as separate
# functions means the per-topic counts can be measured independently - which
# is what decides whether a topic is under-represented.

def r_arith(rng, n):
    out = []
    for _ in range(n):
        a = rng.randint(3, 99)
        b = rng.randint(3, 99)
        kind = rng.randrange(4)
        if kind == 0:
            q = rng.choice(('what is %d plus %d', 'how much is %d + %d',
                            'add %d and %d', '%d + %d',
                            'what is %d plus %d',
                            'sum of %d and %d'))
            out.append((q % (a, b), str(a + b)))
            if rng.randrange(4) == 0:
                out.append(('what is %s plus %s' % (word(a), word(b)),
                            str(a + b)))
        elif kind == 1:
            big, small = max(a, b), min(a, b)
            q = rng.choice(('what is %d minus %d', 'how much is %d - %d',
                            'subtract %d from %d', '%d - %d'))
            if q.count('%d') == 2 and 'subtract' in q:
                out.append((q % (small, big), str(big - small)))
            else:
                out.append((q % (big, small), str(big - small)))
        elif kind == 2:
            a = rng.randint(2, 12)
            b = rng.randint(2, 12)
            q = rng.choice(('what is %d times %d', 'how much is %d x %d',
                            '%d times %d', 'multiply %d by %d'))
            out.append((q % (a, b), str(a * b)))
        else:
            b = rng.randint(2, 12)
            q = rng.randint(1, 12) * b
            qq = rng.choice(('what is %d divided by %d', '%d / %d',
                             'divide %d by %d'))
            out.append((qq % (q, b), str(q // b)))
    return out


def r_words(rng, n):
    out = []
    for _ in range(n):
        v = rng.randint(0, 999)
        k = rng.randrange(3)
        if k == 0:
            out.append(('how do you write %s in digits' % word(v), str(v)))
        elif k == 1:
            out.append(('what is %d in words' % v, word(v)))
        else:
            out.append(('spell %d digit by digit' % v, digits(v)))
    return out


def r_primes(rng, n):
    out = []
    for _ in range(n):
        v = rng.randint(2, 2000)
        j = rng.randrange(3)
        if j == 0:
            out.append(('is %d a prime number' % v,
                        'yes' if is_prime(v) else 'no'))
        elif j == 1:
            out.append(('what is the next prime after %d' % v,
                        str(next_prime(v))))
        else:
            out.append(('is %d prime' % v, 'yes' if is_prime(v) else 'no'))
    return out


def r_clock(rng, n):
    """The prime clock: super-primes as checkpoints, its own time source."""
    out = []
    for _ in range(n):
        k = rng.randint(1, 60)
        j = rng.randrange(4)
        if j == 0:
            out.append(('which prime is number %d' % k, str(nth_prime(k))))
        elif j == 1:
            out.append(('what is prime number %d' % k, str(nth_prime(k))))
        elif j == 2:
            out.append(('what is super prime %d' % k, str(super_prime(k))))
        else:
            out.append(('which super prime is number %d' % k,
                        str(super_prime(k))))
    return out


def r_powers(rng, n):
    out = []
    for _ in range(n):
        v = rng.randint(2, 99)
        k = rng.randrange(4)
        if k == 0:
            out.append(('what is %d squared' % v, str(v * v)))
        elif k == 1:
            out.append(('what is %d cubed' % v, str(v * v * v)))
        elif k == 2:
            e = rng.randint(1, 30)
            out.append(('what is 2 to the %d' % e, str(2 ** e)))
        else:
            e = rng.randint(1, 19)
            out.append(('what is 3 to the %d' % e, str(3 ** e)))
    return out


def r_bases(rng, n):
    out = []
    for _ in range(n):
        v = rng.randint(0, 255)
        k = rng.randrange(3)
        if k == 0:
            out.append(('what is %d in hex' % v, format(v, 'x')))
        elif k == 1:
            out.append(('what is %d in binary' % v, format(v, 'b')))
        else:
            out.append(('what is 0x%x in decimal' % v, str(v)))
    return out


def r_units(rng, n):
    """Binary prefixes and time: exact by definition, so verifiable."""
    out = []
    for _ in range(n):
        k = rng.randrange(5)
        if k == 0:
            out.append(('how many bytes in a kilobyte', '1024'))
        elif k == 1:
            out.append(('how many kilobytes in a megabyte', '1024'))
        elif k == 2:
            out.append(('how many seconds in an hour', '3600'))
        elif k == 3:
            v = rng.randint(1, 16)
            out.append(('how many bits in %d bytes' % v, str(v * 8)))
        else:
            v = rng.randint(1, 8)
            out.append(('how many bits in %d kilobytes' % v, str(v * 1024 * 8)))
    return out


def r_compare(rng, n):
    """Comparison, in the two shapes that have different answer forms.

    The first version picked the shape by testing whether the template
    contained "is ", which is true of "which is bigger X or Y" as well as
    "is X bigger than Y" - so a quarter of the answers were "no" to a
    question that asked for a number. A template and its answer have to be
    chosen together.
    """
    out = []
    for _ in range(n):
        a = rng.randint(0, 999)
        b = rng.randint(0, 999)
        if a == b:
            b = (b + 1) % 1000
        big, small = max(a, b), min(a, b)
        if rng.randrange(2) == 0:
            # answer is the larger number
            q = rng.choice(('which is bigger %d or %d',
                            'which number is larger %d or %d',
                            'what is larger %d or %d'))
            out.append((q % (a, b), str(big)))
        else:
            # answer is yes or no
            q = rng.choice(('is %d bigger than %d', 'is %d larger than %d',
                            'is %d greater than %d'))
            out.append((q % (a, b), 'yes' if a > b else 'no'))
        del small
    return out


def r_letters(rng, n):
    out = []
    words = ('prime sieve arithmetic clock checkpoint arena vault scheduler '
             'memory tensor gradient model runtime trainer deterministic '
             'cooperative segment window factor exponent logarithm square '
             'root quotient remainder divisor multiple sequence alphabet '
             'integer fraction decimal binary hexadecimal register buffer '
             'pointer structure function variable constant operator compiler '
             'linker loader kernel device driver interrupt protocol packet '
             'checksum parity entropy modulus radix mantissa sign exponent '
             'overflow underflow precision accuracy tolerance threshold').split()
    for _ in range(n):
        w = rng.choice(words)
        k = rng.randrange(4)
        if k == 0:
            out.append(('what is the first letter of %s' % w, w[0]))
        elif k == 1:
            out.append(('what is the last letter of %s' % w, w[-1]))
        elif k == 2:
            out.append(('how many letters in %s' % w, str(len(w))))
        elif k == 3:
            out.append(('spell %s backwards' % w, w[::-1]))
        elif k == 4:
            i = rng.randrange(len(w))
            out.append(('what letter is number %d in %s' % (i + 1, w), w[i]))
        else:
            out.append(('what comes after %s in the alphabet' % w[0],
                        chr(ord(w[0]) + 1) if w[0] != 'z' else 'a'))
    return out


RECIPES = (
    ('arithmetic', r_arith, 0.30),
    ('number words', r_words, 0.12),
    ('primes', r_primes, 0.14),
    ('prime clock', r_clock, 0.10),
    ('powers', r_powers, 0.10),
    ('bases', r_bases, 0.12),
    ('units', r_units, 0.02),
    ('comparison', r_compare, 0.04),
    ('letters', r_letters, 0.06),
)


def build(total):
    rng = random.Random(SEED)
    seen = set()
    lines = []
    counts = dict((name, 0) for name, _, _ in RECIPES)

    # Round-robin by weight rather than by blocks, so a truncated corpus
    # still has every topic in it. A corpus that stops halfway through the
    # alphabet teaches a model that the alphabet ends at 'm'.
    quota = [(name, fn, int(round(total * w))) for name, fn, w in RECIPES]
    for name, fn, want in quota:
        got = 0
        # A recipe can have a bounded output space - "how many seconds in an
        # hour" has one answer - so a quota it can never fill must not become
        # an infinite loop. The first version of this did exactly that:
        # asking for 10 000 distinct unit facts hung the generator rather
        # than producing the ten that exist.
        attempts = 0
        limit = want * 20 + 1000
        while got < want and attempts < limit:
            batch = max(64, want // 8)
            attempts += batch
            fresh = 0
            for q, a in fn(rng, batch):
                if len(q) + len(a) + 1 > MAXLEN:
                    continue
                key = (q, a)
                if key in seen:
                    continue
                seen.add(key)
                lines.append('%s|%s' % (q, a))
                counts[name] += 1
                got += 1
                fresh += 1
                if got >= want:
                    break
            if fresh == 0:
                break               # the well is dry; take what we have
    # A topic that could not fill its quota leaves the corpus short of
    # `total`. Say so in the counts and let main() report it, rather than
    # quietly writing a smaller file than was asked for.
    rng.shuffle(lines)
    return lines, counts


def main(argv):
    stats = '--stats' in argv
    args = [a for a in argv[1:] if not a.startswith('--')]
    out = args[0] if args else os.path.join('build', 'corpus.txt')
    total = int(args[1]) if len(args) > 1 else 200000

    lines, counts = build(total)

    if stats:
        print('%-14s %8s  %5s' % ('topic', 'lines', 'share'))
        for name, _, _ in RECIPES:
            print('%-14s %8d  %4.1f%%' % (
                name, counts[name], 100.0 * counts[name] / max(1, len(lines))))
        print('%-14s %8d' % ('total', len(lines)))
        qs = set(l.split('|', 1)[0] for l in lines)
        print('%-14s %8d' % ('distinct Q', len(qs)))
        print('%-14s %8d' % ('longest', max(len(l) for l in lines)))
        for l in lines[:5]:
            print('   ' + l)
        return 0

    d = os.path.dirname(out)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    io.open(out, 'w', encoding='ascii', newline='\n').write(
        '\n'.join(lines) + '\n')
    print('wrote %s: %d lines, %d distinct questions' % (
        out, len(lines),
        len(set(l.split('|', 1)[0] for l in lines))))
    if len(lines) < total:
        # Say it out loud. The generator has a bounded number of distinct
        # facts per topic, so asking for more than the topics contain would
        # otherwise produce a quietly smaller file and a mystery later.
        print('note: asked for %d, produced %d - the recipes have that many'
              ' distinct facts between them' % (total, len(lines)))
    for name, _, _ in RECIPES:
        print('   %-14s %7d' % (name, counts[name]))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
