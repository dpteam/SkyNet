/* sk_clock.c - the Prime Clock.  See sk_clock.h.
 *
 * Segmented sieve of Eratosthenes.  Base primes up to 65536 are built
 * once; each segment then crosses off its own composites.  Memory is
 * fixed and known at compile time (about 25 KB), which is what a DOS
 * build needs to know before it asks the arena for anything.
 *
 * SKYNET M0 / subsystem: core/clock
 */
#include <string.h>
#include "sk_clock.h"

#define SK_BASE_LIMIT 65536
#define SK_BASE_MAX   7000      /* pi(65536) = 6542 */

static sk_u16 sk_base_primes[SK_BASE_MAX];
static sk_u32 sk_base_count;
static int    sk_base_ready;
static sk_u8  sk_base_bits[SK_BASE_LIMIT / 8];
static sk_u8  sk_flags[SK_CLOCK_WINDOW];

static const sk_u32 sk_small[3] = { 2UL, 3UL, 5UL };

static void sk_clock_base(void)
{
    sk_u32 i, j;

    if (sk_base_ready)
        return;
    memset(sk_base_bits, 0, sizeof(sk_base_bits));
    sk_base_bits[0] = 3;               /* 0 and 1 are not prime */
    sk_base_count = 0;
    for (i = 2; i < (sk_u32)SK_BASE_LIMIT; i++) {
        if (sk_base_bits[i >> 3] & (sk_u8)(1 << (i & 7)))
            continue;
        if (sk_base_count < (sk_u32)SK_BASE_MAX)
            sk_base_primes[sk_base_count++] = (sk_u16)i;
        for (j = i * i; j < (sk_u32)SK_BASE_LIMIT; j += i)
            sk_base_bits[j >> 3] |= (sk_u8)(1 << (j & 7));
    }
    sk_base_ready = 1;
}

int sk_clock_is_prime(sk_u32 n)
{
    sk_u32 i, p;

    if (n < 2)
        return 0;
    if (!sk_base_ready)
        sk_clock_base();
    for (i = 0; i < sk_base_count; i++) {
        p = (sk_u32)sk_base_primes[i];
        if (p > n / p)                 /* p*p > n without overflowing */
            return 1;
        if (n % p == 0)
            return 0;
    }
    return 1;
}

/* Cross off the composites of one segment of odd candidates.
   On return c->seg_lo is the first candidate of the segment and c->next
   points past it, so a segment is never built twice. */
static void sk_clock_segment(sk_clock *c)
{
    sk_u32 i, p, start, k, span, seg_hi;

    if (!sk_base_ready)
        sk_clock_base();

    span = (sk_u32)SK_CLOCK_WINDOW * 2UL;
    c->seg_lo = c->next;
    seg_hi = c->seg_lo + span;
    if (seg_hi > SK_CLOCK_LIMIT || seg_hi < c->seg_lo)
        seg_hi = SK_CLOCK_LIMIT;
    c->next = seg_hi;

    memset(sk_flags, 0, (size_t)SK_CLOCK_WINDOW);
    if (seg_hi <= c->seg_lo)
        return;

    for (i = 0; i < sk_base_count; i++) {
        p = (sk_u32)sk_base_primes[i];
        if (p == 2)
            continue;                  /* we only sieve odd candidates */
        if (p * p >= seg_hi)
            break;
        start = p * p;
        if (start < c->seg_lo)
            start = c->seg_lo + ((p - (c->seg_lo % p)) % p);
        if ((start & 1UL) == 0)
            start += p;                /* p is odd here: makes start odd */
        for (k = start; k < seg_hi; k += p * 2UL)
            sk_flags[(k - c->seg_lo) >> 1] = 1;
    }
}

void sk_clock_init(sk_clock *c)
{
    memset(c, 0, sizeof(*c));
    c->next = 7;                       /* 2, 3, 5 are served directly */
    c->phase = 0;
    c->seg_pos = (sk_u32)SK_CLOCK_WINDOW;
    c->ready = 1;
    if (!sk_base_ready)
        sk_clock_base();
}

void sk_clock_seek(sk_clock *c, sk_u32 last_value, sk_u32 last_index)
{
    sk_clock_init(c);
    if (last_index < 3 || last_value < 5)
        return;                        /* replay from the beginning */

    c->phase = 3;
    c->value = last_value;
    c->index = last_index;
    c->next = last_value + 2;
    if ((c->next & 1UL) == 0)
        c->next++;
    c->seg_pos = (sk_u32)SK_CLOCK_WINDOW;
}

int sk_clock_next(sk_clock *c)
{
    if (!c->ready)
        sk_clock_init(c);

    for (;;) {
        sk_u32 pos, cand;

        if (c->phase < 3) {
            c->value = sk_small[c->phase];
            c->phase++;
            c->index++;
            c->is_super = sk_clock_is_prime(c->index);
            if (c->is_super)
                c->supers++;
            return 1;
        }

        if (c->seg_pos >= (sk_u32)SK_CLOCK_WINDOW) {
            if (c->seg_lo >= SK_CLOCK_LIMIT)
                return 0;
            sk_clock_segment(c);
            c->seg_pos = 0;
            continue;
        }

        pos = c->seg_pos++;
        cand = c->seg_lo + pos * 2UL;
        if (cand >= SK_CLOCK_LIMIT)
            return 0;
        if (!sk_flags[pos]) {
            c->value = cand;
            c->index++;
            c->is_super = sk_clock_is_prime(c->index);
            if (c->is_super)
                c->supers++;
            return 1;
        }
    }
}

sk_u32 sk_clock_supers_upto(sk_clock *c, sk_u32 n)
{
    sk_clock tmp;
    sk_u32 cnt;

    tmp = *c;
    cnt = 0;
    while (tmp.index < n) {
        if (!sk_clock_next(&tmp))
            break;
        if (tmp.is_super)
            cnt++;
    }
    return cnt;
}
