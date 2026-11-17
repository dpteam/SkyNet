/* sk_clock.h - the Prime Clock.
 *
 * This is SkyNet's only source of time.  Not the RTC, not the wall clock,
 * not a tick counter that resets: the sequence of primes, and every
 * super-prime (the p_n for which n is itself prime) is a checkpoint,
 * a self-test and a scheduler rebalance point.
 *
 * Why: a machine that must run for 50 years cannot trust a clock that
 * drifts, and cannot trust a "resume from tick 9 999 999 999" that
 * depends on how fast the previous CPU was.  Prime index is monotone,
 * reproducible, and verifiable in one line of arithmetic.
 *
 * Super-primes (OEIS A006450 - primes whose index is itself prime):
 *   3, 5, 11, 17, 31, 41, 59, 67, 83, 109, 127, 157, 179, 191, 211, ...
 * Every one of them is a checkpoint, a self-test and a scheduler
 * rebalance point.
 *
 * SKYNET M0 / subsystem: core/clock
 */
#ifndef SK_CLOCK_H
#define SK_CLOCK_H

#include "sk_types.h"

/* Segment width in odd candidates.  Costs 2*W bytes of flags. */
#ifndef SK_CLOCK_WINDOW
#  define SK_CLOCK_WINDOW 4096
#endif

/* Highest value we will ever sieve to. */
#define SK_CLOCK_LIMIT 0xFFFFFFF0UL

typedef struct {
    sk_u32 value;       /* last prime handed out                     */
    sk_u32 index;       /* 1-based index of value in the prime series */
    sk_u32 supers;      /* how many super-primes we have crossed      */
    sk_u32 next;        /* next odd candidate to sieve                */
    sk_u32 seg_lo;      /* first odd candidate in the current segment */
    sk_u32 seg_pos;     /* cursor inside the segment                  */
    int    phase;       /* 0..2 small primes, 3 sieving               */
    int    is_super;    /* set by sk_clock_next() for the last value  */
    int    ready;
} sk_clock;

void sk_clock_init(sk_clock *c);

/* Continue after last_value/last_index.  Used by the recovery path. */
void sk_clock_seek(sk_clock *c, sk_u32 last_value, sk_u32 last_index);

/* Advance.  Returns 1 and fills c->value / c->index / c->is_super,
   or 0 when the clock is exhausted (past SK_CLOCK_LIMIT). */
int sk_clock_next(sk_clock *c);

/* Is n prime?  Used to decide whether an index is a super-prime index.
   Trial division by the base primes: cheap, and n stays small. */
int sk_clock_is_prime(sk_u32 n);

/* Number of super-primes in the first n primes - used by the selftest. */
sk_u32 sk_clock_supers_upto(sk_clock *c, sk_u32 n);

#endif /* SK_CLOCK_H */
