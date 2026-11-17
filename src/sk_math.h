/* sk_math.h - SkyNet math core.
 *
 * Everything numeric in SkyNet goes through this file.  No <math.h>, no
 * floating point, no rand().  Deterministic on every target: the same
 * inputs produce bit-identical outputs on an 8088 and on x86_64.
 *
 * SKYNET M0 / subsystem: core/math
 */
#ifndef SK_MATH_H
#define SK_MATH_H

#include "sk_types.h"

/* ---- constants --------------------------------------------------------
   Constants are DERIVED at init, not typed in.  pi comes from Machin's
   formula, e from its series, sqrt2 from Newton.  A typed-in constant is
   a claim; a derived one is a theorem. */
typedef struct {
    sk_fix pi;
    sk_fix e;
    sk_fix sqrt2;
    sk_fix ln2;
    sk_fix golden;      /* (1+sqrt5)/2 */
    sk_fix feigenbaum;  /* 4.6692 - the period-doubling constant */
} sk_consts;

void sk_math_init(void);
const sk_consts *sk_consts_get(void);

/* ---- fixed point ------------------------------------------------------ */
sk_fix sk_fix_from_int(sk_i32 v);
sk_i32 sk_fix_to_int(sk_fix v);
sk_fix sk_fix_mul(sk_fix a, sk_fix b);
sk_fix sk_fix_div(sk_fix a, sk_fix b);
sk_fix sk_fix_sqrt(sk_fix a);
sk_fix sk_fix_exp(sk_fix x);
sk_fix sk_fix_tanh(sk_fix x);
sk_fix sk_fix_abs(sk_fix a);

/* ---- PRNG: xoshiro128** -----------------------------------------------
   Seeded from the Prime Clock state, never from the wall clock, so a
   run is reproducible from a checkpoint. */
typedef struct {
    sk_u32 s[4];
} sk_rng;

void   sk_rng_seed(sk_rng *r, sk_u32 a, sk_u32 b, sk_u32 c, sk_u32 d);
sk_u32 sk_rng_next(sk_rng *r);
sk_fix sk_rng_fix(sk_rng *r);          /* uniform in [0, 1)            */
sk_i32 sk_rng_range(sk_rng *r, sk_i32 lo, sk_i32 hi);

/* Logistic map x <- r*x*(1-x).  Used as the exploration temperature
   instead of rand(): it is chaotic but fully determined by (x, r). */
sk_fix sk_logistic(sk_fix x, sk_fix r);

/* ---- CRC32C (Castagnoli) ---------------------------------------------- */
void   sk_crc_init(void);
sk_u32 sk_crc32c(const void *data, sk_u32 len);
/* Chainable form: pass the previous result, or 0 to start. */
sk_u32 sk_crc32c_update(sk_u32 crc, const void *data, sk_u32 len);

#endif /* SK_MATH_H */
