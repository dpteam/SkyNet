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

/* ---- 64-bit accumulator as two 32-bit halves --------------------------
   A dot product of 96 int8 weights against Q16.16 activations needs up to
   38 bits, and C89 has no portable 64-bit type here (long is 32 bits on
   DOS, long long is banned).  The sum is kept as hi:lo in two's complement;
   sk_acc_add() handles carry and sign extension, sk_acc_result() saturates
   on overflow instead of wrapping.  Shared by the tensor and forward
   passes, which is why it lives here and not in either of them. */
void   sk_acc_add(sk_u32 *hi, sk_u32 *lo, sk_i32 v);
sk_fix sk_acc_result(sk_u32 hi, sk_u32 lo);

/* y = ((hi:lo) * scale) >> SK_SCALE_BITS, saturating.
 *
 * `scale` carries SK_SCALE_BITS (24) fractional bits, not 16, because the
 * row scales are the dominant source of error in the fixed-point model: a
 * scale near 0.004 keeps only 16 significant bits at Q16.16, which is a
 * 0.4% systematic error on every weight of that row.
 *
 * Why the 64-bit accumulator: activations in Tiny-2.1 reach about 2900, so
 * a 96-wide dot product of int8 weights against them needs around 35 bits
 * before the scale brings it back down. Accumulating in 32 bits overflows
 * in the middle of a perfectly valid computation. */
sk_fix sk_acc_scale(sk_u32 hi, sk_u32 lo, sk_fix scale);

/* One value times a Q8.24 scale, returning Q16.16.  Sign-extends into the
   64-bit accumulator so a negative value is not read as a huge unsigned
   one - the same trap that made sk_acc_scale wrong for half its inputs. */
sk_fix sk_scale_fix(sk_fix v, sk_fix scale24);

/* ---- IEEE-754 binary32 -> Q16.16 ---------------------------------------
   Model files store their row scales as 4-byte floats, and the core has no
   floating point at all (invariant I1).  So the bit pattern is converted
   arithmetically:

     float32 = (-1)^s * 1.m * 2^(e-127)     for normal numbers
     mantissa m is the 24-bit integer (implicit 1 included)
     value * 2^16 = m * 2^(e-127-23+16) = m * 2^(e-134)

   One integer shift, exact for every value that fits Q16.16, saturating
   outside it.  No union type-punning, no compiler float support, no
   dependence on the host's FPU rounding mode.

   Subnormals and the non-finite patterns are not needed by the model
   format and are clamped: a scale of zero would make a whole row dead
   anyway, and NaN/Inf in a weight file is corruption we want to see as a
   saturation rather than as undefined behaviour. */
sk_fix sk_f32_to_fix(sk_u32 bits);

/* Same conversion with a caller-chosen number of fractional bits.
 *
 * The row scales are the one place where Q16.16 is not enough: a scale of
 * 0.0039 stored with 16 fractional bits is 255/65536, a 0.4% error on
 * every weight in that row. That error is systematic, so it shows up as a
 * 1% drift in the final scores and occasionally flips the chosen letter.
 * With 24 fractional bits the same scale is 65431/2^24 and the error drops
 * to 0.00002%. */
sk_i32 sk_f32_to_q(sk_u32 bits, int frac_bits);

/* Fractional bits used for the weight-table scales. */
#define SK_SCALE_BITS 24

#endif /* SK_MATH_H */
