/* sk_grad.c - fixed-point arithmetic for gradients.  See sk_grad.h.
 *
 * SKYNET M4 / subsystem: learn/grad
 */
#include <stdio.h>
#include "sk_grad.h"
#include "sk_math.h"

/* Signed 32x32 -> 64 bit product shifted right by `shift`, saturating.
 *
 * Written out by hand because C89 has no portable 64-bit type here: long is
 * 32 bits on DOS, and long long is banned by the C89 rule. The limbs are
 * the same trick sk_fix_mul uses, extended to a full 64-bit result so the
 * shift can be larger than 16. */
static sk_i32 sk_mul_shift(sk_i32 a, sk_i32 b, int shift)
{
    sk_u32 ua, ub, a1, a0, b1, b0;
    sk_u32 p0, p1, p2, p3;
    sk_u32 mid, lo, hi;
    int neg;

    if (a == 0 || b == 0)
        return 0;

    neg = 0;
    if (a < 0) { neg = 1; a = -a; }
    if (b < 0) { neg = !neg; b = -b; }
    if (a < 0 || b < 0)             /* negation of the minimum value */
        return neg ? SK_FIX_MIN : SK_FIX_MAX;

    ua = (sk_u32)a;
    ub = (sk_u32)b;
    a1 = ua >> 16; a0 = ua & 0xFFFFUL;
    b1 = ub >> 16; b0 = ub & 0xFFFFUL;

    p0 = SK_U32(a0 * b0);
    p1 = SK_U32(a0 * b1);
    p2 = SK_U32(a1 * b0);
    p3 = SK_U32(a1 * b1);

    /* lo = p0 + ((p1 + p2) << 16) */
    mid = SK_U32((p0 >> 16) + (p1 & 0xFFFFUL) + (p2 & 0xFFFFUL));
    lo  = SK_U32((p0 & 0xFFFFUL) | (mid << 16));
    /* hi = p3 + (p1 >> 16) + (p2 >> 16) + (mid >> 16) */
    hi  = SK_U32(p3 + (p1 >> 16) + (p2 >> 16) + (mid >> 16));

    /* shift the 64-bit value right */
    if (shift <= 0) {
        if (hi != 0UL)
            return neg ? SK_FIX_MIN : SK_FIX_MAX;
        return neg ? -(sk_i32)lo : (sk_i32)lo;
    }
    if (shift >= 64) {
        lo = 0;
        hi = 0;
    } else if (shift >= 32) {
        int s2 = shift - 32;
        lo = (s2 == 0) ? hi : SK_U32(hi >> s2);
        hi = 0;
    } else {
        lo = SK_U32((lo >> shift) | (hi << (32 - shift)));
        hi = SK_U32(hi >> shift);
    }

    if (hi != 0UL)
        return neg ? SK_FIX_MIN : SK_FIX_MAX;
    if (lo > (sk_u32)SK_FIX_MAX)
        return neg ? SK_FIX_MIN : SK_FIX_MAX;

    return neg ? -(sk_i32)lo : (sk_i32)lo;
}

sk_i32 sk_g24_mul(sk_i32 a, sk_i32 b)
{
    return sk_mul_shift(a, b, 24);
}

sk_i32 sk_g24_from16(sk_fix v)
{
    /* v * 2^8 */
    if (v > (SK_FIX_MAX >> 8))
        return SK_FIX_MAX;
    if (v < (SK_FIX_MIN >> 8))
        return SK_FIX_MIN;
    return v << 8;
}

sk_fix sk_g24_to16(sk_i32 g)
{
    return g >> 8;
}

void sk_grad_dx_add(sk_fix *dx, sk_i32 dy24, sk_i32 scale24, sk_i32 w)
{
    sk_i32 t;

    if (dx == SK_NULL || w == 0)
        return;

    t = sk_g24_mul(dy24, scale24);      /* D*S*2^24 */
    if (t == 0)
        return;
    /* t * w can overflow 32 bits when both are large; the gradient values
       in practice keep |t| below 2^24, but the guard makes the failure
       explicit instead of silent. */
    if (t > (SK_FIX_MAX / (w > 0 ? w : -w)))
        t = (w > 0) ? (SK_FIX_MAX / w) : (SK_FIX_MAX / -w);
    if (t < (SK_FIX_MIN / (w > 0 ? w : -w)))
        t = (w > 0) ? (SK_FIX_MIN / w) : (SK_FIX_MIN / -w);

    *dx += (sk_fix)((t * w) >> 8);
}

sk_i32 sk_g24_apply(sk_i32 scale24, sk_i32 grad24, sk_i32 lr24)
{
    sk_i32 delta;

    if (lr24 == 0 || grad24 == 0)
        return scale24;

    /* Both operands are Q8.24, so the product shifted by 24 is Q8.24.
       The learning rate has to live in Q8.24 and not Q16.16: the gradient
       of a row scale is dL/dlogits * dot, and with dot around 600 a useful
       step needs a rate near 1e-5, which is zero in Q16.16. Using a
       Q16.16 rate there gave a step 15x the scale itself and the loss went
       up on every epoch - which is exactly what the first run did. */
    delta = sk_g24_mul(grad24, lr24);

    /* Clamp: one saturated gradient must not move a row scale by more than
       an eighth of itself, or the row goes dead and never recovers. */
    if (delta > (SK_FIX_ONE / 8))
        delta = SK_FIX_ONE / 8;
    if (delta < -(SK_FIX_ONE / 8))
        delta = -(SK_FIX_ONE / 8);

    scale24 -= delta;
    if (scale24 < 0)
        scale24 = 0;
    return scale24;
}

/* ------------------------------------------------------------------ */
/* selftest                                                           */
/* ------------------------------------------------------------------ */

int sk_grad_selftest(void)
{
    int ok;
    sk_i32 v;
    sk_fix dx;

    ok = 1;

    /* 1.0 * 1.0 = 1.0 in Q8.24 */
    v = sk_g24_mul(16777216L, 16777216L);
    if (v != 16777216L) {
        printf("  FAIL: g24 1.0*1.0 = %ld\n", (long)v);
        ok = 0;
    }

    /* 0.5 * 0.5 = 0.25 */
    v = sk_g24_mul(8388608L, 8388608L);
    if (v < 4194300L || v > 4194308L) {
        printf("  FAIL: g24 0.5*0.5 = %ld\n", (long)v);
        ok = 0;
    }

    /* negative * positive keeps the sign */
    v = sk_g24_mul(-8388608L, 8388608L);
    if (v > -4194300L || v < -4194308L) {
        printf("  FAIL: g24 -0.5*0.5 = %ld\n", (long)v);
        ok = 0;
    }

    /* conversion round trip */
    v = sk_g24_from16(SK_FIX_ONE);
    if (v != 16777216L) {
        printf("  FAIL: from16 1.0 = %ld\n", (long)v);
        ok = 0;
    }
    if (sk_g24_to16(v) != SK_FIX_ONE) {
        printf("  FAIL: to16 round trip\n");
        ok = 0;
    }

    /* dx += dy*scale*w: dy=0.001, scale=0.004, w=127
       expected dx = 0.001*0.004*127 = 5.08e-4, in Q16.16 = 33 */
    dx = 0;
    sk_grad_dx_add(&dx, 16777L, 67108L, 127);
    if (dx < 30 || dx > 36) {
        printf("  FAIL: grad_dx = %ld, expected ~33\n", (long)dx);
        ok = 0;
    }

    /* The update actually moves the scale in the right direction.
       grad = 1.0 in Q8.24 and lr = 168 (1e-5), so delta = 168, which is
       visible against a scale of 67108. */
    v = sk_g24_apply(67108L, 16777216L, 168L);
    if (v != 67108L - 168L) {
        printf("  FAIL: apply gave %ld, expected %ld\n",
               (long)v, (long)(67108L - 168L));
        ok = 0;
    }

    /* and a huge gradient is clamped rather than destroying the row */
    v = sk_g24_apply(67108L, SK_FIX_MAX, 168L);
    if (v < 0 || v < 67108L - (SK_FIX_ONE / 8) * 2) {
        printf("  FAIL: apply did not clamp\n");
        ok = 0;
    }

    printf("  grad: Q8.24 arithmetic %s\n", ok ? "ok" : "FAILED");
    return ok;
}
