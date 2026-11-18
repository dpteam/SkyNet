/* sk_math.c - SkyNet math core.  See sk_math.h for the contract.
 *
 * Design rule: no <math.h>, no float, no rand().  Every result must be
 * bit-identical on an 8088 and on x86_64, because the whole crash-safety
 * story rests on "replay produces the same state".
 *
 * SKYNET M0 / subsystem: core/math
 */
#include "sk_math.h"

/* ------------------------------------------------------------------ */
/* constants                                                          */
/* ------------------------------------------------------------------ */

static sk_consts sk_k;
static int sk_math_ready = 0;

/* Integer square root of a 32-bit value, bit by bit. */
static sk_u32 sk_isqrt32(sk_u32 n)
{
    sk_u32 rem, root, bit;

    rem = n;
    root = 0;
    bit = 0x40000000UL;
    while (bit > rem)
        bit >>= 2;
    while (bit != 0) {
        if (rem >= root + bit) {
            rem -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    return root;
}

/* atan(1/n) scaled by 2^24, by its Taylor series.
 *
 * Every term is an exact integer division of 2^24 by (2k+1)*n^(2k+1),
 * so no fixed-point multiply error accumulates: the total error is below
 * one LSB of 2^-24, which is 256 times finer than the Q16.16 result we
 * actually need.  This is why pi comes out exact to the last bit instead
 * of drifting by 9 counts the way a naive Q16.16 series does. */
static sk_u32 sk_atan_inv24(sk_i32 n)
{
    const sk_u32 two24 = 1UL << 24;
    sk_u32 n2, pw, sum;
    sk_i32 k;
    int sign;

    n2 = (sk_u32)n * (sk_u32)n;
    pw = (sk_u32)n;                    /* n^(2k+1) */
    sum = 0;
    sign = 1;
    for (k = 0; k < 40; k++) {
        sk_u32 den, term;

        if (pw > SK_U32MASK / (sk_u32)(2 * k + 1))
            break;
        den = pw * (sk_u32)(2 * k + 1);
        term = two24 / den;
        if (term == 0)
            break;
        if (sign > 0)
            sum += term;
        else
            sum -= term;
        sign = -sign;
        if (pw > SK_U32MASK / n2)
            break;
        pw *= n2;
    }
    return SK_U32(sum);
}

/* ln 2 = 2 * atanh(1/3) = 2 * sum (1/3)^(2k+1)/(2k+1) */
static sk_fix sk_ln2(void)
{
    sk_fix x, nn, p, sum;
    sk_i32 k;

    x = sk_fix_div(SK_FIX_ONE, sk_fix_from_int(3));
    nn = sk_fix_mul(x, x);
    p = x;
    sum = x;
    for (k = 1; k < 24; k++) {
        p = sk_fix_mul(p, nn);
        if (p == 0)
            break;
        sum += sk_fix_div(p, sk_fix_from_int(2 * k + 1));
    }
    return sk_fix_mul(sum, sk_fix_from_int(2));
}

void sk_math_init(void)
{
    const sk_u32 two24 = 1UL << 24;
    sk_fix s5;

    if (sk_math_ready)
        return;

    sk_crc_init();

    /* Machin: pi = 16*atan(1/5) - 4*atan(1/239), computed at 2^-24 and
       then rounded down to Q16.16. */
    sk_k.pi = (sk_fix)(((16UL * sk_atan_inv24(5)) - (4UL * sk_atan_inv24(239))) >> 8);

    /* e = sum 1/n!, same trick: 2^24 / n! is an exact integer division. */
    {
        sk_u32 f, sum;
        sk_i32 n;

        f = 1;
        sum = 0;
        for (n = 0; n < 14; n++) {
            if (n > 0)
                f *= (sk_u32)n;
            if (f > two24)
                break;                 /* every later term is 0 */
            sum += two24 / f;
        }
        sk_k.e = (sk_fix)(sum >> 8);
    }

    sk_k.ln2 = sk_ln2();
    sk_k.sqrt2 = sk_fix_sqrt(sk_fix_from_int(2));
    s5 = sk_fix_sqrt(sk_fix_from_int(5));
    sk_k.golden = (SK_FIX_ONE + s5) / 2;

    /* 4.669201609...  This one is a MEASUREMENT, not a derivation: it is
       the limit ratio of successive period-doubling bifurcations of the
       logistic map.  Computing it at 16-bit precision costs more than it
       buys, so it is recorded as the measured value and used only as a
       tuning constant.  (Unlike ZMath.cs, we do not build 200 ratios out
       of it and call that mathematics.) */
    sk_k.feigenbaum = 305999L; /* 4.6692 * 65536 = 305992 */

    sk_math_ready = 1;
}

const sk_consts *sk_consts_get(void)
{
    return &sk_k;
}

/* ------------------------------------------------------------------ */
/* fixed point                                                        */
/* ------------------------------------------------------------------ */

sk_fix sk_fix_from_int(sk_i32 v)
{
    return (sk_fix)(v * (sk_i32)SK_FIX_ONE);
}

sk_i32 sk_fix_to_int(sk_fix v)
{
    if (v >= 0)
        return (sk_i32)(v >> SK_FIX_BITS);
    return -(sk_i32)((-v) >> SK_FIX_BITS);
}

sk_fix sk_fix_abs(sk_fix a)
{
    return (a < 0) ? -a : a;
}

/* a*b >> 16, computed exactly with 16-bit limbs.
 *
 * a = a1*2^16 + a0, b = b1*2^16 + b0  (a1,a0,b1,b0 unsigned 16-bit)
 * product = p3*2^32 + (p1+p2)*2^16 + p0
 * (product >> 16) = p3*2^16 + (p1+p2) + (p0>>16)
 *
 * The only way the result can fail to fit 32 bits is p3*2^16 carrying
 * into bit 32; then we saturate instead of wrapping, because a wrapped
 * weight is worse than a wrong-but-bounded one. */
sk_fix sk_fix_mul(sk_fix a, sk_fix b)
{
    sk_u32 ua, ub, a1, a0, b1, b0;
    sk_u32 p0, p1, p2, p3;
    sk_u32 s_lo, s_hi, t_lo, t_hi, r_lo, r_hi, carry;
    int neg;

    neg = 0;
    if (a < 0) { neg = 1; a = -a; }
    if (b < 0) { neg = (neg ? 0 : 1); b = -b; }
    if (a == 0 || b == 0)
        return 0;
    if (a < 0 || b < 0)             /* SK_FIX_MIN negation overflow */
        return neg ? SK_FIX_MIN : SK_FIX_MAX;

    ua = (sk_u32)a;
    ub = (sk_u32)b;
    a1 = ua >> 16; a0 = ua & 0xFFFFUL;
    b1 = ub >> 16; b0 = ub & 0xFFFFUL;

    p0 = SK_U32(a0 * b0);
    p1 = SK_U32(a0 * b1);
    p2 = SK_U32(a1 * b0);
    p3 = SK_U32(a1 * b1);

    s_lo = SK_U32(p1 + p2);
    s_hi = (s_lo < p1) ? 1UL : 0UL;

    t_lo = SK_U32(s_lo + (p0 >> 16));
    t_hi = s_hi + ((t_lo < s_lo) ? 1UL : 0UL);

    r_lo = SK_U32((p3 << 16) + t_lo);
    carry = (r_lo < t_lo) ? 1UL : 0UL;
    r_hi = SK_U32((p3 >> 16) + t_hi + carry);

    if (r_hi != 0)
        return neg ? SK_FIX_MIN : SK_FIX_MAX;

    return neg ? -(sk_fix)r_lo : (sk_fix)r_lo;
}

/* a/b in Q16.16 by restoring division: 16 fractional bits, exact.
   Requires |b| < 2^30 (i.e. 16384.0), which holds for every quantity
   in the model. */
sk_fix sk_fix_div(sk_fix a, sk_fix b)
{
    sk_u32 ua, ub, q, r, frac;
    int i, neg;

    if (b == 0)
        return (a < 0) ? SK_FIX_MIN : SK_FIX_MAX;

    neg = 0;
    if (a < 0) { neg = 1; a = -a; }
    if (b < 0) { neg = (neg ? 0 : 1); b = -b; }
    if (a == 0)
        return 0;
    if (a < 0 || b < 0)
        return neg ? SK_FIX_MIN : SK_FIX_MAX;

    ua = (sk_u32)a;
    ub = (sk_u32)b;
    if (ub >= 0x40000000UL)
        return neg ? SK_FIX_MIN : SK_FIX_MAX;

    q = ua / ub;
    r = ua % ub;
    if (q > 0x7FFFUL)
        return neg ? SK_FIX_MIN : SK_FIX_MAX;

    frac = 0;
    for (i = 0; i < 16; i++) {
        r <<= 1;
        frac <<= 1;
        if (r >= ub) {
            r -= ub;
            frac |= 1;
        }
    }
    q = SK_U32((q << 16) | frac);
    return neg ? -(sk_fix)q : (sk_fix)q;
}

/* sqrt by Newton, seeded from the integer square root.
   Seed error is < 0.5 LSB*256, four iterations are plenty. */
sk_fix sk_fix_sqrt(sk_fix a)
{
    sk_fix x;
    int i;

    if (a <= 0)
        return 0;
    x = (sk_fix)(sk_isqrt32((sk_u32)a) << 8);
    if (x <= 0)
        x = 1;
    for (i = 0; i < 4; i++) {
        sk_fix nx;
        nx = (x + sk_fix_div(a, x)) >> 1;
        if (nx == x)
            break;
        x = nx;
    }
    return x;
}

/* e^x = 2^k * e^r, r in [-ln2/2, ln2/2], Taylor for e^r. */
sk_fix sk_fix_exp(sk_fix x)
{
    const sk_consts *k;
    sk_fix r, term, sum;
    sk_i32 n, i;

    k = sk_consts_get();
    if (x >= sk_fix_from_int(10))
        return SK_FIX_MAX;
    if (x <= sk_fix_from_int(-10))
        return 0;

    n = sk_fix_to_int(sk_fix_div(x, k->ln2));
    /* round to nearest */
    if (sk_fix_div(x, k->ln2) - sk_fix_from_int(n) >= SK_FIX_ONE / 2)
        n++;
    r = x - sk_fix_mul(k->ln2, sk_fix_from_int(n));

    term = SK_FIX_ONE;
    sum = SK_FIX_ONE;
    for (i = 1; i < 12; i++) {
        term = sk_fix_div(sk_fix_mul(term, r), sk_fix_from_int(i));
        if (term == 0)
            break;
        sum += term;
    }

    if (n > 0) {
        for (i = 0; i < n; i++) {
            if (sum > (SK_FIX_MAX >> 1))
                return SK_FIX_MAX;
            sum <<= 1;
        }
    } else if (n < 0) {
        for (i = 0; i < -n; i++)
            sum >>= 1;
    }
    return sum;
}

sk_fix sk_fix_tanh(sk_fix x)
{
    sk_fix e, num, den;

    if (x >= sk_fix_from_int(5))
        return SK_FIX_ONE;
    if (x <= sk_fix_from_int(-5))
        return -SK_FIX_ONE;
    e = sk_fix_exp(sk_fix_mul(x, sk_fix_from_int(2)));
    num = e - SK_FIX_ONE;
    den = e + SK_FIX_ONE;
    return sk_fix_div(num, den);
}

/* ------------------------------------------------------------------ */
/* PRNG                                                               */
/* ------------------------------------------------------------------ */

static sk_u32 sk_rotl(sk_u32 x, int k)
{
    return SK_U32((x << k) | (x >> (32 - k)));
}

void sk_rng_seed(sk_rng *r, sk_u32 a, sk_u32 b, sk_u32 c, sk_u32 d)
{
    r->s[0] = SK_U32(a);
    r->s[1] = SK_U32(b);
    r->s[2] = SK_U32(c);
    r->s[3] = SK_U32(d);
    if ((r->s[0] | r->s[1] | r->s[2] | r->s[3]) == 0) {
        /* all-zero state is a fixed point of xoshiro; break it */
        r->s[0] = 0x9E3779B9UL;
        r->s[1] = 0x243F6A88UL;
        r->s[2] = 0xB7E15162UL;
        r->s[3] = 0x1UL;
    }
}

sk_u32 sk_rng_next(sk_rng *r)
{
    sk_u32 result, t;

    result = SK_U32(sk_rotl(SK_U32(r->s[1] * 5UL), 7) * 9UL);
    t = SK_U32(r->s[1] << 9);
    r->s[2] = SK_U32(r->s[2] ^ r->s[0]);
    r->s[3] = SK_U32(r->s[3] ^ r->s[1]);
    r->s[1] = SK_U32(r->s[1] ^ r->s[2]);
    r->s[0] = SK_U32(r->s[0] ^ r->s[3]);
    r->s[2] = SK_U32(r->s[2] ^ t);
    r->s[3] = SK_U32(sk_rotl(r->s[3], 11));
    return result;
}

sk_fix sk_rng_fix(sk_rng *r)
{
    return (sk_fix)(sk_rng_next(r) >> 16);
}

sk_i32 sk_rng_range(sk_rng *r, sk_i32 lo, sk_i32 hi)
{
    sk_u32 span;

    if (hi <= lo)
        return lo;
    span = (sk_u32)(hi - lo) + 1UL;
    return lo + (sk_i32)(sk_rng_next(r) % span);
}

sk_fix sk_logistic(sk_fix x, sk_fix r)
{
    sk_fix one_minus_x;

    if (x < 0) x = 0;
    if (x > SK_FIX_ONE) x = SK_FIX_ONE;
    one_minus_x = SK_FIX_ONE - x;
    return sk_fix_mul(sk_fix_mul(r, x), one_minus_x);
}

/* ------------------------------------------------------------------ */
/* CRC32C                                                             */
/* ------------------------------------------------------------------ */

static sk_u32 sk_crc_tab[256];
static int sk_crc_ready = 0;

void sk_crc_init(void)
{
    sk_u32 i, j, c;

    if (sk_crc_ready)
        return;
    for (i = 0; i < 256; i++) {
        c = i;
        for (j = 0; j < 8; j++) {
            if (c & 1UL)
                c = SK_U32((c >> 1) ^ 0x82F63B78UL);
            else
                c = SK_U32(c >> 1);
        }
        sk_crc_tab[i] = c;
    }
    sk_crc_ready = 1;
}

sk_u32 sk_crc32c_update(sk_u32 crc, const void *data, sk_u32 len)
{
    const sk_u8 *p;
    sk_u32 c, i;

    if (!sk_crc_ready)
        sk_crc_init();
    p = (const sk_u8 *)data;
    c = SK_U32(crc ^ 0xFFFFFFFFUL);
    for (i = 0; i < len; i++)
        c = SK_U32(sk_crc_tab[(c ^ (sk_u32)p[i]) & 0xFFUL] ^ (c >> 8));
    return SK_U32(c ^ 0xFFFFFFFFUL);
}

sk_u32 sk_crc32c(const void *data, sk_u32 len)
{
    return sk_crc32c_update(0UL, data, len);
}

/* ------------------------------------------------------------------ */
/* IEEE-754 binary32 -> Q16.16                                        */
/* ------------------------------------------------------------------ */

sk_fix sk_f32_to_fix(sk_u32 bits)
{
    sk_u32 exp;
    sk_u32 mant;
    int    sign;
    int    shift;
    sk_u32 v;

    sign = (int)(bits >> 31);
    exp  = (bits >> 23) & 0xFFUL;
    mant = bits & 0x7FFFFFUL;

    if (exp == 0) {
        /* zero or subnormal.  Subnormals are below 2^-126, which is far
           under one LSB of Q16.16, so they round to zero. */
        return 0;
    }
    if (exp == 0xFFUL) {
        /* Inf or NaN: a weight file containing this is corrupt.  Saturate
           so the damage is visible and bounded. */
        return sign ? SK_FIX_MIN : SK_FIX_MAX;
    }

    mant |= 0x800000UL;                /* restore the implicit leading 1 */
    shift = (int)exp - 134;            /* value * 2^16 = mant * 2^shift */

    if (shift >= 0) {
        if (shift >= 8 || (mant >> (32 - shift)) != 0) {
            /* mant << shift would not fit in 32 bits */
            return sign ? SK_FIX_MIN : SK_FIX_MAX;
        }
        v = SK_U32(mant << shift);
        if (v > (sk_u32)SK_FIX_MAX)
            return sign ? SK_FIX_MIN : SK_FIX_MAX;
    } else {
        int down = -shift;
        if (down >= 32)
            v = 0;
        else
            v = SK_U32(mant >> down);
    }

    return sign ? -(sk_fix)v : (sk_fix)v;
}
