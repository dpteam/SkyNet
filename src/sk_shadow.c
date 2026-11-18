/* sk_shadow.c - the weights as trainable Q8.24 masters.  See sk_shadow.h.
 *
 * SKYNET M4 / subsystem: learn/shadow
 */
#include <stdio.h>
#include "sk_shadow.h"
#include "sk_math.h"
#include "sk_grad.h"

sk_u32 sk_shadow_bytes(const sk_table *t)
{
    if (t == SK_NULL)
        return 0;
    return t->rows * t->cols * 4UL;
}

int sk_shadow_load(const sk_table *t, sk_i32 *w)
{
    signed char code[SK_T_MAXHID];
    sk_u32 r, j;
    sk_fix scale;

    if (t == SK_NULL || w == SK_NULL || t->pool == SK_NULL)
        return 0;

    for (r = 0; r < t->rows; r++) {
        if (!sk_table_scale_read(t, r, &scale))
            return 0;
        if (!sk_table_read_row(t, r, code))
            return 0;

        if (scale < 0) {
            /* A negative scale flips a whole row. It should never occur -
               export() derives it from a magnitude - but reading one as if
               it were positive would turn the row inside out, so refuse
               rather than guess. */
            printf("[shadow] row %lu has a negative scale\n", (unsigned long)r);
            return 0;
        }

        /* code * scale24 is exact: no shift, no rounding. The only failure
           is overflow, and 127 * 2^24 is already close enough to INT32_MAX
           that a scale above 1.0 would not fit. */
        if (scale > (SK_FIX_MAX / 127)) {
            printf("[shadow] row %lu scale %ld is too large to master\n",
                   (unsigned long)r, (long)scale);
            return 0;
        }

        for (j = 0; j < t->cols; j++)
            w[r * t->cols + j] = (sk_i32)code[j] * scale;
    }
    return 1;
}

/* Round to nearest, symmetric about zero. Written out rather than using a
 * division because C89's / truncates toward zero and the asymmetry is
 * exactly the kind of thing that makes a round trip drift. */
static sk_i32 sk_round_div(sk_i32 num, sk_i32 den)
{
    if (den <= 0)
        return 0;
    if (num >= 0)
        return (num + den / 2) / den;
    return -((-num + den / 2) / den);
}

int sk_shadow_store(const sk_table *t, const sk_i32 *w)
{
    signed char code[SK_T_MAXHID];
    sk_u32 r, j;
    sk_i32 maxabs;
    sk_fix scale;

    if (t == SK_NULL || w == SK_NULL || t->pool == SK_NULL)
        return 0;

    for (r = 0; r < t->rows; r++) {
        const sk_i32 *row = w + r * t->cols;

        maxabs = 0;
        for (j = 0; j < t->cols; j++) {
            sk_i32 a = (row[j] < 0) ? -row[j] : row[j];

            if (a > maxabs)
                maxabs = a;
        }

        /* A row that trained its way to zero, or a row that started there.
           Zero scale and zero codes is the honest representation: keeping
           the old scale would leave a row that means nothing claiming to
           have a magnitude. */
        if (maxabs == 0) {
            for (j = 0; j < t->cols; j++)
                code[j] = 0;
            if (!sk_table_scale_write(t, r, 0))
                return 0;
            if (!sk_table_write_row(t, r, code))
                return 0;
            continue;
        }

        /* scale = max/127, exactly as export() computes it. Integer division
           here is what makes the round trip exact: a row whose maximum is
           127*scale24 divides back to scale24 with no remainder. */
        scale = maxabs / 127;
        if (scale < 1)
            scale = 1;          /* one LSB; below that codes cannot move */

        for (j = 0; j < t->cols; j++) {
            sk_i32 c = sk_round_div(row[j], scale);

            if (c > 127)
                c = 127;
            if (c < -127)
                c = -127;
            code[j] = (signed char)c;
        }

        if (!sk_table_scale_write(t, r, scale))
            return 0;
        if (!sk_table_write_row(t, r, code))
            return 0;
    }
    return 1;
}

int sk_shadow_step(sk_i32 *w, const sk_fix *grad, sk_i32 lr24, sk_u32 n)
{
    sk_u32 i;
    int changed = 0;

    if (w == SK_NULL || grad == SK_NULL || lr24 == 0)
        return 0;

    for (i = 0; i < n; i++) {
        sk_i32 before = w[i];

        /* lr24 * grad16 -> Q8.24 step. sk_g24_mul is a signed 32x32 -> 64
           product shifted by 24, so the answer saturates instead of wrapping
           when a gradient is absurd. */
        w[i] = (sk_i32)(w[i] - sk_g24_mul(sk_g24_from16(grad[i]), lr24));
        if (w[i] != before)
            changed++;
    }
    return changed;
}

sk_fix sk_shadow_max(const sk_i32 *w, sk_u32 n)
{
    sk_u32 i;
    sk_i32 mx = 0;

    if (w == SK_NULL)
        return 0;
    for (i = 0; i < n; i++) {
        sk_i32 a = (w[i] < 0) ? -w[i] : w[i];

        if (a > mx)
            mx = a;
    }
    /* Q8.24 to Q16.16 by shifting, which is exact in the direction that
       matters (dropping eight bits, not inventing eight). */
    return mx >> 8;
}
