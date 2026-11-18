/* sk_forward.c - the Tiny-2.1 forward pass in Q16.16.
 *
 * SKYNET M2b / subsystem: model/forward
 */
#include <string.h>
#include <stdio.h>
#include "sk_forward.h"
#include "sk_math.h"

/* Work buffers.  Static, not automatic: on DOS the stack is 8 KB and these
 * come to about 3.5 KB on their own, on top of every caller frame.  TINY.C
 * makes the same choice for the same reason. */
static sk_fix bx[SK_T_MAXD];
static sk_fix bh[SK_T_MAXD];
static sk_fix bq[SK_T_MAXD];
static sk_fix bt[SK_T_MAXD];
static sk_fix ba[SK_T_MAXD];
static sk_fix bm[SK_T_MAXHID];
static sk_fix bs[SK_T_CTX];

/* ---- KV cache layout ------------------------------------------------
   Per layer, all inside the pool:
     base                       : CTX * D bytes, int8 keys
     base + CTX*D               : CTX * D bytes, int8 values
     base + 2*CTX*D             : CTX * 4 bytes, Q16.16 key scales
     base + 2*CTX*D + CTX*4     : CTX * 4 bytes, Q16.16 value scales
   Keeping the scales in the pool rather than in the context keeps sk_ctx
   small enough to live on the DOS stack. */
static sk_u32 kv_layer_bytes(const sk_model *m)
{
    sk_u32 cd = (sk_u32)SK_T_CTX * (sk_u32)m->D;
    return 2UL * cd + 8UL * (sk_u32)SK_T_CTX;
}

static sk_u32 kv_layer_base(const sk_model *m, int layer)
{
    return (sk_u32)layer * kv_layer_bytes(m);
}

sk_u32 sk_ctx_kv_bytes(const sk_model *m)
{
    if (m == SK_NULL)
        return 0;
    return (sk_u32)m->L * kv_layer_bytes(m);
}

int sk_ctx_open(sk_ctx *c, sk_model *m, void *kv_storage, sk_u32 kv_size)
{
    sk_u32 need;

    if (c == SK_NULL || m == SK_NULL || !m->loaded)
        return 0;
    memset(c, 0, sizeof(*c));
    c->m = m;
    need = sk_ctx_kv_bytes(m);

    /* Prefer EMS: on DOS this is the only way to hold 100 KB of cache. */
    if (!sk_pool_open(&c->kv, need)) {
        if (kv_storage == SK_NULL || kv_size < need)
            return 0;
        if (!sk_pool_open_from(&c->kv, kv_size, kv_storage))
            return 0;
    }
    c->pos = 0;
    c->ready = 1;
    return 1;
}

void sk_ctx_close(sk_ctx *c)
{
    if (c == SK_NULL)
        return;
    sk_pool_close(&c->kv);
    c->ready = 0;
    c->pos = 0;
}

int sk_ctx_hidden(sk_ctx *c, sk_fix *out)
{
    int i;

    if (c == SK_NULL || out == SK_NULL || c->m == SK_NULL)
        return 0;
    for (i = 0; i < c->m->D; i++)
        out[i] = c->hf[i];
    return 1;
}

void sk_ctx_reset(sk_ctx *c)
{
    if (c != SK_NULL)
        c->pos = 0;
}

/* ---- primitives ------------------------------------------------------ */

/* y = x / sqrt(mean(x^2) + eps) * g, the RMS norm the model uses.
 *
 * The sum of squares does not fit Q16.16: activations in this model reach
 * about 2900, so the sum is ~8e6 and needs 39 bits.  Rather than carry a
 * 64-bit square root, x is shifted down until the largest element is below
 * 8, the mean is computed there, and the resulting 1/sqrt is shifted back
 * by the same amount.  The norm is scale-invariant, so this is exact up to
 * the shift's own rounding - and it keeps every intermediate in range.
 *
 * The epsilon is 1e-5 in the original; here it is one LSB of Q16.16
 * (1.5e-5), the closest representable value, doing the same job: keeping a
 * zero-variance vector from dividing by zero. */
static void sk_norm(const sk_fix *x, const sk_fix *g, sk_fix *y, int n)
{
    sk_u32 hi, lo;
    sk_fix mx, s, mean, inv, limit;
    int i, shift;

    mx = 0;
    for (i = 0; i < n; i++) {
        sk_fix a = (x[i] < 0) ? -x[i] : x[i];
        if (a > mx)
            mx = a;
    }

    /* shift until the largest element is at most 8.0 */
    limit = 8L * SK_FIX_ONE;
    shift = 0;
    while (mx > limit) {
        mx >>= 1;
        shift++;
    }

    hi = 0;
    lo = 0;
    for (i = 0; i < n; i++) {
        sk_fix v = x[i] >> shift;
        sk_acc_add(&hi, &lo, sk_fix_mul(v, v));
    }
    s = sk_acc_result(hi, lo);
    mean = sk_fix_div(s, sk_fix_from_int(n));
    mean += 1;                          /* one LSB, stands in for 1e-5 */
    if (mean <= 0) {
        for (i = 0; i < n; i++)
            y[i] = 0;
        return;
    }
    inv = sk_fix_div(SK_FIX_ONE, sk_fix_sqrt(mean));
    for (i = shift; i > 0; i--)
        inv >>= 1;

    for (i = 0; i < n; i++)
        y[i] = sk_fix_mul(sk_fix_mul(x[i], inv), g[i]);
}

/* Quantise n values to int8 with one scale, exactly the way the original
   store() does: k = 127/max, round half away from zero, scale = max/127.
   The rounding matters - truncation would drift every cached vector down
   by up to one count and the attention weights would slowly diverge. */
static sk_fix sk_quant_store(sk_pool *pool, sk_u32 off, const sk_fix *src, int n)
{
    sk_u8 buf[SK_T_MAXD];
    sk_fix big, t;
    int i;

    big = 0;
    for (i = 0; i < n; i++) {
        sk_fix a = (src[i] < 0) ? -src[i] : src[i];
        if (a > big)
            big = a;
    }
    if (big <= 0)
        big = 1;                        /* all-zero vector: scale 1/127 */

    /* Scale each element by 127/big directly, instead of precomputing
       k = 127/big and multiplying.  Precomputing looked cheaper and was
       wrong: k has to be held in a fixed-point word, and for a vector whose
       largest element is below 1 the value 127/big exceeds what Q8.24 can
       represent.  It saturated, every element then quantised to +-127, and
       the attention values for that position were destroyed - which is why
       8 of 26 single-letter prompts produced scores 50-80% off while the
       rest were within 1%.  A division per element is a few thousand
       instructions per position and cannot overflow. */
    for (i = 0; i < n; i++) {
        sk_fix num = sk_fix_mul(src[i], sk_fix_from_int(127));

        t = sk_fix_div(num, big);
        /* round half away from zero, like the original cast */
        if (t >= 0)
            t = (t + 32768L) >> 16;
        else
            t = -(((-t) + 32768L) >> 16);
        if (t > 127)
            t = 127;
        if (t < -127)
            t = -127;
        buf[i] = (sk_u8)(sk_i8)t;
    }
    if (!sk_pool_write(pool, off, buf, (sk_u32)n))
        return 0;

    /* Return the scale in the same Q8.24 form the weight tables use, so a
       single multiply path handles both. big is Q16.16, so
       (big/127) * 256 is (big/127) in Q8.24. */
    return sk_fix_mul(sk_fix_div(big, sk_fix_from_int(127)),
                      sk_fix_from_int(256));
}

static int sk_load_scale(sk_pool *pool, sk_u32 off, sk_fix *out)
{
    sk_u8 b[4];

    if (!sk_pool_read(pool, off, b, 4))
        return 0;
    *out = (sk_fix)((sk_u32)b[0] | ((sk_u32)b[1] << 8)
                  | ((sk_u32)b[2] << 16) | ((sk_u32)b[3] << 24));
    return 1;
}

static int sk_save_scale(sk_pool *pool, sk_u32 off, sk_fix v)
{
    sk_u8 b[4];

    b[0] = (sk_u8)(v & 0xFFL);
    b[1] = (sk_u8)((v >> 8) & 0xFFL);
    b[2] = (sk_u8)((v >> 16) & 0xFFL);
    b[3] = (sk_u8)((v >> 24) & 0xFFL);
    return sk_pool_write(pool, off, b, 4);
}

/* dot of an int8 row against Q16.16 activations, returned as a 64-bit pair.
   Returning a single sk_fix would overflow: 24 terms of 127 * 58 is 176784,
   which does not fit Q16.16 even though the scaled result does. */
static void sk_dot_row(const sk_u8 *row, const sk_fix *x, int n,
                       sk_u32 *hi_out, sk_u32 *lo_out)
{
    sk_u32 hi, lo;
    int i;

    hi = 0;
    lo = 0;
    for (i = 0; i < n; i++)
        sk_acc_add(&hi, &lo,
                   sk_fix_mul(sk_fix_from_int((sk_i32)(sk_i8)row[i]), x[i]));
    *hi_out = hi;
    *lo_out = lo;
}

/* ---- the step -------------------------------------------------------- */

int sk_forward_step(sk_ctx *c, int tok, sk_fix *scores)
{
    sk_model *m;
    int l, h, i, j, p;
    sk_fix es, ps;
    sk_u32 lb, kbase, vbase, ksbase, vsbase;

    if (c == SK_NULL || !c->ready || scores == SK_NULL)
        return 0;
    m = c->m;
    if (tok < 0 || tok >= m->vocab)
        return 0;
    p = c->pos;
    if (p < 0 || p >= SK_T_CTX)
        return 0;

    c->peak = 0;
    c->saturated = 0;

    /* x = emb[tok] * emb_scale + pos[p] * pos_scale */
    if (!sk_table_row(&m->emb, (sk_u32)tok, bx))
        return 0;
    if (!sk_table_row(&m->pos, (sk_u32)p, bt))
        return 0;
    for (i = 0; i < m->D; i++)
        bx[i] += bt[i];


    for (l = 0; l < m->L; l++) {
        sk_fix g1[SK_T_MAXD];
        sk_fix g2[SK_T_MAXD];

        if (!sk_model_gain_row(m, SK_GAIN_G1, l, g1)
            || !sk_model_gain_row(m, SK_GAIN_G2, l, g2))
            return 0;

        lb = kv_layer_base(m, l);
        kbase = lb;
        vbase = lb + (sk_u32)SK_T_CTX * (sk_u32)m->D;
        ksbase = vbase + (sk_u32)SK_T_CTX * (sk_u32)m->D;
        vsbase = ksbase + 4UL * (sk_u32)SK_T_CTX;

        /* ---- attention -------------------------------------------- */
        sk_norm(bx, g1, bh, m->D);


        if (!sk_table_matvec(&m->wk[l], bh, bt))
            return 0;
        es = sk_quant_store(&c->kv, kbase + (sk_u32)p * (sk_u32)m->D,
                            bt, m->D);
        if (!sk_save_scale(&c->kv, ksbase + 4UL * (sk_u32)p, es))
            return 0;

        if (!sk_table_matvec(&m->wv[l], bh, bt))
            return 0;
        ps = sk_quant_store(&c->kv, vbase + (sk_u32)p * (sk_u32)m->D,
                            bt, m->D);
        if (!sk_save_scale(&c->kv, vsbase + 4UL * (sk_u32)p, ps))
            return 0;

        if (!sk_table_matvec(&m->wq[l], bh, bq))
            return 0;

        for (h = 0; h < m->H; h++) {
            sk_fix big, sum, s;
            sk_u8 row[SK_T_MAXD];

            /* scores for this head against every earlier position */
            big = SK_FIX_MIN;
            for (j = 0; j <= p; j++) {
                sk_fix ks, kd;
                sk_u32 dh, dl;

                if (!sk_pool_read(&c->kv,
                                  kbase + (sk_u32)j * (sk_u32)m->D,
                                  row, (sk_u32)m->D))
                    return 0;
                if (!sk_load_scale(&c->kv, ksbase + 4UL * (sk_u32)j, &ks))
                    return 0;

                /* Fold the division by sqrt(HD) into the key scale before
                   the multiply: dot*ks alone is around 70000, which is past
                   Q16.16, but dot*(ks/sqrt(HD)) is around 14000 and fits. */
                kd = sk_fix_div(ks, m->att_div);
                sk_dot_row(row + h * m->HD, bq + h * m->HD, m->HD, &dh, &dl);
                s = sk_acc_scale(dh, dl, kd);

                bs[j] = s;
                if (s > big)
                    big = s;
            }

            /* softmax, with the max subtracted so exp stays in (0, 1] */
            sum = 0;
            for (j = 0; j <= p; j++) {
                bs[j] = sk_fix_exp(bs[j] - big);
                sum += bs[j];
            }
            if (sum <= 0)
                sum = 1;

            for (i = 0; i < m->HD; i++)
                ba[h * m->HD + i] = 0;

            for (j = 0; j <= p; j++) {
                sk_fix vs, w;
                sk_u8 vrow[SK_T_MAXD];

                if (bs[j] == 0)
                    continue;
                if (!sk_pool_read(&c->kv,
                                  vbase + (sk_u32)j * (sk_u32)m->D,
                                  vrow, (sk_u32)m->D))
                    return 0;
                if (!sk_load_scale(&c->kv, vsbase + 4UL * (sk_u32)j, &vs))
                    return 0;

                /* weight * value_scale / sum, then times the int8 value.
                   vs is Q8.24, so it needs sk_scale_fix, not sk_fix_mul -
                   treating it as Q16.16 made every weight 256 times too
                   large, which cancelled out only at position 0. */
                w = sk_fix_div(sk_scale_fix(bs[j], vs), sum);
                for (i = 0; i < m->HD; i++) {
                    sk_fix v = sk_fix_mul(
                        sk_fix_from_int((sk_i32)(sk_i8)vrow[h * m->HD + i]),
                        w);
                    ba[h * m->HD + i] += v;
                }
            }
        }

        if (!sk_table_matvec(&m->wo[l], ba, bt))
            return 0;

        for (i = 0; i < m->D; i++)
            bx[i] += bt[i];

        /* Track the residual magnitude.  This is the number that says
           whether Q16.16 is still adequate for this input: the model's
           activations reach about 2900 and the format tops out at 32768. */
        for (i = 0; i < m->D; i++) {
            sk_fix a = (bx[i] < 0) ? -bx[i] : bx[i];
            if (a > c->peak)
                c->peak = a;
            if (a >= SK_FIX_MAX)
                c->saturated = 1;
        }


        /* ---- feed forward ----------------------------------------- */
        sk_norm(bx, g2, bh, m->D);
        if (!sk_table_matvec(&m->f1[l], bh, bm))
            return 0;
        for (i = 0; i < m->HID; i++) {
            if (bm[i] < 0)
                bm[i] = 0;
        }
        if (!sk_table_matvec(&m->f2[l], bm, bt))
            return 0;
        for (i = 0; i < m->D; i++)
            bx[i] += bt[i];
    }

    /* ---- output projection, weight-tied to the embedding ----------- */
    {
        sk_fix gf[SK_T_MAXD];

        if (!sk_model_gain_row(m, SK_GAIN_GF, 0, gf))
            return 0;
        sk_norm(bx, gf, bh, m->D);
        memcpy(c->hf, bh, (size_t)m->D * sizeof(sk_fix));
        if (!sk_table_matvec(&m->emb, bh, scores))
            return 0;
    }

    c->pos++;
    return 1;
}

/* ------------------------------------------------------------------ */
/* decoding                                                           */
/* ------------------------------------------------------------------ */

/* Token for a character, the same mapping TINY.C uses: c - 30, i.e. the
   vocabulary is ' '..'~' starting at 2, with 0 = pad and 1 = newline. */
static int sk_tok_of(int ch)
{
    int t = ch - 30;
    if (t < 2)
        t = 2;
    if (t >= SK_T_VOCAB)
        t = SK_T_VOCAB - 1;
    return t;
}

int sk_generate(sk_ctx *c, const char *question, char *out, int outmax,
                int creativity, sk_rng *rng)
{
    sk_fix scores[SK_T_VOCAB];
    int p, n, letters;
    int tok;

    if (c == SK_NULL || question == SK_NULL || out == SK_NULL || outmax < 1)
        return -1;
    if (!c->ready)
        return -1;

    sk_ctx_reset(c);
    letters = 0;
    n = (int)strlen(question);
    if (n <= 0 || n >= SK_T_CTX - 1)
        return -1;

    /* the question */
    for (p = 0; p < n; p++) {
        if (!sk_forward_step(c, sk_tok_of((unsigned char)question[p]), scores))
            return -1;
    }
    /* token 1 is the end of the question */
    if (!sk_forward_step(c, 1, scores))
        return -1;

    while (c->pos < SK_T_CTX && letters < outmax - 1) {
        int best = 0;
        int i;

        for (i = 1; i < SK_T_VOCAB; i++) {
            if (scores[i] > scores[best])
                best = i;
        }

        if (creativity <= 0) {
            tok = best;
        } else {
            /* t = creativity/5; weight = exp((s - best)/t), and anything
               below 0.15 is dropped so the model never emits an unlikely
               letter just because the dice said so. */
            sk_fix t, sum, r, w[SK_T_VOCAB];
            sk_fix total;

            t = sk_fix_div(sk_fix_from_int(creativity), sk_fix_from_int(5));
            if (t <= 0)
                t = SK_FIX_ONE;
            sum = 0;
            for (i = 0; i < SK_T_VOCAB; i++) {
                sk_fix d = sk_fix_div(scores[i] - scores[best], t);
                sk_fix e = sk_fix_exp(d);
                if (e < 9830)               /* 0.15 in Q16.16 */
                    e = 0;
                w[i] = e;
                sum += e;
            }
            if (sum <= 0) {
                tok = best;
            } else {
                total = sum;
                r = sk_rng_range(rng, 0, 9999);
                r = sk_fix_div(sk_fix_mul(total, sk_fix_from_int((int)r)),
                               sk_fix_from_int(10000));
                tok = best;
                for (i = 0; i < SK_T_VOCAB; i++) {
                    r -= w[i];
                    if (r < 0 && w[i] > 0) {
                        tok = i;
                        break;
                    }
                }
            }
        }

        if (tok <= 1)
            break;                          /* end of the answer */

        out[letters++] = (char)(tok + 30);
        if (!sk_forward_step(c, tok, scores))
            return -1;
    }

    out[letters] = '\0';
    return letters;
}

/* ------------------------------------------------------------------ */
/* selftest                                                           */
/* ------------------------------------------------------------------ */

int sk_forward_selftest(void)
{
    /* The forward pass needs a model, so the only thing testable without
       one is the quantiser's rounding contract, which is what the cached
       vectors depend on. */
    sk_u8 storage[256];
    sk_pool p;
    sk_fix src[4];
    sk_fix scale;
    sk_u8 out[4];

    memset(storage, 0, sizeof(storage));
    if (!sk_pool_open_from(&p, sizeof(storage), storage))
        return 0;

    src[0] = sk_fix_from_int(100);
    src[1] = sk_fix_from_int(-50);
    src[2] = sk_fix_from_int(0);
    src[3] = sk_fix_from_int(25);

    scale = sk_quant_store(&p, 0, src, 4);
    if (scale == 0)
        return 0;
    if (!sk_pool_read(&p, 0, out, 4))
        return 0;

    /* max is 100, so 100 must map to 127 and -50 to -64 (round away) */
    if ((sk_i32)(sk_i8)out[0] != 127)
        return 0;
    if ((sk_i32)(sk_i8)out[1] != -64 && (sk_i32)(sk_i8)out[1] != -63)
        return 0;
    if ((sk_i32)(sk_i8)out[2] != 0)
        return 0;

    /* scale * 127 must reconstruct the maximum within one count.  The scale
       is Q8.24, so it has to go through sk_scale_fix, not sk_fix_mul. */
    {
        sk_fix back = sk_scale_fix(sk_fix_from_int(127), scale);
        sk_fix d = back - src[0];
        if (d < 0)
            d = -d;
        if (d > 700)                     /* ~0.01 in Q16.16 */
            return 0;
    }

    /* Regression for the bug that made 8 of 26 single-letter prompts wrong:
       a vector whose largest element is below 1.0 used to overflow the
       precomputed scale factor, saturate, and quantise every element to
       +-127.  The check is simply that the quantised values keep their
       proportions. */
    {
        sk_fix small[4];
        sk_u8 q[4];

        small[0] = SK_FIX_ONE / 2;      /* 0.5   */
        small[1] = -SK_FIX_ONE / 4;     /* -0.25 */
        small[2] = 0;
        small[3] = SK_FIX_ONE / 8;      /* 0.125 */

        if (sk_quant_store(&p, 64, small, 4) == 0)
            return 0;
        if (!sk_pool_read(&p, 64, q, 4))
            return 0;

        /* 0.5 is the maximum, so it must map to 127; -0.25 is half that in
           magnitude, so -64; 0.125 is a quarter, so 32.  If the scale had
           overflowed these would all be +-127. */
        if ((sk_i32)(sk_i8)q[0] != 127)
            return 0;
        if ((sk_i32)(sk_i8)q[1] != -64 && (sk_i32)(sk_i8)q[1] != -63)
            return 0;
        if ((sk_i32)(sk_i8)q[2] != 0)
            return 0;
        if ((sk_i32)(sk_i8)q[3] != 32)
            return 0;
    }

    return 1;
}
