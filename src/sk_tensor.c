/* sk_tensor.c - int8 weight tables and the TNY2 loader.
 *
 * SKYNET M2 / subsystem: model/tensor
 */
#include <stdio.h>
#include <string.h>
#include "sk_tensor.h"
#include "sk_math.h"

#define SK_T_VOCAB 97
#define SK_T_CTX   128

static sk_u32 sk_rd16(const unsigned char *b)
{
    return (sk_u32)b[0] | ((sk_u32)b[1] << 8);
}

static sk_u32 sk_rd32(const unsigned char *b)
{
    return (sk_u32)b[0] | ((sk_u32)b[1] << 8)
         | ((sk_u32)b[2] << 16) | ((sk_u32)b[3] << 24);
}

static void sk_why(char *why, sk_u32 len, const char *msg)
{
    if (why == SK_NULL || len == 0)
        return;
    strncpy(why, msg, (size_t)(len - 1));
    why[len - 1] = '\0';
}

/* Bytes for one table: rows of 4-byte scales, then rows*cols weights. */
static sk_u32 sk_table_bytes(sk_u32 rows, sk_u32 cols)
{
    return rows * 4UL + rows * cols;
}

sk_u32 sk_model_size(sk_u32 D, sk_u32 L, sk_u32 HID,
                     sk_u32 vocab, sk_u32 ctx)
{
    sk_u32 n;
    sk_u32 l;

    n = sk_table_bytes(vocab, D) + sk_table_bytes(ctx, D);
    for (l = 0; l < L; l++) {
        n += 4UL * D;                       /* g1 gains                */
        n += 4UL * sk_table_bytes(D, D);    /* wq wk wv wo             */
        n += 4UL * D;                       /* g2 gains                */
        n += sk_table_bytes(HID, D);
        n += sk_table_bytes(D, HID);
    }
    n += 4UL * D;                           /* gf gains                */
    return n;
}

/* Read one table: rows of (float32 scale, cols int8 weights).
   The scales are converted and stored first, then the weights, so a table
   occupies two contiguous blocks in the pool and the struct stays tiny. */
static int sk_load_table(sk_model *m, FILE *fp, sk_table *t,
                         sk_u32 rows, sk_u32 cols, char *why, sk_u32 whylen)
{
    unsigned char buf[4 + SK_T_MAXHID];
    sk_u32 r;
    sk_u32 need;
    sk_u32 wbase;

    if (rows > (sk_u32)SK_T_MAXHID || cols > (sk_u32)SK_T_MAXHID) {
        sk_why(why, whylen, "table larger than the program supports");
        return 0;
    }
    need = sk_table_bytes(rows, cols);
    if (m->used > m->pool.bytes || need > m->pool.bytes - m->used) {
        sk_why(why, whylen, "weights do not fit in the storage given");
        return 0;
    }

    t->pool = &m->pool;
    t->base = m->used;
    t->rows = rows;
    t->cols = cols;
    wbase = m->used + rows * 4UL;

    for (r = 0; r < rows; r++) {
        sk_fix scale;
        unsigned char sb[4];

        if (fread(buf, 1, (size_t)(4 + cols), fp) != (size_t)(4 + cols)) {
            sk_why(why, whylen, "file ends inside a weight table");
            return 0;
        }
        /* The float is converted arithmetically, not cast: the core has no
           floating point and must produce identical results on a machine
           with an FPU and one without. */
        scale = (sk_fix)sk_f32_to_q(sk_rd32(buf), SK_SCALE_BITS);
        sb[0] = (unsigned char)(scale & 0xFFL);
        sb[1] = (unsigned char)((scale >> 8) & 0xFFL);
        sb[2] = (unsigned char)((scale >> 16) & 0xFFL);
        sb[3] = (unsigned char)((scale >> 24) & 0xFFL);
        if (!sk_pool_write(&m->pool, m->used + r * 4UL, sb, 4)) {
            sk_why(why, whylen, "cannot store scales");
            return 0;
        }
        if (!sk_pool_write(&m->pool, wbase + r * cols, buf + 4, cols)) {
            sk_why(why, whylen, "cannot store weights");
            return 0;
        }
        /* One checksum over the weights as they appear in the file.  The
           verifier recomputes it from the store, so the format is parsed
           in exactly one place - a second hand-written walk of the file
           got the g2 position wrong and reported a false mismatch. */
        m->weights_crc = sk_crc32c_update(m->weights_crc, buf + 4, cols);
    }
    m->used += need;
    return 1;
}

/* Read the scale of one row. */
static int sk_table_scale(const sk_table *t, sk_u32 row, sk_fix *out)
{
    unsigned char sb[4];

    if (t == SK_NULL || out == SK_NULL || row >= t->rows)
        return 0;
    if (!sk_pool_read(t->pool, t->base + row * 4UL, sb, 4))
        return 0;
    *out = (sk_fix)((sk_u32)sb[0] | ((sk_u32)sb[1] << 8)
                  | ((sk_u32)sb[2] << 16) | ((sk_u32)sb[3] << 24));
    return 1;
}

static int sk_load_gains(FILE *fp, sk_fix *g, sk_u32 n)
{
    unsigned char buf[4];
    sk_u32 i;

    for (i = 0; i < n; i++) {
        if (fread(buf, 1, 4, fp) != 4)
            return 0;
        g[i] = sk_f32_to_fix(sk_rd32(buf));
    }
    return 1;
}

/* Write a gains row at an explicit offset in the pool.  The file order is
   g1, tables, g2 per layer, so the destination is computed from the layout
   rather than appended blindly. */
static int sk_store_gains_at(sk_model *m, const sk_fix *g, sk_u32 n, sk_u32 base)
{
    unsigned char sb[4];
    sk_u32 i;

    if (base > m->pool.bytes || n * 4UL > m->pool.bytes - base)
        return 0;
    for (i = 0; i < n; i++) {
        sk_fix v = g[i];
        sb[0] = (unsigned char)(v & 0xFFL);
        sb[1] = (unsigned char)((v >> 8) & 0xFFL);
        sb[2] = (unsigned char)((v >> 16) & 0xFFL);
        sb[3] = (unsigned char)((v >> 24) & 0xFFL);
        if (!sk_pool_write(&m->pool, base + i * 4UL, sb, 4))
            return 0;
    }
    return 1;
}

int sk_model_gain_row(sk_model *m, int which, int layer, sk_fix *out)
{
    unsigned char sb[4];
    sk_u32 base;
    sk_u32 i;
    sk_u32 D;

    if (m == SK_NULL || out == SK_NULL || !m->loaded)
        return 0;
    D = (sk_u32)m->D;

    switch (which) {
    case SK_GAIN_G1:
        if (layer < 0 || layer >= m->L)
            return 0;
        base = m->gain_base + (sk_u32)layer * D * 4UL;
        break;
    case SK_GAIN_G2:
        if (layer < 0 || layer >= m->L)
            return 0;
        base = m->gain_base + ((sk_u32)m->L * D + (sk_u32)layer * D) * 4UL;
        break;
    case SK_GAIN_GF:
        base = m->gain_base + 2UL * (sk_u32)m->L * D * 4UL;
        break;
    default:
        return 0;
    }

    for (i = 0; i < D; i++) {
        if (!sk_pool_read(&m->pool, base + i * 4UL, sb, 4))
            return 0;
        out[i] = (sk_fix)((sk_u32)sb[0] | ((sk_u32)sb[1] << 8)
                        | ((sk_u32)sb[2] << 16) | ((sk_u32)sb[3] << 24));
    }
    return 1;
}

int sk_model_load(sk_model *m, const char *path, sk_pool *pool,
                  char *why, sk_u32 whylen)
{
    FILE *fp;
    unsigned char hdr[12];
    int l;
    sk_u32 need;
    sk_pool keep;

    if (m == SK_NULL || path == SK_NULL || pool == SK_NULL) {
        sk_why(why, whylen, "bad arguments");
        return 0;
    }
    /* Copy the pool before clearing the model: callers naturally pass
       &m.pool, and memset(m, 0, ...) would wipe it out from under us. */
    keep = *pool;
    memset(m, 0, sizeof(*m));

    fp = fopen(path, "rb");
    if (fp == SK_NULL) {
        sk_why(why, whylen, "cannot open the model file");
        return 0;
    }
    if (fread(hdr, 1, 12, fp) != 12 || memcmp(hdr, "TNY2", 4) != 0) {
        fclose(fp);
        sk_why(why, whylen, "not a TNY2 model file");
        return 0;
    }

    m->D   = (int)sk_rd16(hdr + 4);
    m->L   = (int)sk_rd16(hdr + 6);
    m->H   = (int)sk_rd16(hdr + 8);
    m->HID = (int)sk_rd16(hdr + 10);

    if (m->D < 8 || m->D > SK_T_MAXD || m->L < 1 || m->L > SK_T_MAXL
        || m->H < 1 || m->D % m->H != 0
        || m->HID < 8 || m->HID > SK_T_MAXHID) {
        fclose(fp);
        sk_why(why, whylen, "network shape out of range");
        return 0;
    }
    m->HD = m->D / m->H;
    m->vocab = SK_T_VOCAB;
    m->ctx = SK_T_CTX;
    m->att_div = sk_fix_sqrt(sk_fix_from_int(m->HD));

    need = sk_model_size((sk_u32)m->D, (sk_u32)m->L, (sk_u32)m->HID,
                         (sk_u32)m->vocab, (sk_u32)m->ctx);
    if (keep.bytes < need) {
        fclose(fp);
        sk_why(why, whylen, "weight store too small for this model");
        return 0;
    }

    /* The weights go into the caller's pool: EMS on DOS, a flat buffer
       elsewhere.  The pool never hands out pointers, so the same code runs
       on a 640 KB machine and on a host with the same result. */
    m->pool = keep;
    m->used = 0;

    if (!sk_load_table(m, fp, &m->emb, (sk_u32)m->vocab, (sk_u32)m->D, why, whylen)
        || !sk_load_table(m, fp, &m->pos, (sk_u32)m->ctx, (sk_u32)m->D, why, whylen)) {
        fclose(fp);
        return 0;
    }

    /* gains are model data too: they go into the pool, not into the struct.
       Reserve the whole block up front, then fill it in file order. */
    {
        sk_u32 gbytes = (2UL * (sk_u32)m->L + 1UL) * (sk_u32)m->D * 4UL;

        if (m->used > m->pool.bytes || gbytes > m->pool.bytes - m->used) {
            fclose(fp);
            sk_why(why, whylen, "no room for the gains");
            return 0;
        }
        m->gain_base = m->used;
        m->used += gbytes;
    }

    for (l = 0; l < m->L; l++) {
        sk_fix tmp[SK_T_MAXD];
        sk_u32 D = (sk_u32)m->D;

        if (!sk_load_gains(fp, tmp, D)
            || !sk_store_gains_at(m, tmp, D, m->gain_base + (sk_u32)l * D * 4UL)) {
            sk_why(why, whylen, "cannot store g1 gains");
            fclose(fp);
            return 0;
        }
        if (!sk_load_table(m, fp, &m->wq[l], D, D, why, whylen)
            || !sk_load_table(m, fp, &m->wk[l], D, D, why, whylen)
            || !sk_load_table(m, fp, &m->wv[l], D, D, why, whylen)
            || !sk_load_table(m, fp, &m->wo[l], D, D, why, whylen)) {
            fclose(fp);
            return 0;
        }
        if (!sk_load_gains(fp, tmp, D)
            || !sk_store_gains_at(m, tmp, D,
                                  m->gain_base + ((sk_u32)m->L * D + (sk_u32)l * D) * 4UL)) {
            sk_why(why, whylen, "cannot store g2 gains");
            fclose(fp);
            return 0;
        }
        if (!sk_load_table(m, fp, &m->f1[l], (sk_u32)m->HID, D, why, whylen)
            || !sk_load_table(m, fp, &m->f2[l], D, (sk_u32)m->HID, why, whylen)) {
            fclose(fp);
            return 0;
        }
    }

    {
        sk_fix tmp[SK_T_MAXD];
        sk_u32 D = (sk_u32)m->D;

        if (!sk_load_gains(fp, tmp, D)
            || !sk_store_gains_at(m, tmp, D,
                                  m->gain_base + 2UL * (sk_u32)m->L * D * 4UL)) {
            sk_why(why, whylen, "cannot store the final gains");
            fclose(fp);
            return 0;
        }
    }

    /* The original program checks this too: a file with trailing bytes is
       a file from a different version, and guessing is worse than saying so. */
    if (fgetc(fp) != EOF) {
        fclose(fp);
        sk_why(why, whylen, "trailing bytes after the last table");
        return 0;
    }
    fclose(fp);

    m->loaded = 1;
    sk_why(why, whylen, "ok");
    return 1;
}

int sk_table_row(const sk_table *t, sk_u32 row, sk_fix *out)
{
    unsigned char buf[SK_T_MAXHID];
    sk_fix s;
    sk_u32 j;
    sk_u32 wbase;

    if (t == SK_NULL || out == SK_NULL || t->pool == SK_NULL)
        return 0;
    if (row >= t->rows || t->cols > (sk_u32)SK_T_MAXHID)
        return 0;
    if (!sk_table_scale(t, row, &s))
        return 0;

    wbase = t->base + t->rows * 4UL;
    if (!sk_pool_read(t->pool, wbase + row * t->cols, buf, t->cols))
        return 0;

    for (j = 0; j < t->cols; j++)
        out[j] = sk_scale_fix(sk_fix_from_int((sk_i32)(signed char)buf[j]), s);
    return 1;
}

int sk_table_matvec(const sk_table *t, const sk_fix *x, sk_fix *y)
{
    unsigned char buf[SK_T_MAXHID];
    sk_u32 i, j;
    sk_u32 wbase;
    sk_fix scale;

    if (t == SK_NULL || x == SK_NULL || y == SK_NULL || t->pool == SK_NULL)
        return 0;
    if (t->cols > (sk_u32)SK_T_MAXHID)
        return 0;

    wbase = t->base + t->rows * 4UL;

    for (i = 0; i < t->rows; i++) {
        sk_u32 hi, lo;

        if (!sk_pool_read(t->pool, wbase + i * t->cols, buf, t->cols))
            return 0;

        hi = 0;
        lo = 0;
        for (j = 0; j < t->cols; j++) {
            sk_i32 w = (sk_i32)(signed char)buf[j];

            /* w is an INTEGER, x is Q16.16.  The product w*x is already in
               Q16.16, so the weight must be promoted with sk_fix_from_int:
               passing the raw 48 would be read as 48/65536.  That exact
               mistake produced an output five orders of magnitude too
               small in the first version. */
            sk_acc_add(&hi, &lo, sk_fix_mul(sk_fix_from_int(w), x[j]));
        }

        if (!sk_table_scale(t, i, &scale))
            return 0;
        y[i] = sk_acc_scale(hi, lo, scale);
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* selftest                                                           */
/* ------------------------------------------------------------------ */

sk_u32 sk_model_weights_crc(const sk_model *m)
{
    unsigned char buf[SK_T_MAXHID];
    sk_u32 crc;
    int l, k;

    if (m == SK_NULL || !m->loaded)
        return 0;

    crc = 0;
    for (l = -1; l < m->L; l++) {
        const sk_table *set[6];
        int nt;

        if (l < 0) {
            set[0] = &m->emb;
            set[1] = &m->pos;
            nt = 2;
        } else {
            set[0] = &m->wq[l];
            set[1] = &m->wk[l];
            set[2] = &m->wv[l];
            set[3] = &m->wo[l];
            set[4] = &m->f1[l];
            set[5] = &m->f2[l];
            nt = 6;
        }

        for (k = 0; k < nt; k++) {
            const sk_table *t = set[k];
            sk_u32 r;

            if (t->cols > (sk_u32)SK_T_MAXHID)
                return 0;
            for (r = 0; r < t->rows; r++) {
                if (!sk_pool_read(t->pool, t->base + t->rows * 4UL + r * t->cols,
                                  buf, t->cols))
                    return 0;
                crc = sk_crc32c_update(crc, buf, t->cols);
            }
        }
    }
    return crc;
}
int sk_tensor_selftest(void)
{
    int ok;
    sk_fix v;

    ok = 1;

    /* IEEE-754 conversions with values whose Q16.16 form is exact. */
    v = sk_f32_to_fix(0x3F800000UL);           /* 1.0   */
    if (v != SK_FIX_ONE) { printf("  FAIL: f32 1.0 -> %ld\n", (long)v); ok = 0; }

    v = sk_f32_to_fix(0x40000000UL);           /* 2.0   */
    if (v != 2L * SK_FIX_ONE) { printf("  FAIL: f32 2.0 -> %ld\n", (long)v); ok = 0; }

    v = sk_f32_to_fix(0x3F000000UL);           /* 0.5   */
    if (v != SK_FIX_ONE / 2) { printf("  FAIL: f32 0.5 -> %ld\n", (long)v); ok = 0; }

    v = sk_f32_to_fix(0xBF800000UL);           /* -1.0  */
    if (v != -SK_FIX_ONE) { printf("  FAIL: f32 -1.0 -> %ld\n", (long)v); ok = 0; }

    v = sk_f32_to_fix(0x00000000UL);           /* 0.0   */
    if (v != 0) { printf("  FAIL: f32 0.0 -> %ld\n", (long)v); ok = 0; }

    v = sk_f32_to_fix(0x40490FDBUL);           /* pi    */
    if (v < 205884L || v > 205890L) {
        printf("  FAIL: f32 pi -> %ld\n", (long)v);
        ok = 0;
    }

    /* 1e10 does not fit Q16.16: it must saturate, not wrap. */
    v = sk_f32_to_fix(0x501502F9UL);
    if (v != SK_FIX_MAX) { printf("  FAIL: f32 1e10 -> %ld\n", (long)v); ok = 0; }

    /* NaN must saturate rather than propagate garbage. */
    v = sk_f32_to_fix(0x7FC00000UL);
    if (v != SK_FIX_MAX) { printf("  FAIL: f32 NaN -> %ld\n", (long)v); ok = 0; }

    printf("  tensor: f32 conversions %s\n", ok ? "ok" : "FAILED");
    return ok;
}
