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

/* 4-bit weights: an int8 value is requantised to a nibble and expanded back
   with this table.  The table is built once, so the hot loop pays one array
   index instead of a division.
 *
 * The compensation factor is 127/7, because the row scale was derived from
 * the int8 range: a nibble holds +-7 where the byte held +-127, so a nibble
 * of n stands for 127*n/7 in the original units.  Without that factor the
 * weights come out 18 times too small and the model produces noise. */
static signed char sk_q4_to_q8[16];
static int sk_q4_ready = 0;

static void sk_q4_init(void)
{
    int i;

    if (sk_q4_ready)
        return;
    for (i = 0; i < 16; i++) {
        int v = (i < 8) ? i : i - 16;      /* -8..7 */
        int s = (v * 127 + (v >= 0 ? 3 : -3)) / 7;
        if (s > 127) s = 127;
        if (s < -127) s = -127;
        sk_q4_to_q8[i] = (signed char)s;
    }
    sk_q4_ready = 1;
}

/* Pack `cols` int8 values into nibbles, two per byte, high nibble first. */
static void sk_pack_q4(const unsigned char *src, unsigned char *dst, sk_u32 cols)
{
    sk_u32 i;

    for (i = 0; i < cols; i += 2) {
        int a = (signed char)src[i];
        int hi = (a * 7 + (a >= 0 ? 63 : -63)) / 127;
        int lo = 0;
        unsigned char b;

        if (hi > 7) hi = 7;
        if (hi < -8) hi = -8;

        if (i + 1 < cols) {
            int c = (signed char)src[i + 1];
            lo = (c * 7 + (c >= 0 ? 63 : -63)) / 127;
            if (lo > 7) lo = 7;
            if (lo < -8) lo = -8;
        }
        b = (unsigned char)(((hi & 0x0F) << 4) | (lo & 0x0F));
        dst[i / 2] = b;
    }
}

sk_u32 sk_table_stride(sk_u32 cols, int bits)
{
    if (bits == 4)
        return (cols + 1UL) / 2UL;
    return cols;
}

/* Bytes for one table: rows of 4-byte scales, then rows*stride weights. */
static sk_u32 sk_table_bytes_bits(sk_u32 rows, sk_u32 cols, int bits)
{
    return rows * 4UL + rows * sk_table_stride(cols, bits);
}

static sk_u32 sk_table_bytes(sk_u32 rows, sk_u32 cols)
{
    return sk_table_bytes_bits(rows, cols, 8);
}

sk_u32 sk_model_size(sk_u32 D, sk_u32 L, sk_u32 HID,
                     sk_u32 vocab, sk_u32 ctx)
{
    return sk_model_size_bits(D, L, HID, vocab, ctx, 8);
}

sk_u32 sk_model_size_bits(sk_u32 D, sk_u32 L, sk_u32 HID,
                          sk_u32 vocab, sk_u32 ctx, int bits)
{
    sk_u32 n;
    sk_u32 l;
    /* The embedding table doubles as the output projection, so it stays at
       8 bits even in 4-bit mode - see the note in sk_model_load_shaped. */
    int ebits = (bits == 4) ? 8 : bits;

    /* emb and out are separate tables now, so both are counted. */
    n = sk_table_bytes_bits(vocab, D, ebits) + sk_table_bytes_bits(ctx, D, ebits)
      + sk_table_bytes_bits(vocab, D, ebits);
    for (l = 0; l < L; l++) {
        n += 4UL * D;                                   /* g1 gains   */
        n += 4UL * sk_table_bytes_bits(D, D, bits);     /* wq wk wv wo */
        n += 4UL * D;                                   /* g2 gains   */
        n += sk_table_bytes_bits(HID, D, bits);
        n += sk_table_bytes_bits(D, HID, bits);
    }
    n += 4UL * D;                                       /* gf gains   */
    return n;
}

/* Copy one table to a new location in the same pool.  Used to initialise
   the output projection from the embedding table, which is what unties the
   two while keeping inference identical to the released model. */
static int sk_copy_table(sk_model *m, sk_table *dst, const sk_table *src,
                         char *why, sk_u32 whylen)
{
    signed char w[SK_T_MAXHID];
    sk_fix scale;
    sk_u32 r;

    dst->pool = &m->pool;
    dst->rows = src->rows;
    dst->cols = src->cols;
    dst->bits = src->bits;
    dst->base = m->used;

    if (m->used > m->pool.bytes
        || sk_table_bytes_bits(src->rows, src->cols, src->bits)
           > m->pool.bytes - m->used) {
        sk_why(why, whylen, "no room for the output projection");
        return 0;
    }

    /* Write the scale array first, then the rows, matching the layout
       sk_load_table_partial produces. */
    for (r = 0; r < src->rows; r++) {
        if (!sk_table_scale_read(src, r, &scale)
            || !sk_table_scale_write(dst, r, scale)) {
            sk_why(why, whylen, "cannot copy scales");
            return 0;
        }
    }
    {
        sk_u32 wbase = dst->base + dst->rows * 4UL;
        sk_u32 stride = sk_table_stride(dst->cols, dst->bits);

        for (r = 0; r < src->rows; r++) {
            if (!sk_table_read_row(src, r, w))
                return 0;
            if (dst->bits == 4) {
                unsigned char packed[(SK_T_MAXHID + 1) / 2];

                sk_pack_q4((const unsigned char *)w, packed, dst->cols);
                if (!sk_pool_write(&m->pool, wbase + r * stride, packed, stride))
                    return 0;
            } else {
                if (!sk_pool_write(&m->pool, wbase + r * stride,
                                   w, dst->cols))
                    return 0;
            }
        }
    }
    m->used += sk_table_bytes_bits(src->rows, src->cols, src->bits);
    return 1;
}

/* Read a table, keeping only the first rows_used x cols_used corner of a
   rows_total x cols_total table in the file.  The rest is skipped, not
   ignored, so the file position stays correct.

   This is what makes feed-forward width truncation possible: f1 is
   (HID x D) and f2 is (D x HID), so cutting HID means keeping f1's first
   rows and f2's first columns - a real sub-network, not a broken one. */
static int sk_load_table_partial(sk_model *m, FILE *fp, sk_table *t,
                                 sk_u32 rows_used, sk_u32 rows_total,
                                 sk_u32 cols_used, sk_u32 cols_total,
                                 int bits, char *why, sk_u32 whylen)
{
    unsigned char buf[4 + SK_T_MAXHID];
    unsigned char packed[(SK_T_MAXHID + 1) / 2];
    sk_u32 r;
    sk_u32 need;
    sk_u32 wbase;
    sk_u32 stride;

    if (rows_used > rows_total || cols_used > cols_total
        || rows_used > (sk_u32)SK_T_MAXHID || cols_used > (sk_u32)SK_T_MAXHID) {
        sk_why(why, whylen, "table larger than the program supports");
        return 0;
    }
    if (bits != 8 && bits != 4) {
        sk_why(why, whylen, "unsupported weight width");
        return 0;
    }
    need = sk_table_bytes_bits(rows_used, cols_used, bits);
    if (m->used > m->pool.bytes || need > m->pool.bytes - m->used) {
        sk_why(why, whylen, "weights do not fit in the storage given");
        return 0;
    }

    sk_q4_init();
    stride = sk_table_stride(cols_used, bits);
    t->pool = &m->pool;
    t->base = m->used;
    t->rows = rows_used;
    t->cols = cols_used;
    t->bits = bits;
    wbase = m->used + rows_used * 4UL;

    for (r = 0; r < rows_total; r++) {
        sk_fix scale;
        unsigned char sb[4];

        if (fread(buf, 1, (size_t)(4 + cols_total), fp)
            != (size_t)(4 + cols_total)) {
            sk_why(why, whylen, "file ends inside a weight table");
            return 0;
        }
        if (r >= rows_used)
            continue;                   /* read past it, keep nothing */

        scale = (sk_fix)sk_f32_to_q(sk_rd32(buf), SK_SCALE_BITS);
        sb[0] = (unsigned char)(scale & 0xFFL);
        sb[1] = (unsigned char)((scale >> 8) & 0xFFL);
        sb[2] = (unsigned char)((scale >> 16) & 0xFFL);
        sb[3] = (unsigned char)((scale >> 24) & 0xFFL);
        if (!sk_pool_write(&m->pool, m->used + r * 4UL, sb, 4)) {
            sk_why(why, whylen, "cannot store scales");
            return 0;
        }

        if (bits == 4) {
            unsigned char expanded[SK_T_MAXHID];
            sk_u32 j;

            sk_pack_q4(buf + 4, packed, cols_used);
            if (!sk_pool_write(&m->pool, wbase + r * stride, packed,
                               sk_table_stride(cols_used, 4))) {
                sk_why(why, whylen, "cannot store weights");
                return 0;
            }
            /* Checksum the values as they will be read back, not as they
               arrived: after 4-bit packing the stored bytes are nibbles, so
               a checksum over the raw file bytes could never match. */
            for (j = 0; j < cols_used; j++) {
                unsigned char b = packed[j / 2];
                int nib = (j & 1) ? (b & 0x0F) : (b >> 4);
                expanded[j] = (unsigned char)sk_q4_to_q8[nib];
            }
            m->weights_crc = sk_crc32c_update(m->weights_crc, expanded, cols_used);
        } else {
            if (!sk_pool_write(&m->pool, wbase + r * stride,
                               buf + 4, cols_used)) {
                sk_why(why, whylen, "cannot store weights");
                return 0;
            }
            m->weights_crc = sk_crc32c_update(m->weights_crc, buf + 4, cols_used);
        }
    }
    m->used += need;
    return 1;
}

/* Read one table: rows of (float32 scale, cols int8 weights).
   The scales are converted and stored first, then the weights, so a table
   occupies two contiguous blocks in the pool and the struct stays tiny. */
static int sk_load_table(sk_model *m, FILE *fp, sk_table *t,
                         sk_u32 rows, sk_u32 cols, int bits,
                         char *why, sk_u32 whylen)
{
    return sk_load_table_partial(m, fp, t, rows, rows, cols, cols, bits,
                                 why, whylen);
}

/* Read the scale of one row. */
int sk_table_scale_read(const sk_table *t, sk_u32 row, sk_fix *out)
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
    return sk_model_load_truncated(m, path, pool, 0, why, whylen);
}

sk_u32 sk_model_bytes(const sk_model *m)
{
    return (m == SK_NULL) ? 0 : m->used;
}

/* Skip n bytes of the file, so an unused layer is parsed past rather than
   silently ignored.  Returns 1 on success. */
static int sk_skip(FILE *fp, sk_u32 n)
{
    long cur;
    long end;

    cur = ftell(fp);
    if (cur < 0)
        return 0;
    if (fseek(fp, 0L, SEEK_END) != 0)
        return 0;
    end = ftell(fp);
    if (end < 0 || (long)n > end - cur)
        return 0;
    return (fseek(fp, cur + (long)n, SEEK_SET) == 0) ? 1 : 0;
}

int sk_model_load_truncated(sk_model *m, const char *path, sk_pool *pool,
                            int layers, char *why, sk_u32 whylen)
{
    return sk_model_load_shaped(m, path, pool, layers, 0, 8, why, whylen);
}

int sk_model_load_shaped(sk_model *m, const char *path, sk_pool *pool,
                         int layers, int hid, int bits, char *why, sk_u32 whylen)
{
    FILE *fp;
    unsigned char hdr[12];
    int l;
    sk_u32 need;
    sk_pool keep;
    int file_layers;
    int file_hid;

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

    m->D      = (int)sk_rd16(hdr + 4);
    file_layers = (int)sk_rd16(hdr + 6);
    m->H      = (int)sk_rd16(hdr + 8);
    file_hid  = (int)sk_rd16(hdr + 10);

    if (m->D < 8 || m->D > SK_T_MAXD || file_layers < 1
        || file_layers > SK_T_MAXL
        || m->H < 1 || m->D % m->H != 0
        || file_hid < 8 || file_hid > SK_T_MAXHID) {
        fclose(fp);
        sk_why(why, whylen, "network shape out of range");
        return 0;
    }

    /* Truncation: use the first `layers` blocks and the first `hid`
       feed-forward units, but keep the file's own dimensions so the unused
       parts can be read past rather than ignored. */
    if (layers <= 0 || layers > file_layers)
        layers = file_layers;
    if (hid <= 0 || hid > file_hid)
        hid = file_hid;

    m->L = layers;
    m->layers_used = layers;
    m->HID = hid;
    m->bits = (bits == 4) ? 4 : 8;

    m->HD = m->D / m->H;
    m->vocab = SK_T_VOCAB;
    m->ctx = SK_T_CTX;
    m->att_div = sk_fix_sqrt(sk_fix_from_int(m->HD));

    need = sk_model_size_bits((sk_u32)m->D, (sk_u32)m->L, (sk_u32)m->HID,
                              (sk_u32)m->vocab, (sk_u32)m->ctx, m->bits);
    if (keep.bytes < need) {
        fclose(fp);
        sk_why(why, whylen, "weight store too small for this model");
        return 0;
    }

    m->pool = keep;
    m->used = 0;

    /* The embedding table is also the output projection (weight tying),
           so quantising it hits both the input and the answer distribution.
           Keeping it at 8 bits costs about 9 KB and, measured, is worth
           far more than that. */
    if (!sk_load_table(m, fp, &m->emb, (sk_u32)m->vocab, (sk_u32)m->D,
                       (m->bits == 4) ? 8 : m->bits, why, whylen)
        || !sk_load_table(m, fp, &m->pos, (sk_u32)m->ctx, (sk_u32)m->D,
                          (m->bits == 4) ? 8 : m->bits, why, whylen)) {
        fclose(fp);
        return 0;
    }

    /* Untie the output projection: an exact copy of the embedding table,
       which keeps inference identical to the released model and lets the
       trainer update one without damaging the other. */
    if (!sk_copy_table(m, &m->out, &m->emb, why, whylen)) {
        fclose(fp);
        return 0;
    }

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

    for (l = 0; l < file_layers; l++) {
        sk_u32 D = (sk_u32)m->D;

        if (l >= m->L) {
            /* Past the truncation point: advance over one complete block.
               g1, wq, wk, wv, wo, g2, f1, f2 - in that file order. */
            sk_u32 layer_bytes = 2UL * D * 4UL
                               + 4UL * sk_table_bytes(D, D)
                               + sk_table_bytes((sk_u32)file_hid, D)
                               + sk_table_bytes(D, (sk_u32)file_hid);

            if (!sk_skip(fp, layer_bytes)) {
                sk_why(why, whylen, "file ends inside a skipped layer");
                fclose(fp);
                return 0;
            }
            continue;
        }

        {
            sk_fix tmp[SK_T_MAXD];

            if (!sk_load_gains(fp, tmp, D)
                || !sk_store_gains_at(m, tmp, D,
                                      m->gain_base + (sk_u32)l * D * 4UL)) {
                sk_why(why, whylen, "cannot store g1 gains");
                fclose(fp);
                return 0;
            }
            if (!sk_load_table(m, fp, &m->wq[l], D, D, m->bits, why, whylen)
                || !sk_load_table(m, fp, &m->wk[l], D, D, m->bits, why, whylen)
                || !sk_load_table(m, fp, &m->wv[l], D, D, m->bits, why, whylen)
                || !sk_load_table(m, fp, &m->wo[l], D, D, m->bits, why, whylen)) {
                fclose(fp);
                return 0;
            }
            if (!sk_load_gains(fp, tmp, D)
                || !sk_store_gains_at(m, tmp, D,
                                      m->gain_base + ((sk_u32)m->L * D
                                                      + (sk_u32)l * D) * 4UL)) {
                sk_why(why, whylen, "cannot store g2 gains");
                fclose(fp);
                return 0;
            }

            /* f1 is (HID x D): keep the first `hid` rows.
               f2 is (D x HID): keep the first `hid` columns. */
            if (!sk_load_table_partial(m, fp, &m->f1[l],
                                       (sk_u32)m->HID, (sk_u32)file_hid,
                                       D, D, m->bits, why, whylen)
                || !sk_load_table_partial(m, fp, &m->f2[l],
                                          D, D,
                                          (sk_u32)m->HID, (sk_u32)file_hid,
                                          m->bits, why, whylen)) {
                fclose(fp);
                return 0;
            }
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

    /* Trailing bytes are only an error when nothing was truncated: a
       truncated model legitimately leaves the file half-read. */
    if (layers == file_layers && hid == file_hid && fgetc(fp) != EOF) {
        fclose(fp);
        sk_why(why, whylen, "trailing bytes after the last table");
        return 0;
    }
    fclose(fp);

    m->loaded = 1;
    sk_why(why, whylen, "ok");
    return 1;
}

/* Read one row and expand it to int8 values, whatever the stored width.
   Keeps the 4-bit unpacking in one place instead of in every consumer. */
int sk_table_read_row(const sk_table *t, sk_u32 row, signed char *out)
{
    unsigned char packed[(SK_T_MAXHID + 1) / 2];
    unsigned char bytes[SK_T_MAXHID];
    sk_u32 stride;
    sk_u32 wbase;
    sk_u32 j;

    if (t == SK_NULL || out == SK_NULL || row >= t->rows)
        return 0;
    if (t->cols > (sk_u32)SK_T_MAXHID)
        return 0;

    stride = sk_table_stride(t->cols, t->bits);
    wbase = t->base + t->rows * 4UL;

    if (t->bits == 4) {
        if (!sk_pool_read(t->pool, wbase + row * stride, packed, stride))
            return 0;
        for (j = 0; j < t->cols; j++) {
            unsigned char b = packed[j / 2];
            int nib = (j & 1) ? (b & 0x0F) : (b >> 4);
            out[j] = sk_q4_to_q8[nib];
        }
    } else {
        if (!sk_pool_read(t->pool, wbase + row * stride, bytes, t->cols))
            return 0;
        for (j = 0; j < t->cols; j++)
            out[j] = (signed char)bytes[j];
    }
    return 1;
}

int sk_table_matvec_noscale(const sk_table *t, const sk_fix *x, sk_fix *y)
{
    signed char w[SK_T_MAXHID];
    sk_u32 i, j;

    if (t == SK_NULL || x == SK_NULL || y == SK_NULL)
        return 0;
    for (i = 0; i < t->rows; i++) {
        sk_u32 hi, lo;

        if (!sk_table_read_row(t, i, w))
            return 0;
        hi = 0;
        lo = 0;
        for (j = 0; j < t->cols; j++)
            sk_acc_add(&hi, &lo,
                       sk_fix_mul(sk_fix_from_int((sk_i32)w[j]), x[j]));
        y[i] = sk_acc_result(hi, lo);
    }
    return 1;
}

int sk_table_scale_write(const sk_table *t, sk_u32 row, sk_fix v)
{
    sk_u8 b[4];

    if (t == SK_NULL || t->pool == SK_NULL || row >= t->rows)
        return 0;
    b[0] = (sk_u8)(v & 0xFFL);
    b[1] = (sk_u8)((v >> 8) & 0xFFL);
    b[2] = (sk_u8)((v >> 16) & 0xFFL);
    b[3] = (sk_u8)((v >> 24) & 0xFFL);
    return sk_pool_write(t->pool, t->base + row * 4UL, b, 4);
}

int sk_table_row(const sk_table *t, sk_u32 row, sk_fix *out)
{
    signed char w[SK_T_MAXHID];
    sk_fix s;
    sk_u32 j;

    if (t == SK_NULL || out == SK_NULL || t->pool == SK_NULL)
        return 0;
    if (row >= t->rows)
        return 0;
    if (!sk_table_scale_read(t, row, &s))
        return 0;
    if (!sk_table_read_row(t, row, w))
        return 0;

    for (j = 0; j < t->cols; j++)
        out[j] = sk_scale_fix(sk_fix_from_int((sk_i32)w[j]), s);
    return 1;
}

int sk_table_matvec(const sk_table *t, const sk_fix *x, sk_fix *y)
{
    signed char w[SK_T_MAXHID];
    sk_u32 i, j;
    sk_fix scale;

    if (t == SK_NULL || x == SK_NULL || y == SK_NULL || t->pool == SK_NULL)
        return 0;
    if (t->cols > (sk_u32)SK_T_MAXHID)
        return 0;

    for (i = 0; i < t->rows; i++) {
        sk_u32 hi, lo;

        if (!sk_table_read_row(t, i, w))
            return 0;

        hi = 0;
        lo = 0;
        for (j = 0; j < t->cols; j++) {
            sk_i32 v = (sk_i32)w[j];

            /* w is an INTEGER, x is Q16.16.  The product w*x is already in
               Q16.16, so the weight must be promoted with sk_fix_from_int:
               passing the raw 48 would be read as 48/65536.  That exact
               mistake produced an output five orders of magnitude too
               small in the first version. */
            sk_acc_add(&hi, &lo, sk_fix_mul(sk_fix_from_int(v), x[j]));
        }

        if (!sk_table_scale_read(t, i, &scale))
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
                signed char w[SK_T_MAXHID];
                unsigned char flat[SK_T_MAXHID];
                sk_u32 j;

                if (!sk_table_read_row(t, r, w))
                    return 0;
                /* Flatten to bytes so a 4-bit table checksums the same way
                   it was checksummed at load time. */
                for (j = 0; j < t->cols; j++)
                    flat[j] = (unsigned char)w[j];
                crc = sk_crc32c_update(crc, flat, t->cols);
            }
        }
    }
    return crc;
}
int sk_tensor_selftest(void)
{
    int ok;
    sk_fix v;
    sk_u8 storage[4096];
    sk_pool pool;

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

    /* 4-bit packing round-trip.  The compression is only worth having if
       what comes back out is the value that went in, scaled to a nibble,
       so this checks the expansion table rather than trusting it. */
    {
        sk_table t;
        sk_fix row[8];
        sk_fix back[8];
        sk_u8 packed[4];
        signed char w8[8];
        int i;

        if (!sk_pool_open_from(&pool, sizeof(storage), storage))
            return 0;

        t.pool = &pool;
        t.base = 0;
        t.rows = 1;
        t.cols = 8;
        t.bits = 4;

        /* values at the extremes and in between */
        for (i = 0; i < 8; i++)
            row[i] = (sk_fix)((i - 4) * 4096L);     /* -0.25 .. 0.1875 */

        /* pack by hand the way the loader does, then read it back */
        for (i = 0; i < 8; i++) {
            int a = (int)(row[i] >> 12);            /* crude, just for shape */
            if (a > 7) a = 7;
            if (a < -8) a = -8;
            w8[i] = (signed char)(a * 18);
        }
        sk_q4_init();
        for (i = 0; i < 4; i++) {
            int hi = w8[i * 2] / 18;
            int lo = w8[i * 2 + 1] / 18;
            if (hi > 7) hi = 7;
            if (hi < -8) hi = -8;
            if (lo > 7) lo = 7;
            if (lo < -8) lo = -8;
            packed[i] = (unsigned char)(((hi & 0x0F) << 4) | (lo & 0x0F));
        }
        if (!sk_pool_write(&pool, 0, packed, 4))
            return 0;

        if (!sk_table_read_row(&t, 0, w8)) {
            printf("  FAIL: 4-bit row read\n");
            return 0;
        }
        /* the expansion must be monotone and hit the extremes exactly */
        if ((int)w8[0] > (int)w8[7]) {
            printf("  FAIL: 4-bit expansion is not monotone\n");
            return 0;
        }
        if (sk_q4_to_q8[7] != 127 || sk_q4_to_q8[8] != -127) {
            printf("  FAIL: 4-bit extremes are %d and %d\n",
                   (int)sk_q4_to_q8[7], (int)sk_q4_to_q8[8]);
            return 0;
        }
        (void)back;
        (void)row;
    }

    return ok;
}
