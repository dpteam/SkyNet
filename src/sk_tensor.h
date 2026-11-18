/* sk_tensor.h - int8 weight tables and the TINY.BIN loader.
 *
 * The representation is the one Tiny-2.1 uses and the one that survives a
 * 640 KB machine: one signed byte per weight, one scale per row.  A row is
 * dequantised on demand as `w * scale`, so the resident cost is 1 byte per
 * weight instead of 4.
 *
 * Everything here is Q16.16.  The file stores its scales as IEEE-754
 * floats, which sk_f32_to_fix() converts exactly without using a single
 * floating-point instruction (see sk_math.h for the arithmetic).
 *
 * SKYNET M2 / subsystem: model/tensor
 */
#ifndef SK_TENSOR_H
#define SK_TENSOR_H

#include "sk_types.h"
#include "sk_mem.h"

/* Bounds taken from TINY.C so that a file which the original program
   accepts is also accepted here. */
#define SK_T_MAXD   128
#define SK_T_MAXHID 512
#define SK_T_MAXL   8
#define SK_T_VOCAB  97
#define SK_T_CTX    128

/* One weight table: rows x cols values, one Q8.24 scale per row.
 *
 * Both the weights AND the scales live in the pool.  Keeping the scales
 * resident in the struct was the obvious design and it does not fit: with
 * SK_T_MAXL layers x 6 tables x 512 scales x 4 bytes, sk_model alone
 * exceeds 64 KB, which is the whole data segment on a 16-bit compiler.
 * OpenWatcom rejected it with "Maximum struct or union size is 64K" - a
 * limit no host compiler has, so only the DOS build could catch it.
 *
 * Layout inside the pool for a table at offset `base`:
 *   base + 0                    : rows * 4 bytes of Q8.24 scales
 *   base + rows*4 + r*stride    : row r, `cols` values at `bits` per value
 *
 * `bits` is 8 or 4.  Four-bit weights halve the storage and are the one
 * compression that degrades gradually on this model - unlike truncating
 * layers or units, which falls off a cliff.
 */
typedef struct {
    sk_pool *pool;        /* store holding scales and weights   */
    sk_u32   base;        /* byte offset of the scale array     */
    sk_u32   rows;
    sk_u32   cols;
    int      bits;        /* 8 or 4                              */
} sk_table;

/* Bytes one row of `cols` values occupies at `bits` per value. */
sk_u32 sk_table_stride(sk_u32 cols, int bits);

/* A loaded network.  Layout follows TNY2 exactly. */
typedef struct {
    int   D;              /* model width                        */
    int   L;              /* layers                             */
    int   H;              /* attention heads                    */
    int   HID;            /* feed-forward middle width           */
    int   HD;             /* D / H                              */
    int   vocab;
    int   ctx;
    sk_fix att_div;       /* sqrt(HD), Q16.16                   */

    sk_table emb;
    sk_table pos;

    /* The output projection.
     *
     * In the released model this IS the embedding table (weight tying):
     * `logits = emb * hf`. That is efficient but it means one tensor serves
     * two purposes, and the measurement in docs/06-training.md showed what
     * that costs: training the output side degrades the input side, and the
     * loss stops falling across examples.
     *
     * So the two are untied here. `out` starts as an exact copy of `emb`,
     * which keeps inference bit-identical to the released model, and can
     * then be trained independently. Cost: 97 x D bytes plus 97 scales,
     * about 9.7 KB for the released shape. */
    sk_table out;

    sk_table wq[SK_T_MAXL];
    sk_table wk[SK_T_MAXL];
    sk_table wv[SK_T_MAXL];
    sk_table wo[SK_T_MAXL];
    sk_table f1[SK_T_MAXL];
    sk_table f2[SK_T_MAXL];

    sk_fix g1_unused_placeholder[1];

    /* The per-layer gains live in the pool, at gain_base:
         g1[l] : gain_base + l*D*4
         g2[l] : gain_base + (L*D + l*D)*4
         gf    : gain_base + 2*L*D*4
       Keeping them resident as arrays was 9 KB of struct - more than the
       8 KB DOS stack, so cmd_model died with no output at all.  Same lesson
       as the scales: anything sized by SK_T_MAX* belongs in the pool. */
    sk_u32 gain_base;
    sk_pool pool;         /* the one store all weights share     */
    sk_u32  used;         /* bytes consumed inside it            */
    sk_u32  weights_crc;  /* CRC32C of every weight byte, in file order */

    /* How many layers are actually used.  Loading can stop early, which is
       how the quality-versus-size curve is measured: a truncated model is
       a real model, just shallower. */
    int     layers_used;
    int     bits;         /* 8 or 4 bits per weight              */

    int     loaded;
} sk_model;

/* Load a TNY2 file.  The caller supplies an already-open pool, so the same
   loader works with EMS on DOS and with a flat buffer on a host - the
   choice of storage belongs to the caller, not to the file format.
   Returns 1 on success, and on failure fills `why` with a short reason. */
int sk_model_load(sk_model *m, const char *path, sk_pool *pool,
                  char *why, sk_u32 whylen);

/* Load only the first `layers` layers of the file.
 *
 * This is how the quality-versus-size curve is measured without training
 * anything: a shallower network is a valid network, it just has fewer
 * transformer blocks. The remaining layers are still read from the file -
 * they have to be skipped, not ignored - so the parse stays honest about
 * the format. Pass 0 for "all layers".
 *
 * The point is to answer "what do I lose if I only have N kilobytes"
 * with a measurement instead of an opinion. */
int sk_model_load_truncated(sk_model *m, const char *path, sk_pool *pool,
                            int layers, char *why, sk_u32 whylen);

/* Same, but also cut the feed-forward width to `hid` units.
 *
 * Truncating layers turned out to be a cliff, not a slope: three of four
 * blocks give 5.7% accuracy and four give 100%. The feed-forward width is
 * the other dimension that can shrink, and it is continuous, so it is the
 * one worth measuring. f1 loses its last rows and f2 its last columns,
 * which is exactly the sub-network that keeps those units. */
int sk_model_load_shaped(sk_model *m, const char *path, sk_pool *pool,
                         int layers, int hid, int bits, char *why, sk_u32 whylen);

/* Bytes of weight storage this model actually occupies. */
sk_u32 sk_model_bytes(const sk_model *m);

#define SK_GAIN_G1 0
#define SK_GAIN_G2 1
#define SK_GAIN_GF 2

/* Read one gains row (D values) as Q16.16. */
int sk_model_gain_row(sk_model *m, int which, int layer, sk_fix *out);

/* Bytes needed for the weights of a network of this shape, before
   loading.  Lets the caller size the storage without parsing the file. */
sk_u32 sk_model_size(sk_u32 D, sk_u32 L, sk_u32 HID,
                     sk_u32 vocab, sk_u32 ctx);
sk_u32 sk_model_size_bits(sk_u32 D, sk_u32 L, sk_u32 HID,
                          sk_u32 vocab, sk_u32 ctx, int bits);

/* y[i] = scale[row i] * sum_j w[i][j] * x[j], accumulated in Q16.16.
   This is the hot loop of the whole model, so it accumulates in 32 bits
   per row and multiplies by the scale once, which keeps the rounding
   error at one LSB per output instead of one per product. */
int sk_table_matvec(const sk_table *t, const sk_fix *x, sk_fix *y);

/* Read a row as raw int8 values, whatever the stored width. */
int sk_table_read_row(const sk_table *t, sk_u32 row, signed char *out);

/* Read and write one row scale (Q8.24). */
int sk_table_scale_read(const sk_table *t, sk_u32 row, sk_fix *out);
int sk_table_scale_write(const sk_table *t, sk_u32 row, sk_fix v);

/* y[i] = sum_j w[i][j] * x[j], WITHOUT the row scale.  The trainer needs
   the un-scaled dot product because dL/dscale is exactly dy times it. */
int sk_table_matvec_noscale(const sk_table *t, const sk_fix *x, sk_fix *y);

/* Look one row up as Q16.16 values (used for embeddings). */
int sk_table_row(const sk_table *t, sk_u32 row, sk_fix *out);

/* Recompute the checksum of the weights by reading them back out of the
   store.  Compared against sk_model.weights_crc, which was taken while
   parsing the file, this proves the storage path is lossless without
   needing a second parser of the file format. */
sk_u32 sk_model_weights_crc(const sk_model *m);

/* Checks that need no model file. */
int sk_tensor_selftest(void);

#endif /* SK_TENSOR_H */
