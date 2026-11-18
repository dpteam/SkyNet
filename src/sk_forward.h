/* sk_forward.h - the Tiny-2.1 forward pass in Q16.16.
 *
 * The architecture, taken from the released TINY.C and not from memory:
 *
 *   x = emb[tok] + pos[p]
 *   for each layer:
 *     h  = rmsnorm(x, g1)          s = 1/sqrt(mean(x^2) + eps)
 *     k  = quantise(wk * h)        stored as int8 + one scale per position
 *     v  = quantise(wv * h)        stored the same way
 *     q  = wq * h
 *     for each head:
 *       s_j = dot(q_head, k_head[j]) * scale_k[j] / sqrt(HD)
 *       a   = softmax(s) . (scale_v[j] * v_head[j])
 *     x += wo * a
 *     h  = rmsnorm(x, g2)
 *     x += f2 * relu(f1 * h)
 *   out = emb * rmsnorm(x, gf)     (weight tying: the output projection is
 *                                   the embedding table read the other way)
 *
 * Two details that are easy to miss and change the numbers:
 *   - the KV cache is QUANTISED to int8 with a per-position scale, so even
 *     the float implementation is lossy there;
 *   - the output projection reuses `emb`, not a separate matrix.
 *
 * SKYNET M2b / subsystem: model/forward
 */
#ifndef SK_FORWARD_H
#define SK_FORWARD_H

#include "sk_types.h"
#include "sk_tensor.h"

/* Bytes of KV cache this model needs: per layer, k and v as int8 plus one
   Q16.16 scale per position. */
sk_u32 sk_ctx_kv_bytes(const sk_model *m);

typedef struct {
    sk_model *m;
    sk_pool   kv;                    /* EMS on DOS, flat buffer on host */
    int       pos;                   /* next free position              */
    int       ready;
} sk_ctx;

/* Open the context.  `kv_storage` backs the cache when EMS is not used. */
int  sk_ctx_open(sk_ctx *c, sk_model *m, void *kv_storage, sk_u32 kv_size);
void sk_ctx_close(sk_ctx *c);

/* Feed one letter at position c->pos and return the score for every
   letter of the vocabulary.  `scores` must hold SK_T_VOCAB values.
   Advances c->pos. */
int  sk_forward_step(sk_ctx *c, int tok, sk_fix *scores);

/* Reset the context so a new sequence can start at position 0. */
void sk_ctx_reset(sk_ctx *c);

/* Checks that need no model file. */
int  sk_forward_selftest(void);

#endif /* SK_FORWARD_H */
