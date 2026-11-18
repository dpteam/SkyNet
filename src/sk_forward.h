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
#include "sk_math.h"
#include "sk_tensor.h"

/* Bytes of KV cache this model needs: per layer, k and v as int8 plus one
   Q16.16 scale per position. */
sk_u32 sk_ctx_kv_bytes(const sk_model *m);

typedef struct {
    sk_model *m;
    sk_pool   kv;                    /* EMS on DOS, flat buffer on host */
    int       pos;                   /* next free position              */
    int       ready;
    /* Peak magnitude seen in the residual stream during the last step.
       The model's activations reach about 2900, and Q16.16 tops out at
       32768, so this is the number that says whether the fixed-point
       representation is still holding. It is reported, not guessed at. */
    sk_fix    peak;
    sk_fix    hf[SK_T_MAXD];   /* final normed state of the last step */
    int       saturated;             /* any value hit the Q16.16 ceiling */
} sk_ctx;

/* Open the context.  `kv_storage` backs the cache when EMS is not used. */
int  sk_ctx_open(sk_ctx *c, sk_model *m, void *kv_storage, sk_u32 kv_size);
void sk_ctx_close(sk_ctx *c);

/* Feed one letter at position c->pos and return the score for every
   letter of the vocabulary.  `scores` must hold SK_T_VOCAB values.
   Advances c->pos. */
int  sk_forward_step(sk_ctx *c, int tok, sk_fix *scores);

/* The final normed hidden state produced by the last sk_forward_step.
   The trainer needs it to compute the output-projection gradient, and
   reconstructing it outside would mean duplicating the forward pass. */
int sk_ctx_hidden(sk_ctx *c, sk_fix *out);

/* Reset the context so a new sequence can start at position 0. */
void sk_ctx_reset(sk_ctx *c);

/* ---- decoding --------------------------------------------------------
   Mirrors TINY.C's pick():
     creativity 0  -> always the best letter (deterministic)
     creativity >0 -> sample among letters within exp(diff/t) >= 0.15,
                      t = creativity/5, using the caller's RNG so the
                      sequence is reproducible from a seed.
   Generation stops at a token <= 1, which is the model's newline, or when
   the answer buffer is full. */
int sk_generate(sk_ctx *c, const char *question, char *out, int outmax,
                int creativity, sk_rng *rng);

/* Checks that need no model file. */
int  sk_forward_selftest(void);

#endif /* SK_FORWARD_H */
