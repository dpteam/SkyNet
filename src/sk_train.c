/* sk_train.c - gradient training for the output projection scales.
 *
 * SCOPE, and why it is this narrow.
 *
 * The output projection is the last operation in the network:
 *
 *     logits[t] = Wemb * hf[t]          hf = norm(x_final, gf)
 *
 * so its scale gradient needs no backpropagation at all:
 *
 *     dL/dscale[r] = sum_t dL/dlogits[t][r] * dot[t][r]
 *
 * Every number below is therefore an EXACT derivative, not an
 * approximation. That matters more than breadth: a trainer that updates
 * parameters with approximate gradients is a trainer that quietly does not
 * learn, and the usual response is to blame the learning rate.
 *
 * The embedding table is also the output projection (weight tying), so
 * training it touches both the input representation and the answer
 * distribution. Full backpropagation through attention is written out in
 * docs/06-training.md as the next step; it is standard, but a sign error
 * there costs a session, so it is not being rushed in unverified.
 *
 * SKYNET M4 / subsystem: learn/train
 */
#include <string.h>
#include <stdio.h>
#include "sk_train.h"
#include "sk_grad.h"
#include "sk_math.h"
#include "sk_forward.h"
#include "sk_arena.h"

/* ---------------------------------------------------------------------
 * Gradient gain: every gradient in this file carries a factor of
 * SK_GRAD_GAIN, removed again in the accessors.
 *
 * Why, measured rather than assumed. On "how many senses do people have" the
 * model answers with probability 0.9988, so dlogits[target] is
 *
 *     (0.9988 - 1.0) = -0.0012,  which is -80 in Q16.16.
 *
 * The next step is dyr = dy * scale, with the row scale near 0.005:
 *
 *     0.0012 * 0.005 = 6e-6,  which is 0.4 in Q16.16 - it truncates to ZERO.
 *
 * So the entire activation gradient below a confident prediction is exactly
 * zero while the true gradient is merely small. Measured consequence: dwcheck
 * failed with "every weight gradient is zero" on a single confident position,
 * and on longer sequences most of dW came out zero.
 *
 * Multiplying dlogits by 64 buys six bits for everything below it: 0.4
 * becomes 25, and 25 is a number. The gain is divided out in the accessors,
 * so every caller sees the gradient it expects.
 *
 * Why 64 and not 256: dyr saturates at SK_FIX_MAX. dy is at most 65536 (a
 * target probability of zero) times the gain, times a scale near 0.005. At 64
 * that is 21 000, inside the 32 768 ceiling; at 256 it would reach 84 000 and
 * clamp, turning a small gradient into a large wrong one.
 *
 * Which accumulators carry it, because they are not all the same:
 *
 *   g_emb, g_f1, g_f2   dscale = dy*dot          RAW. dot is large, the
 *                                                product already fills the
 *                                                accumulator, and 64x on top
 *                                                saturates it.
 *   g_gf, g_g2          the norm gain gradient   GAINED, they come out of
 *   dw_f1, dw_f2        and dW = dy*x             the gained activation path.
 *
 * The split is a consequence of where the bits are lost, not a preference:
 * the activation path multiplies by a row scale near 0.005 and that is what
 * underflows. Every accessor divides by the gain for the kinds that carry
 * it, and sk_train_dw_get does the same, so no caller sees it.
 * ------------------------------------------------------------------- */
#define SK_GRAD_SHIFT 6
#define SK_GRAD_GAIN  ((sk_i32)1 << SK_GRAD_SHIFT)

typedef struct {
    sk_fix *hf;       /* final normed activations        T x D   */
    sk_fix *logits;   /* raw scores                      T x V   */
    sk_fix *dotemb;   /* dot products of the projection  T x V   */
    sk_i32 *dlogits;  /* dL/d logits, Q8.24              T x V   */
    sk_i32 *g_emb;    /* accumulated scale gradient      V       */
    sk_i32 *g_gf;     /* final gains                    D       */
    sk_i32 *g_g2;     /* last block g2 gains            D       */
    sk_i32 *g_f1;     /* last block f1 row scales       HID     */
    sk_i32 *g_f2;     /* last block f2 row scales       D       */

    sk_fix *t_mid, *t_dotf1, *t_dotf2, *t_xff, *t_xf, *t_dotout;
    sk_fix *t_hn2;
    sk_fix *t_inv2, *t_invf;
    sk_arena arena;
    int      T;
    sk_fix   last_loss;
    sk_i32   drift;
    int      grad_count;
    sk_i32   avg[SK_T_VOCAB];     /* RMSProp running |grad|, one per row */
    sk_i32  *avg_f1;              /* last block f1 rows                HID */
    sk_i32  *avg_f2;              /* last block f2 rows                  D */
    sk_i32  *avg_gf;              /* final norm gains                    D */
    sk_i32  *avg_g2;              /* last block norm gains               D */

    /* Per-weight gradients for the last block's feed-forward tables.
     * Row-major dL/dW_eff in Q16.16: the SAME quantity PyTorch calls
     * weight.grad, so it can be compared with the reference trainer
     * directly instead of through a scaled proxy. NULL unless
     * SK_TRAIN_CODES was requested. */
    sk_i32  *dw_f1;               /* HID x D */
    sk_i32  *dw_f2;               /* D x HID */
} sk_priv;

static sk_priv *sk_p(sk_train *t)
{
    return (sk_priv *)t->priv;
}

static void sk_tr_softmax(sk_fix *v, int n)
{
    sk_fix mx, sum;
    int i;

    mx = v[0];
    for (i = 1; i < n; i++)
        if (v[i] > mx)
            mx = v[i];
    sum = 0;
    for (i = 0; i < n; i++) {
        v[i] = sk_fix_exp(v[i] - mx);
        sum += v[i];
    }
    if (sum <= 0) {
        for (i = 0; i < n; i++)
            v[i] = (i == 0) ? SK_FIX_ONE : 0;
        return;
    }
    for (i = 0; i < n; i++)
        v[i] = sk_fix_div(v[i], sum);
}

sk_u32 sk_train_bytes(const sk_model *m, int flags)
{
    sk_u32 T = (sk_u32)SK_TRAIN_MAXT;
    sk_u32 D, V, HID, F;
    sk_u32 total;

    if (m == SK_NULL)
        return 0;
    D = (sk_u32)m->D;
    V = (sk_u32)SK_T_VOCAB;
    HID = (sk_u32)m->HID;

    /* Every buffer sk_train_open allocates, counted here. Leaving one out
       is how this failed twice: first the KV cache, then the trace
       buffers. Both times the arena ran dry inside sk_tr_run, the forward
       pass returned 0, and every gradient came back as zero - which reads
       like "the gradient is broken", not "there is no memory". */
    total = (sk_u32)sizeof(sk_priv);

    /* sk_fix buffers */
    F = T * D * 2UL          /* hf, logits are sk_fix */
      + T * V * 2UL          /* logits, dotemb */
      + T * D * 2UL          /* t_hn2, t_xf */
      + T * D * 2UL          /* t_xff, t_dotf2 */
      + T * HID * 2UL        /* t_mid, t_dotf1 */
      + T * V                /* t_dotout */
      + T * 2UL;             /* t_inv2, t_invf */

    /* sk_i32 buffers */
    total += F * 4UL;
    total += (T * V) * 4UL;  /* dlogits */
    if (flags & SK_TRAIN_CODES) {
        /* Both feed-forward tables of the last block. Counted here rather
           than in sk_train_open so the caller's allocation and the
           trainer's appetite cannot disagree - that mismatch is how the
           trace buffers once ran the arena dry and made every gradient read
           as zero. */
        total += (HID * D + D * HID) * 4UL;
    }
    total += (V + D * 3UL + HID + D) * 4UL;  /* gradient accumulators */
    total += (HID + D * 4UL) * 4UL;          /* running |grad| per row */
    total += sk_ctx_kv_bytes(m);

    return total + 131072UL;
}

int sk_train_open(sk_train *t, sk_model *m, void *storage, sk_u32 bytes,
                  int flags)
{
    sk_priv *p;
    sk_u32 T, D, V, HID;

    if (t == SK_NULL || m == SK_NULL || !m->loaded || storage == SK_NULL)
        return 0;
    memset(t, 0, sizeof(*t));
    t->m = m;

    if (bytes < (sk_u32)sizeof(sk_priv))
        return 0;
    p = (sk_priv *)storage;
    memset(p, 0, sizeof(*p));
    if (!sk_arena_attach(&p->arena, (sk_u8 *)storage + sizeof(sk_priv),
                         bytes - (sk_u32)sizeof(sk_priv)))
        return 0;

    T = (sk_u32)SK_TRAIN_MAXT;
    D = (sk_u32)m->D;
    V = (sk_u32)SK_T_VOCAB;
    HID = (sk_u32)m->HID;

    p->hf = (sk_fix *)sk_arena_alloc(&p->arena, T * D * 4UL);
    p->logits = (sk_fix *)sk_arena_alloc(&p->arena, T * V * 4UL);
    p->dotemb = (sk_fix *)sk_arena_alloc(&p->arena, T * V * 4UL);
    p->dlogits = (sk_i32 *)sk_arena_alloc(&p->arena, T * V * 4UL);
    p->g_emb = (sk_i32 *)sk_arena_alloc(&p->arena, V * 4UL);
    p->g_gf = (sk_i32 *)sk_arena_alloc(&p->arena, D * 4UL);
    p->g_g2 = (sk_i32 *)sk_arena_alloc(&p->arena, D * 4UL);
    p->g_f1 = (sk_i32 *)sk_arena_alloc(&p->arena, HID * 4UL);
    p->g_f2 = (sk_i32 *)sk_arena_alloc(&p->arena, D * 4UL);
    p->avg_f1 = (sk_i32 *)sk_arena_alloc(&p->arena, HID * 4UL);
    p->avg_f2 = (sk_i32 *)sk_arena_alloc(&p->arena, D * 4UL);
    p->avg_gf = (sk_i32 *)sk_arena_alloc(&p->arena, D * 4UL);
    p->avg_g2 = (sk_i32 *)sk_arena_alloc(&p->arena, D * 4UL);
    if (flags & SK_TRAIN_CODES) {
        p->dw_f1 = (sk_i32 *)sk_arena_alloc(&p->arena, HID * D * 4UL);
        p->dw_f2 = (sk_i32 *)sk_arena_alloc(&p->arena, D * HID * 4UL);
        if (p->dw_f1 == SK_NULL || p->dw_f2 == SK_NULL) {
            printf("[train] no room for the per-weight gradients "
                   "(%lu bytes asked for)\n", (unsigned long)bytes);
            return 0;
        }
        memset(p->dw_f1, 0, (size_t)HID * D * sizeof(sk_i32));
        memset(p->dw_f2, 0, (size_t)D * HID * sizeof(sk_i32));
    }

    p->t_hn2 = (sk_fix *)sk_arena_alloc(&p->arena, T * D * 4UL);
    p->t_mid = (sk_fix *)sk_arena_alloc(&p->arena, T * HID * 4UL);
    p->t_dotf1 = (sk_fix *)sk_arena_alloc(&p->arena, T * HID * 4UL);
    p->t_dotf2 = (sk_fix *)sk_arena_alloc(&p->arena, T * D * 4UL);
    p->t_xff = (sk_fix *)sk_arena_alloc(&p->arena, T * D * 4UL);
    p->t_xf = (sk_fix *)sk_arena_alloc(&p->arena, T * D * 4UL);
    p->t_dotout = (sk_fix *)sk_arena_alloc(&p->arena, T * V * 4UL);
    p->t_inv2 = (sk_fix *)sk_arena_alloc(&p->arena, T * 4UL);
    p->t_invf = (sk_fix *)sk_arena_alloc(&p->arena, T * 4UL);

    /* Every buffer, checked together. Listing only the first few here let
       the trace buffers fail silently when sk_train_bytes under-counted:
       the forward then returned 0 inside sk_tr_run and every gradient came
       back as zero, which read as "the backward pass is broken". */
    if (p->hf == SK_NULL || p->logits == SK_NULL || p->dotemb == SK_NULL
        || p->dlogits == SK_NULL || p->g_emb == SK_NULL
        || p->g_gf == SK_NULL || p->g_g2 == SK_NULL || p->g_f1 == SK_NULL
        || p->g_f2 == SK_NULL || p->avg_f1 == SK_NULL || p->avg_f2 == SK_NULL
        || p->avg_gf == SK_NULL || p->avg_g2 == SK_NULL
        || p->t_hn2 == SK_NULL || p->t_mid == SK_NULL
        || p->t_dotf1 == SK_NULL || p->t_dotf2 == SK_NULL
        || p->t_xff == SK_NULL || p->t_xf == SK_NULL
        || p->t_dotout == SK_NULL || p->t_inv2 == SK_NULL
        || p->t_invf == SK_NULL) {
        printf("[train] storage too small (%lu bytes)\n",
               (unsigned long)bytes);
        return 0;
    }

    /* Optimiser state, zeroed once. The arena does not zero memory, and a
       garbage running average makes the first steps wild. */
    memset(p->avg, 0, sizeof(p->avg));
    memset(p->avg_f1, 0, (size_t)HID * sizeof(sk_i32));
    memset(p->avg_f2, 0, (size_t)D * sizeof(sk_i32));
    memset(p->avg_gf, 0, (size_t)D * sizeof(sk_i32));
    memset(p->avg_g2, 0, (size_t)D * sizeof(sk_i32));

    t->priv = p;
    return 1;
}

void sk_train_close(sk_train *t)
{
    if (t != SK_NULL)
        memset(t, 0, sizeof(*t));
}

void sk_train_zero_grads(sk_train *t)
{
    sk_priv *p;

    if (t == SK_NULL || t->priv == SK_NULL)
        return;
    p = sk_p(t);

    /* Every accumulator, not just the first one.
     *
     * This is worth a comment because the omission was invisible for two
     * rounds of debugging: the arena does not zero memory, so the four
     * accumulators added later started life as whatever malloc returned.
     * The output projection still matched numerically - its accumulator was
     * the one being cleared - while everything downstream of it reported
     * sign-flipped gradients and relative errors of 10^10. Garbage, not
     * algebra. */
    memset(p->g_emb, 0, (size_t)SK_T_VOCAB * sizeof(sk_i32));
    memset(p->g_gf, 0, (size_t)t->m->D * sizeof(sk_i32));
    memset(p->g_g2, 0, (size_t)t->m->D * sizeof(sk_i32));
    memset(p->g_f1, 0, (size_t)t->m->HID * sizeof(sk_i32));
    memset(p->g_f2, 0, (size_t)t->m->D * sizeof(sk_i32));
    if (p->dw_f1 != SK_NULL)
        memset(p->dw_f1, 0,
               (size_t)t->m->HID * (size_t)t->m->D * sizeof(sk_i32));
    if (p->dw_f2 != SK_NULL)
        memset(p->dw_f2, 0,
               (size_t)t->m->D * (size_t)t->m->HID * sizeof(sk_i32));
    p->grad_count = 0;

    /* NOT the running averages: they are optimiser state and have to
       survive the update that clears the gradients. Resetting them here
       would make every step start from a zero average, which turns RMSProp
       into a raw gradient step and loses the whole point of it. */
}

/* Run the network over the sequence and keep hf and logits.
 *
 * The activations come from the same forward pass the inference path uses,
 * driven one token at a time, so the trainer cannot drift away from what
 * the program actually computes. That is worth the small cost of not
 * batching: a trainer that optimises a slightly different function than the
 * one being deployed is a classic way to get "training works, production
 * does not". */
static int sk_tr_run(sk_train *t, const int *tok, int ntok)
{
    sk_priv *p = sk_p(t);
    sk_model *m = t->m;
    sk_ctx ctx;
    sk_fix *kvstore;
    sk_u32 kvneed, mark;
    int tt, i;

    /* The KV cache is scratch: take a mark and release it, or twenty
       examples exhaust the arena and every later sequence silently fails. */
    mark = sk_arena_mark(&p->arena);
    kvneed = sk_ctx_kv_bytes(m);
    kvstore = (sk_fix *)sk_arena_alloc(&p->arena, kvneed);
    if (kvstore == SK_NULL) {
        sk_arena_release(&p->arena, mark);
        return 0;
    }
    if (!sk_ctx_open(&ctx, m, kvstore, kvneed)) {
        sk_arena_release(&p->arena, mark);
        return 0;
    }

    /* Point the optional trace at the last block. During inference these
       are NULL and the forward pass skips all of it. */
    ctx.tr_layer = m->L - 1;

    for (tt = 0; tt < ntok; tt++) {
        sk_fix scores[SK_T_VOCAB];

        ctx.tr_hn2 = p->t_hn2 + tt * m->D;
        ctx.tr_mid = p->t_mid + tt * m->HID;
        ctx.tr_dotf1 = p->t_dotf1 + tt * m->HID;
        ctx.tr_dotf2 = p->t_dotf2 + tt * m->D;
        ctx.tr_xff = p->t_xff + tt * m->D;

        if (!sk_forward_step(&ctx, tok[tt], scores)) {
            sk_ctx_close(&ctx);
            sk_arena_release(&p->arena, mark);
            return 0;
        }
        /* sk_forward_step advanced the position, so the scores it produced
           belong to position tt. Keep them, and keep the hidden state by
           re-running the final projection input: the context does not
           expose hf, so it is reconstructed from the same tables. */
        for (i = 0; i < SK_T_VOCAB; i++)
            p->logits[tt * SK_T_VOCAB + i] = scores[i];

        p->t_inv2[tt] = ctx.tr_inv2;
        p->t_invf[tt] = ctx.tr_invf;
        memcpy(p->t_xf + tt * m->D, ctx.tr_xf,
               (size_t)m->D * sizeof(sk_fix));

        /* The output-projection dot products: dL/dscale is the incoming
           gradient times the un-scaled projection. One extra matvec per
           position is cheaper than plumbing another trace through the
           forward pass. */
        {
            sk_fix scaled[SK_T_VOCAB];

            if (!sk_table_matvec_dot(&m->out, ctx.hf, scaled,
                                     p->t_dotout + tt * SK_T_VOCAB)) {
                sk_ctx_close(&ctx);
                sk_arena_release(&p->arena, mark);
                return 0;
            }
        }
    }
    sk_ctx_close(&ctx);
    sk_arena_release(&p->arena, mark);
    p->T = ntok;
    return 1;
}

sk_fix sk_train_sequence(sk_train *t, const int *tokens, int ntok,
                         int answer_from)
{
    sk_priv *p;
    int T, tt, i, n;
    sk_fix loss_acc;

    if (t == SK_NULL || tokens == SK_NULL || t->priv == SK_NULL)
        return 0;
    p = sk_p(t);

    if (ntok < 2 || ntok > SK_TRAIN_MAXT || answer_from < 0
        || answer_from >= ntok - 1)
        return 0;

    if (!sk_tr_run(t, tokens, ntok))
        return 0;

    T = ntok;
    loss_acc = 0;
    n = 0;
    memset(p->dlogits, 0, (size_t)T * (size_t)SK_T_VOCAB * sizeof(sk_i32));

    for (tt = answer_from; tt + 1 < T; tt++) {
        sk_fix *lg = p->logits + tt * SK_T_VOCAB;
        int target = tokens[tt + 1];
        sk_fix prob;
        sk_fix lg2[SK_T_VOCAB];

        memcpy(lg2, lg, sizeof(lg2));
        sk_tr_softmax(lg2, SK_T_VOCAB);
        prob = lg2[target];
        if (prob < 1)
            prob = 1;
        /* Keep the fractional part: truncating to an integer made every
           loss read 1.0000 and hid whether training did anything at all. */
        loss_acc += -sk_log_fix(prob);

        for (i = 0; i < SK_T_VOCAB; i++) {
            sk_i32 d = (i == target) ? lg2[i] - SK_FIX_ONE : lg2[i];

            /* Q16.16, not Q8.24.
             *
             * This was the reason a whole rewrite of the backward pass
             * failed the checker on the path that already worked: dlogits
             * kept Q8.24 while every other gradient was Q16.16, so the
             * product with the projection dot was off by 256 and the
             * "regression" from 97/97 to 88/97 looked like a rounding
             * problem rather than a unit mismatch.
             *
             * Q8.24 was chosen here originally because a softmax value is
             * at most 1 and Q16.16 holds only 15 bits of it. But d reaches
             * -1 (the target term), and Q16.16 holds that exactly; the
             * gradient it feeds is a product with dot, which is large
             * enough that 15 bits of fraction is plenty. One convention for
             * the whole file beats precision that is not needed. */
            /* The gain goes on here, at the only place a gradient is
               created. Everything downstream inherits it and the accessors
               remove it, so there is one knob and it cannot get out of
               step. */
            p->dlogits[tt * SK_T_VOCAB + i] = d * SK_GRAD_GAIN;
        }
        n++;
    }

    if (n == 0)
        return 0;
    p->last_loss = loss_acc / n;
    return p->last_loss;
}

/* ---------------------------------------------------------------------
 * Gradient convention: everything is Q16.16, everywhere, with no exceptions.
 *
 * Getting this uniform is what made the second attempt at this function
 * work. The first kept dlogits in Q8.24 while every other gradient was
 * Q16.16, so the product with the projection dot was 256x off, and the
 * numerical checker reported it as 9 rows losing accuracy on the path that
 * already worked - which looked like a rounding problem and was a unit
 * mismatch.
 *
 * Q16.16 rather than Q8.24 because the gradient reaches ~283 (dL/dlogit
 * times a dot product that reaches ~2000) while Q8.24 saturates at 128.
 * The precision that Q8.24 would buy is recovered later, in the updater,
 * by normalising the gradient against its own running average before
 * converting to Q8.24 for the multiply by the learning rate.
 * ------------------------------------------------------------------- */

/* dx += W^T * dy with the row scale folded in; dscale[i] += dy[i]*dot[i];
 * dW[i][j] += dy[i]*x[j].
 *
 * The dW term is the whole of training the weights, and it costs one
 * multiply per element: dL/dW_eff[i][j] is the row's incoming gradient times
 * the input it was multiplied by. Nothing new has to be derived, which is
 * why adding it here is a handful of lines rather than another pass over the
 * network.
 *
 * Note what it is NOT: dL/dcode. The stored value is an integer code and
 * W_eff = scale * code, so dL/dcode = scale * dL/dW_eff. The caller applies
 * the scale, because only it knows whether it is updating the code or the
 * scale.
 *
 * `x` is the vector this table was applied to (the input), and `dw` is
 * row-major rows x cols. Either may be NULL, and then dW is skipped. */
static int sk_tr_back(const sk_table *tab, const sk_i32 *dy, sk_i32 *dx,
                      sk_i32 *dscale, const sk_fix *dot,
                      sk_i32 *dw, const sk_fix *x)
{
    signed char w[SK_T_MAXHID];
    sk_u32 i, j;
    sk_fix scale;

    for (i = 0; i < tab->rows; i++) {
        sk_i32 dyr;

        if (!sk_table_scale_read(tab, i, &scale))
            return 0;
        /* dscale is stored WITHOUT the gain, and the division is exact:
           the gain is a power of two, so dy/GAIN is the original integer.
           This is not a detail. Keeping the gain here saturated the
           accumulator - dy reaches 65536, dot reaches 5.7e7 for the output
           projection, and their product already sits at 868 in Q16.16;
           times 64 it exceeds the 32768 ceiling and two rows of the
           numerical check went from agreeing to 364% off. */
        if (dscale != SK_NULL && dot != SK_NULL)
            dscale[i] += sk_fix_mul(dy[i] / SK_GRAD_GAIN, dot[i]);

        /* Before the early exit below: a row whose dy rounds to zero in the
           activation path still contributes to dW when dy itself is not
           zero, and skipping it would quietly bias the weight gradient. */
        if (dw != SK_NULL && x != SK_NULL && dy[i] != 0) {
            sk_i32 *row = dw + (size_t)i * (size_t)tab->cols;

            for (j = 0; j < tab->cols; j++)
                if (x[j] != 0)
                    row[j] += sk_fix_mul(dy[i], x[j]);
        }

        dyr = sk_scale_fix(dy[i], scale);
        if (dyr == 0)
            continue;
        if (!sk_table_read_row(tab, i, w))
            return 0;
        for (j = 0; j < tab->cols; j++) {
            if (w[j] != 0)
                dx[j] += dyr * (sk_i32)w[j];
        }
    }
    return 1;
}

/* Backward through the RMS norm: dy -> dx and the gain gradient.
 *   y[j]  = x[j] * inv * g[j]
 *   dx[j] = inv*g[j]*dy[j] - (inv^3/n) * x[j] * sum_k dy[k]*x[k]*g[k]
 *   dg[j] = dy[j] * x[j] * inv
 */
static void sk_tr_norm_back(const sk_fix *x, const sk_fix *g, sk_fix inv,
                            const sk_i32 *dy, sk_i32 *dx, sk_i32 *dg, int n)
{
    sk_i32 dldinv, inv3;
    int i;

    if (inv == 0)
        return;

    dldinv = 0;
    for (i = 0; i < n; i++)
        dldinv += sk_fix_mul(sk_fix_mul(dy[i], x[i]), g[i]);

    inv3 = sk_fix_mul(sk_fix_mul(inv, inv), inv);
    inv3 = -sk_fix_div(inv3, sk_fix_from_int(n));

    for (i = 0; i < n; i++) {
        sk_i32 t1 = sk_fix_mul(dy[i], sk_fix_mul(inv, g[i]));
        sk_i32 t2 = sk_fix_mul(sk_fix_mul(dldinv, inv3), x[i]);

        dx[i] += t1 + t2;
        if (dg != SK_NULL)
            dg[i] += sk_fix_mul(dy[i], sk_fix_mul(x[i], inv));
    }
}

sk_fix sk_train_grads(sk_train *t, const int *tokens, int ntok,
                      int answer_from)
{
    sk_priv *p;
    sk_model *m;
    int last, tt, i;

    if (t == SK_NULL || tokens == SK_NULL || t->priv == SK_NULL)
        return 0;
    p = sk_p(t);
    m = t->m;
    if (!m->loaded || m->L < 1)
        return 0;
    last = m->L - 1;

    /* Forward with the trace on; p->dlogits was filled by
       sk_train_sequence, which must be called first. */
    if (!sk_tr_run(t, tokens, ntok))
        return 0;

    for (tt = answer_from; tt + 1 < ntok; tt++) {
        sk_fix gf[SK_T_MAXD], g2[SK_T_MAXD];
        sk_i32 dhf[SK_T_MAXD];
        sk_i32 dxff[SK_T_MAXD];
        sk_i32 dmid[SK_T_MAXHID];
        sk_i32 dhn2[SK_T_MAXD];

        if (!sk_model_gain_row(m, SK_GAIN_GF, 0, gf)
            || !sk_model_gain_row(m, SK_GAIN_G2, last, g2))
            return 0;

        /* ---- output projection: dL/dhf ---------------------------- */
        memset(dhf, 0, (size_t)m->D * sizeof(sk_i32));
        /* No dW here: the projection's per-weight gradient is
           dlogits[r] * hf[j], which is the same formula, but the scales of
           this table are already trained and its weights are the input
           embedding - one tensor, two jobs. Left alone until weight tying is
           revisited. */
        if (!sk_tr_back(&m->out, p->dlogits + tt * SK_T_VOCAB, dhf,
                        p->g_emb, p->t_dotout + tt * SK_T_VOCAB,
                        SK_NULL, SK_NULL))
            return 0;

        /* ---- final norm: dL/dx after the last block ---------------- */
        memset(dxff, 0, (size_t)m->D * sizeof(sk_i32));
        sk_tr_norm_back(p->t_xf + tt * m->D, gf, p->t_invf[tt],
                        dhf, dxff, p->g_gf, m->D);

        /* ---- last block feed-forward ------------------------------
           f2 is (D x HID): dL/dmid = f2^T * dxff.
           f1 is (HID x D): dL/dhn2 = f1^T * dmid, masked by the relu. */
        memset(dmid, 0, (size_t)m->HID * sizeof(sk_i32));
        if (!sk_tr_back(&m->f2[last], dxff, dmid, p->g_f2,
                        p->t_dotf2 + tt * m->D,
                        p->dw_f2, p->t_mid + tt * m->HID))
            return 0;
        for (i = 0; i < m->HID; i++) {
            if (p->t_mid[tt * m->HID + i] <= 0)
                dmid[i] = 0;
        }

        memset(dhn2, 0, (size_t)m->D * sizeof(sk_i32));
        if (!sk_tr_back(&m->f1[last], dmid, dhn2, p->g_f1,
                        p->t_dotf1 + tt * m->HID,
                        p->dw_f1, p->t_hn2 + tt * m->D))
            return 0;

        /* ---- through the norm into the residual -------------------
           Not propagated into attention yet: that is the next step, and
           the checker will say when it arrives. */
        {
            sk_i32 dxprev[SK_T_MAXD];

            memset(dxprev, 0, (size_t)m->D * sizeof(sk_i32));
            sk_tr_norm_back(p->t_xff + tt * m->D, g2, p->t_inv2[tt],
                            dhn2, dxprev, p->g_g2, m->D);
        }

        /* One count per position, because the loss is a per-position mean.
           Dropping this made every gradient read as zero: the accessors
           divide by it, and zero is a division guard. */
        p->grad_count++;
    }

    return p->last_loss;
}

/* Mean gradient for one parameter: the loss is a mean over answer
   positions, so the accumulated sum is divided by the position count.
   Accumulating a plain sum made every gradient n times too large - the
   numerical checker showed it directly. */
/* Mean gradient for one accumulator.
 *
 * `gained` says whether this accumulator carries SK_GRAD_GAIN. Passing it
 * explicitly rather than deciding by kind means the two places that know -
 * here and sk_tr_back - are the only places the decision is made, and a
 * mismatch shows up immediately as every gradient being 64x off, which the
 * checker reports as "unresolved" rather than as a wrong number.
 *
 * Mean first, then the gain: the other order throws away six bits before the
 * division by the position count. */
static sk_i32 sk_mean_grad(const sk_priv *p, sk_i32 sum, int gained)
{
    sk_i32 m = (p->grad_count > 0) ? (sum / p->grad_count) : sum;

    return gained ? (m / SK_GRAD_GAIN) : m;
}

/* One row scale: read, step, write back, and add to the drift sums.
 *
 * All three row-scale tables (output projection, f1, f2) go through here so
 * that the step, the trust cap and the RMS normalisation cannot drift apart
 * between them - they were written three times by hand at first, and the
 * three copies disagreed about the units. */
static void sk_upd_scale_row(sk_table *tab, sk_u32 row, sk_i32 grad16,
                             sk_i32 lr24, sk_i32 *avg24, sk_i32 trust,
                             sk_i32 *sum_before, sk_i32 *sum_after)
{
    sk_fix sc, after;

    if (!sk_table_scale_read(tab, row, &sc))
        return;
    after = sk_g24_apply_rms(sc, sk_g24_from16(grad16), lr24, avg24, trust);
    if (!sk_table_scale_write(tab, row, after))
        return;
    *sum_before += sc;
    *sum_after += after;
}

/* The gains are Q16.16, the scales are Q8.24. Shifting the gain up two
   bytes lets it go through the same step; a gain above 128 cannot be
   represented that way and is left alone rather than silently wrapped. */
static sk_fix sk_upd_gain(sk_fix gain, sk_i32 grad16, sk_i32 lr24,
                          sk_i32 *avg24, sk_i32 trust)
{
    sk_i32 q24, after;

    if (gain <= 0 || gain > (SK_FIX_MAX >> 8))
        return gain;
    q24 = gain << 8;
    after = sk_g24_apply_rms(q24, sk_g24_from16(grad16), lr24, avg24, trust);
    return after >> 8;
}

void sk_train_update(sk_train *t, sk_i32 lr24, sk_i32 trust)
{
    sk_priv *p;
    sk_model *m;
    sk_fix gainrow[SK_T_MAXD];
    sk_i32 sum_before = 0, sum_after = 0;
    int r, last, changed = 0;

    if (t == SK_NULL || t->priv == SK_NULL)
        return;
    p = sk_p(t);
    m = t->m;
    if (!m->loaded || m->L < 1)
        return;
    last = m->L - 1;

    /* ---- output projection rows -------------------------------- */
    for (r = 0; r < SK_T_VOCAB; r++) {
        if (p->g_emb[r] == 0)
            continue;
        sk_upd_scale_row(&m->out, (sk_u32)r,
                         sk_mean_grad(p, p->g_emb[r], 0),
                         lr24, &p->avg[r], trust, &sum_before, &sum_after);
        changed++;
    }

    /* ---- last block feed-forward rows --------------------------- */
    for (r = 0; r < m->HID; r++) {
        if (p->g_f1[r] == 0)
            continue;
        sk_upd_scale_row(&m->f1[last], (sk_u32)r,
                         sk_mean_grad(p, p->g_f1[r], 0),
                         lr24, &p->avg_f1[r], trust,
                         &sum_before, &sum_after);
        changed++;
    }
    for (r = 0; r < m->D; r++) {
        if (p->g_f2[r] == 0)
            continue;
        sk_upd_scale_row(&m->f2[last], (sk_u32)r,
                         sk_mean_grad(p, p->g_f2[r], 0),
                         lr24, &p->avg_f2[r], trust,
                         &sum_before, &sum_after);
        changed++;
    }

    /* ---- gains -------------------------------------------------
       Q16.16, one element at a time, the row written back once. */
    if (sk_model_gain_row(m, SK_GAIN_GF, 0, gainrow)) {
        int any = 0;

        for (r = 0; r < m->D; r++) {
            if (p->g_gf[r] == 0)
                continue;
            gainrow[r] = sk_upd_gain(gainrow[r],
                                     sk_mean_grad(p, p->g_gf[r], 1),
                                     lr24, &p->avg_gf[r], trust);
            any = 1;
        }
        if (any && sk_model_gain_write(m, SK_GAIN_GF, 0, gainrow))
            changed++;
    }
    if (sk_model_gain_row(m, SK_GAIN_G2, last, gainrow)) {
        int any = 0;

        for (r = 0; r < m->D; r++) {
            if (p->g_g2[r] == 0)
                continue;
            gainrow[r] = sk_upd_gain(gainrow[r],
                                     sk_mean_grad(p, p->g_g2[r], 1),
                                     lr24, &p->avg_g2[r], trust);
            any = 1;
        }
        if (any && sk_model_gain_write(m, SK_GAIN_G2, last, gainrow))
            changed++;
    }

    /* Drift is the number that explains "improves then degrades": if the
       row scales keep shrinking, rows are being pushed toward zero and
       dying, and the loss has to rise eventually. It sums the row-scale
       tables only - the gains are Q16.16 and mixing them in would make the
       number unreadable. */
    if (changed > 0)
        p->drift = sum_after - sum_before;
    sk_train_zero_grads(t);
}

sk_i32 sk_train_grad_row(sk_train *t, int r)
{
    sk_priv *p;

    if (t == SK_NULL || t->priv == SK_NULL || r < 0 || r >= SK_T_VOCAB)
        return 0;
    p = sk_p(t);
    /* Returned RAW in Q16.16, not converted to Q8.24.
       sk_g24_from16 only handles values up to 128 (it saturates above
       SK_FIX_MAX>>8), and a large-dot row reaches 283: converting here hid
       that as a plausible-looking -128.000000. The caller decides how to
       scale. */
    /* Raw: dscale never carried the gain. */
    if (p->grad_count > 0)
        return p->g_emb[r] / p->grad_count;
    return p->g_emb[r];
}

sk_i32 sk_train_drift(sk_train *t)
{
    return (t == SK_NULL || t->priv == SK_NULL) ? 0 : sk_p(t)->drift;
}

/* ---------------------------------------------------------------------
 * Accessors for the numerical gradient checker.
 *
 * sk_train_grad_get returns the MEAN gradient, because the loss is a mean
 * over answer positions and the numeric derivative measures the mean.
 * Returning a raw sum here made the checker report every row as n times too
 * large on the very first run.
 * ------------------------------------------------------------------- */

sk_fix sk_train_dw_get(sk_train *t, int kind, int index)
{
    sk_priv *p;
    sk_model *m;
    sk_i32 n;
    sk_i32 *buf;

    if (t == SK_NULL || t->priv == SK_NULL || index < 0)
        return 0;
    p = sk_p(t);
    m = t->m;

    if (kind == SK_P_DW_F1) {
        if (index >= m->HID * m->D)
            return 0;
        buf = p->dw_f1;
        n = m->HID * m->D;
    } else if (kind == SK_P_DW_F2) {
        if (index >= m->D * m->HID)
            return 0;
        buf = p->dw_f2;
        n = m->D * m->HID;
    } else {
        return 0;
    }
    if (buf == SK_NULL)
        return 0;
    (void)n;
    /* Mean over answer positions, like every other gradient this module
       reports: the loss is a mean, so its derivative is one too. */
    if (p->grad_count > 0)
        return (buf[index] / p->grad_count) / SK_GRAD_GAIN;
    return buf[index] / SK_GRAD_GAIN;
}

sk_fix sk_train_grad_get(sk_train *t, int kind, int index)
{
    sk_priv *p;
    sk_model *m;

    if (t == SK_NULL || t->priv == SK_NULL)
        return 0;
    p = sk_p(t);
    m = t->m;

    switch (kind) {
    case SK_P_OUT:
        if (index < 0 || index >= SK_T_VOCAB)
            return 0;
        return (p->grad_count > 0) ? (p->g_emb[index] / p->grad_count)
                                   : p->g_emb[index];
    case SK_P_GF:
        if (index < 0 || index >= m->D)
            return 0;
        return (p->grad_count > 0)
               ? ((p->g_gf[index] / p->grad_count) / SK_GRAD_GAIN)
               : (p->g_gf[index] / SK_GRAD_GAIN);
    case SK_P_G2:
        if (index < 0 || index >= m->D)
            return 0;
        return (p->grad_count > 0)
               ? ((p->g_g2[index] / p->grad_count) / SK_GRAD_GAIN)
               : (p->g_g2[index] / SK_GRAD_GAIN);
    case SK_P_F1:
        if (index < 0 || index >= m->HID)
            return 0;
        return (p->grad_count > 0) ? (p->g_f1[index] / p->grad_count)
                                   : p->g_f1[index];
    case SK_P_F2:
        if (index < 0 || index >= m->D)
            return 0;
        return (p->grad_count > 0) ? (p->g_f2[index] / p->grad_count)
                                   : p->g_f2[index];
    default:
        return 0;
    }
}

int sk_train_param_get(sk_train *t, int kind, int index, sk_fix *out)
{
    sk_model *m;
    sk_fix row[SK_T_MAXD];
    int last;

    if (t == SK_NULL || t->priv == SK_NULL || out == SK_NULL)
        return 0;
    m = t->m;
    last = m->L - 1;

    switch (kind) {
    case SK_P_OUT:
        return sk_table_scale_read(&m->out, (sk_u32)index, out);
    case SK_P_F1:
        return sk_table_scale_read(&m->f1[last], (sk_u32)index, out);
    case SK_P_F2:
        return sk_table_scale_read(&m->f2[last], (sk_u32)index, out);
    case SK_P_GF:
        if (index < 0 || index >= m->D)
            return 0;
        if (!sk_model_gain_row(m, SK_GAIN_GF, 0, row))
            return 0;
        *out = row[index];
        return 1;
    case SK_P_G2:
        if (index < 0 || index >= m->D)
            return 0;
        if (!sk_model_gain_row(m, SK_GAIN_G2, last, row))
            return 0;
        *out = row[index];
        return 1;
    default:
        return 0;
    }
}

int sk_train_param_set(sk_train *t, int kind, int index, sk_fix value)
{
    sk_model *m;
    sk_fix row[SK_T_MAXD];
    int last;

    if (t == SK_NULL || t->priv == SK_NULL)
        return 0;
    m = t->m;
    last = m->L - 1;

    switch (kind) {
    case SK_P_OUT:
        return sk_table_scale_write(&m->out, (sk_u32)index, value);
    case SK_P_F1:
        return sk_table_scale_write(&m->f1[last], (sk_u32)index, value);
    case SK_P_F2:
        return sk_table_scale_write(&m->f2[last], (sk_u32)index, value);
    case SK_P_GF:
        if (index < 0 || index >= m->D)
            return 0;
        if (!sk_model_gain_row(m, SK_GAIN_GF, 0, row))
            return 0;
        row[index] = value;
        return sk_model_gain_write(m, SK_GAIN_GF, 0, row);
    case SK_P_G2:
        if (index < 0 || index >= m->D)
            return 0;
        if (!sk_model_gain_row(m, SK_GAIN_G2, last, row))
            return 0;
        row[index] = value;
        return sk_model_gain_write(m, SK_GAIN_G2, last, row);
    default:
        return 0;
    }
}

sk_fix sk_train_last_loss(sk_train *t)
{
    return (t == SK_NULL || t->priv == SK_NULL) ? 0 : sk_p(t)->last_loss;
}
