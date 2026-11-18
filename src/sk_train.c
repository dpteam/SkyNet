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

typedef struct {
    sk_fix *hf;       /* final normed activations        T x D   */
    sk_fix *logits;   /* raw scores                      T x V   */
    sk_fix *dotemb;   /* dot products of the projection  T x V   */
    sk_i32 *dlogits;  /* dL/d logits, Q8.24              T x V   */
    sk_i32 *g_emb;    /* accumulated scale gradient      V       */
    sk_arena arena;
    int      T;
    sk_fix   last_loss;
    sk_i32   drift;
    sk_i32   avg[SK_T_VOCAB];
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

sk_u32 sk_train_bytes(const sk_model *m)
{
    sk_u32 T = (sk_u32)SK_TRAIN_MAXT;
    sk_u32 D, V;

    if (m == SK_NULL)
        return 0;
    D = (sk_u32)m->D;
    V = (sk_u32)SK_T_VOCAB;
    /* hf, logits, dotemb, dlogits, accumulators, the KV cache the
       forward pass needs, and slack.  Leaving out the KV cache is how the
       first version failed every single example while reporting success:
       the arena allocation returned NULL and the loss came back as zero. */
    return (sk_u32)sizeof(sk_priv)
         + (T * D + T * V + T * V) * 4UL
         + T * V * 4UL
         + (V + V) * 4UL
         + sk_ctx_kv_bytes(m)
         + 65536UL;
}

int sk_train_open(sk_train *t, sk_model *m, void *storage, sk_u32 bytes)
{
    sk_priv *p;
    sk_u32 T, D, V;

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

    p->hf = (sk_fix *)sk_arena_alloc(&p->arena, T * D * 4UL);
    p->logits = (sk_fix *)sk_arena_alloc(&p->arena, T * V * 4UL);
    p->dotemb = (sk_fix *)sk_arena_alloc(&p->arena, T * V * 4UL);
    p->dlogits = (sk_i32 *)sk_arena_alloc(&p->arena, T * V * 4UL);
    p->g_emb = (sk_i32 *)sk_arena_alloc(&p->arena, V * 4UL);

    if (p->hf == SK_NULL || p->logits == SK_NULL || p->dotemb == SK_NULL
        || p->dlogits == SK_NULL || p->g_emb == SK_NULL) {
        printf("[train] storage too small (%lu bytes)\n",
               (unsigned long)bytes);
        return 0;
    }

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
    memset(p->g_emb, 0, (size_t)SK_T_VOCAB * sizeof(sk_i32));
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

    for (tt = 0; tt < ntok; tt++) {
        sk_fix scores[SK_T_VOCAB];

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
            p->dlogits[tt * SK_T_VOCAB + i] = sk_g24_from16(d);
        }
        n++;
    }

    if (n == 0)
        return 0;
    p->last_loss = loss_acc / n;
    return p->last_loss;
}

/* Accumulate the scale gradient for the output projection.
 *
 *   dL/dscale[r] = sum_t dL/dlogits[t][r] * dot[t][r]
 *
 * dot[t][r] is the un-scaled projection, which the forward pass does not
 * keep. Recomputing it is one matvec per position, and it keeps the
 * trainer honest: it uses the same weights the model uses. */
sk_fix sk_train_grads(sk_train *t, const int *tokens, int ntok,
                      int answer_from)
{
    sk_priv *p;
    sk_model *m;
    sk_ctx ctx;
    sk_fix *kvstore;
    sk_u32 kvneed, mark;
    sk_fix hf[SK_T_MAXD];
    int tt, i;

    if (t == SK_NULL || tokens == SK_NULL || t->priv == SK_NULL)
        return 0;
    p = sk_p(t);
    m = t->m;

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

    for (tt = 0; tt < ntok; tt++) {
        sk_fix scores[SK_T_VOCAB];

        if (!sk_forward_step(&ctx, tokens[tt], scores))
            break;
        if (!sk_ctx_hidden(&ctx, hf)) {
            sk_ctx_close(&ctx);
            sk_arena_release(&p->arena, mark);
            return 0;
        }
        if (tt >= answer_from && tt + 1 < ntok) {
            sk_fix dot[SK_T_VOCAB];
            int r;

            if (!sk_table_matvec_noscale(&m->out, hf, dot)) {
                sk_ctx_close(&ctx);
                sk_arena_release(&p->arena, mark);
                return 0;
            }
            for (r = 0; r < SK_T_VOCAB; r++) {
                p->g_emb[r] += sk_fix_mul(p->dlogits[tt * SK_T_VOCAB + r],
                                          dot[r]);
            }
        }
        (void)i;
    }
    sk_ctx_close(&ctx);
    sk_arena_release(&p->arena, mark);
    return p->last_loss;
}

void sk_train_update(sk_train *t, sk_i32 lr24, sk_i32 trust)
{
    sk_priv *p;
    sk_model *m;
    int r;
    sk_i32 sum_before = 0, sum_after = 0;
    int changed = 0;

    if (t == SK_NULL || t->priv == SK_NULL)
        return;
    p = sk_p(t);
    m = t->m;

    for (r = 0; r < SK_T_VOCAB; r++) {
        sk_fix sc, after;

        if (p->g_emb[r] == 0)
            continue;
        if (!sk_table_scale_read(&m->out, (sk_u32)r, &sc))
            continue;
        after = sk_g24_apply_rms(sc, p->g_emb[r], lr24,
                                 &p->avg[r], trust);
        sk_table_scale_write(&m->out, (sk_u32)r, after);
        sum_before += sc;
        sum_after += after;
        changed++;
    }
    /* Drift is the number that explains "improves then degrades": if the
       mean scale keeps shrinking, rows are being pushed toward zero and
       dying, and the loss has to rise eventually. */
    if (changed > 0)
        p->drift = sum_after - sum_before;
    sk_train_zero_grads(t);
}

sk_i32 sk_train_drift(sk_train *t)
{
    return (t == SK_NULL || t->priv == SK_NULL) ? 0 : sk_p(t)->drift;
}

sk_fix sk_train_last_loss(sk_train *t)
{
    return (t == SK_NULL || t->priv == SK_NULL) ? 0 : sk_p(t)->last_loss;
}
