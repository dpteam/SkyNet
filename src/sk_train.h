/* sk_train.h - training for the output projection scales.
 *
 * Scope is deliberately narrow and documented in sk_train.c: the gradient
 * of the output projection is exact without a backward pass through the
 * network, so every number this module produces is a true derivative.
 *
 * SKYNET M4 / subsystem: learn/train
 */
#ifndef SK_TRAIN_H
#define SK_TRAIN_H

#include "sk_types.h"
#include "sk_tensor.h"

/* Longest sequence the trainer accepts. */
#define SK_TRAIN_MAXT 64

typedef struct {
    sk_model *m;
    void     *priv;      /* private state, owned by sk_train.c */
} sk_train;

/* Bytes of storage a trainer for this model needs. */
sk_u32 sk_train_bytes(const sk_model *m);

/* Open a trainer.  `storage` must be at least sk_train_bytes(m) long. */
int  sk_train_open(sk_train *t, sk_model *m, void *storage, sk_u32 bytes);
void sk_train_close(sk_train *t);

/* Run one sequence and return the mean cross-entropy over the answer
 * positions, Q16.16.  `answer_from` is the index of the first token whose
 * prediction counts toward the loss. */
sk_fix sk_train_sequence(sk_train *t, const int *tokens, int ntok,
                         int answer_from);

/* Accumulate the output-projection scale gradient for a sequence. */
sk_fix sk_train_grads(sk_train *t, const int *tokens, int ntok,
                      int answer_from);

/* Apply the accumulated gradients and zero them. */
void sk_train_update(sk_train *t, sk_i32 lr24, sk_i32 trust);

/* Zero the gradient accumulators without applying them. */
void sk_train_zero_grads(sk_train *t);

/* Net change in the sum of output scales from the last update. Negative
 * means the rows are shrinking; sustained negative drift is what makes a
 * model improve for a few epochs and then degrade. */
sk_i32 sk_train_drift(sk_train *t);

/* The loss of the last sequence. */
sk_fix sk_train_last_loss(sk_train *t);

#endif /* SK_TRAIN_H */
