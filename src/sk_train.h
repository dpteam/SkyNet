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

/* Ask for the per-weight gradient buffers.
 *
 * Off by default because they are not free: dW for the last block's two
 * feed-forward tables is HID*D*4 bytes each, about 196 KB together, and a
 * 640 KB machine must not pay that to adjust row scales. With the flag the
 * trainer also accumulates dL/dW for every element, which is what training
 * the int8 codes needs.
 *
 * C89 note: a flags word rather than a boolean, because the next thing to
 * want here is almost certainly the middle layers' gradients. */
#define SK_TRAIN_CODES 1

/* Bytes of storage a trainer for this model needs, given `flags`. */
sk_u32 sk_train_bytes(const sk_model *m, int flags);

/* Open a trainer.  `storage` must be at least sk_train_bytes(m, flags) long. */
int  sk_train_open(sk_train *t, sk_model *m, void *storage, sk_u32 bytes,
                   int flags);
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

/* Identifies one trainable parameter for the checker. */
#define SK_P_OUT 0     /* output projection row scale      */
#define SK_P_GF  1     /* final gains, element             */
#define SK_P_G2  2     /* last block g2 gains, element     */
#define SK_P_F1  3     /* last block f1 row scale, row     */
#define SK_P_F2  4     /* last block f2 row scale, row     */
#define SK_P_DW_F1 5   /* last block f1 weight, element    */
#define SK_P_DW_F2 6   /* last block f2 weight, element    */

/* Nothing can be added to a value that is not constant, so the function
 * below returns the gradient of the effective weight, not of the code.
 * dL/dcode = dL/dW_eff * scale; the scale is applied by the caller, which is
 * the only place that knows which representation it is updating. */

/* Mean dL/dW_eff for one element of a table, row-major, Q16.16.
 * `index` = row * cols + col. Zero unless SK_TRAIN_CODES was asked for. */
sk_fix sk_train_dw_get(sk_train *t, int kind, int index);

/* Mean gradient for one parameter, raw Q16.16. */
sk_fix sk_train_grad_get(sk_train *t, int kind, int index);

/* Read and write the parameter itself, for the finite difference. */
int sk_train_param_get(sk_train *t, int kind, int index, sk_fix *out);
int sk_train_param_set(sk_train *t, int kind, int index, sk_fix value);

/* The accumulated mean gradient for one output row, raw Q16.16.
 * Not Q8.24: converting saturates above 128 and hides large gradients. */
sk_i32 sk_train_grad_row(sk_train *t, int r);

/* Net change in the sum of output scales from the last update. Negative
 * means the rows are shrinking; sustained negative drift is what makes a
 * model improve for a few epochs and then degrade. */
sk_i32 sk_train_drift(sk_train *t);

/* The loss of the last sequence. */
sk_fix sk_train_last_loss(sk_train *t);

#endif /* SK_TRAIN_H */
