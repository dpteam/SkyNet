/* sk_shadow.h - the weights as Q8.24 masters, so they can be trained.
 *
 * Why this exists at all:
 *
 * A table stores an integer code per weight and one scale per row. Training
 * the codes directly does not work: one code step changes a weight by
 * `scale`, about 0.005, while a useful gradient step is two or three orders
 * of magnitude smaller. Every update would round to zero and the model would
 * never move.
 *
 * The standard fix is a float master copy, which invariant I1 forbids. The
 * alternative needs no float at all: keep each weight as an int32 in Q8.24,
 * where a weight of 0.005 has 24 fractional bits - about 6e-11 of
 * resolution, three orders finer than the smallest step worth taking. The
 * stored codes become what they always were: a compressed view of that
 * master, regenerated after each update by the same rounding export() uses.
 *
 * The representation is exact in both directions, and the reason is worth
 * writing down because it is what makes the round trip testable:
 *
 *      shadow = code * scale24                (integer multiply, no shift)
 *      scale24' = max|shadow_row| / 127        (integer divide)
 *      code' = round(shadow / scale24')
 *
 * If the row holds a 127 - which it does, because that is how the scale was
 * chosen - then max|shadow| = 127*scale24, so scale24' = scale24 and
 * code' = code. Loading and storing back a table reproduces its bytes
 * exactly, and that is proved against the real model by `skynet shadow`.
 *
 * One table at a time. The largest table this model has needs 256 KB of
 * master, and a 640 KB machine must not pay for all of them at once; the
 * caller supplies the buffer.
 *
 * SKYNET M4 / subsystem: learn/shadow
 */
#ifndef SK_SHADOW_H
#define SK_SHADOW_H

#include "sk_types.h"
#include "sk_tensor.h"

/* Bytes of master storage one table needs. */
sk_u32 sk_shadow_bytes(const sk_table *t);

/* Expand the stored codes into masters. Returns 0 if the table cannot be
 * represented - a scale large enough that code*scale overflows int32 is not
 * a case this model produces, but silently wrapping it would be worse than
 * refusing. */
int sk_shadow_load(const sk_table *t, sk_i32 *w);

/* Re-quantise the masters and write them back.
 *
 * This is the same rounding as train2.py's export(): scale from the row
 * maximum, codes rounded to nearest and clipped to +-127. Written that way
 * on purpose - a model that trains against one quantiser and ships with
 * another is a model whose measured loss does not apply to the binary.
 *
 * Rows whose masters are all zero get a zero scale and zero codes rather
 * than a division by zero. */
int sk_shadow_store(const sk_table *t, const sk_i32 *w);

/* Re-quantise and write back, which is the same function. Named separately
 * at the call site in main.c for readability. */

/* One gradient step on the masters.
 *
 * `grad` is the mean dL/dW_eff in Q16.16 for the same layout, or NULL.
 * `lr24` is the learning rate in Q8.24. The step is `w -= lr * grad`, and
 * the result is NOT clamped per element: a trust region belongs at the row
 * level, where it can see the scale, and doing both produced steps that
 * cancelled each other out in the first version.
 *
 * Returns the number of elements that actually changed. */
int sk_shadow_step(sk_i32 *w, const sk_fix *grad, sk_i32 lr24, sk_u32 n);

/* Largest single master in a table, for reporting. Q8.24. */
sk_fix sk_shadow_max(const sk_i32 *w, sk_u32 n);

#endif /* SK_SHADOW_H */
