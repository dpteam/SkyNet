/* sk_grad.h - fixed-point arithmetic for gradients.
 *
 * Why this is a separate module: a training step for a row scale is about
 * 1e-6, which is below one LSB of Q16.16 (1.5e-5). With Q16.16 gradients
 * every update rounds to zero and the model never learns - silently, since
 * nothing overflows and nothing crashes.
 *
 * So gradients are carried in Q8.24 (value = raw / 2^24) while activations
 * stay in Q16.16. The mixed products below are the only places that need
 * care, and each one is derived in its comment rather than guessed at.
 *
 * SKYNET M4 / subsystem: learn/grad
 */
#ifndef SK_GRAD_H
#define SK_GRAD_H

#include "sk_types.h"
#include "sk_math.h"

/* a * b >> 24 for two Q8.24 values, saturating.  Used where a gradient is
   multiplied by another gradient (the norm backward pass). */
sk_i32 sk_g24_mul(sk_i32 a, sk_i32 b);

/* (Q8.24 gradient) * (Q16.16 value) >> 16, result Q8.24.
   This is exactly what sk_fix_mul does, so it is a macro and not a call. */
#define SK_G24_MUL_V(g24, v16) sk_fix_mul((g24), (v16))

/* Convert Q16.16 to Q8.24 (shift left 8), saturating. */
sk_i32 sk_g24_from16(sk_fix v);

/* Convert Q8.24 to Q16.16 (shift right 8). */
sk_fix sk_g24_to16(sk_i32 g);

/* dx += dy * scale * w, where dy is a Q8.24 gradient, scale is Q8.24 and
   w is a small integer weight.
     dy*scale        = D*S*2^48
     >>24            = D*S*2^24        (Q8.24)
     * w             = D*S*w*2^24
     >>8             = D*S*w*2^16      (Q16.16, which is what dx holds)
   So the sequence is mul24, multiply by the integer, shift 8. */
void sk_grad_dx_add(sk_fix *dx, sk_i32 dy24, sk_i32 scale24, sk_i32 w);

/* dscale += dy * dot, both as above: Q8.24 gradient times a Q16.16 dot
   product, giving a Q8.24 gradient contribution. */
#define SK_GRAD_DSCALE_ADD(dscale, dy24, dot16) \
    ((dscale) += sk_fix_mul((dy24), (dot16)))

/* Learning-rate application: scale -= lr * grad, everything in Q8.24.
 *
 * The rate must be Q8.24 and not Q16.16: dL/dscale = dL/dlogits * dot, and
 * with dot around 600 a useful step needs a rate near 1e-5, which rounds to
 * zero in Q16.16. A Q16.16 rate there produced a step fifteen times the
 * scale itself and the loss rose every epoch. */
sk_i32 sk_g24_apply(sk_i32 scale24, sk_i32 grad24, sk_i32 lr24);

/* Same, but the step may not exceed `max_rel` of the current scale.
 *
 * dL/dscale = dL/dlogits * dot, and dot ranges from about 30 to 2000 across
 * rows of the same model. A single absolute rate therefore either does
 * nothing for small rows or overshoots for large ones: measured, the loss
 * fell for five epochs and then rose. A relative cap makes one rate work
 * for all of them. */
sk_i32 sk_g24_apply_trust(sk_i32 scale24, sk_i32 grad24, sk_i32 lr24,
                          sk_i32 max_rel);

/* RMSProp-style step: divide each row's update by the running magnitude of
 * that row's gradient, then apply the same relative trust cap.
 *
 * dL/dscale[i] = dL/dlogit[i] * dot[i], and dot differs by up to 60x across
 * the 97 output rows. One correct row against 96 wrong ones therefore
 * produces a systematic net shrink of the scales, which flattens the output
 * distribution: measured, the loss improved for two epochs and then rose.
 * Normalising per row removes the dependence on dot. */
sk_i32 sk_g24_apply_rms(sk_i32 scale24, sk_i32 grad24, sk_i32 lr24,
                        sk_i32 *avg24, sk_i32 max_rel);

/* Checks. */
int sk_grad_selftest(void);

#endif /* SK_GRAD_H */
