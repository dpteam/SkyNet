/* sk_types.h - SkyNet base types.
 *
 * Strict C89.  No <stdint.h>, no long long, no // comments, no inline.
 * Every target we care about (Turbo C 2.0, OpenWatcom, DJGPP, TCC, GCC,
 * MSVC) can compile this.
 *
 * SKYNET M0 / subsystem: core
 */
#ifndef SK_TYPES_H
#define SK_TYPES_H

/* sk_u32 / sk_i32 are at least 32 bits.  On 16-bit DOS long is 32 bits,
   on 32/64-bit hosts it is 32/64.  Arithmetic that must be exactly
   32 bits wide is wrapped in SK_U32() so the result is identical on
   every target - this is what makes the state files portable. */
typedef unsigned long  sk_u32;
typedef signed   long  sk_i32;
typedef unsigned short sk_u16;
typedef signed   short sk_i16;
typedef unsigned char  sk_u8;
typedef signed   char  sk_i8;

#define SK_U32MASK 0xFFFFFFFFUL

/* Wrap to exactly 32 bits, whatever the width of long is. */
#define SK_U32(x) ((sk_u32)((x) & SK_U32MASK))

typedef int sk_bool;
#define SK_TRUE  1
#define SK_FALSE 0

#define SK_NULL ((void *)0)

/* ---- Q16.16 fixed point -------------------------------------------------
   value = raw / 65536.  Range of the raw type is +-32767.99998, which is
   all the range a 16-bit friendly neural core needs.  Invariants:
     - sk_fix_mul() is exact while |a*b| < 2^62 (i.e. always, it uses
       16-bit limbs and saturates only if the result cannot fit);
     - sk_fix_div() is exact for |b| < 2^30.
   See sk_math.c for the proofs in comments. */
typedef sk_i32 sk_fix;

#define SK_FIX_BITS 16
#define SK_FIX_ONE  ((sk_fix)65536L)
#define SK_FIX_MAX  ((sk_fix)0x7FFFFFFFL)
#define SK_FIX_MIN  ((sk_fix)(-0x7FFFFFFFL))

#endif /* SK_TYPES_H */
