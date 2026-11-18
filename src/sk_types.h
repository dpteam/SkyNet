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

#include <limits.h>

/* ---------------------------------------------------------------------
 * Choosing a type that is EXACTLY 32 bits
 *
 * The whole fixed-point layer, the state file format and the checksums
 * assume 32-bit wraparound.  Writing `unsigned long` and hoping is wrong:
 *
 *   Turbo C 2.0 (16-bit DOS): int=2, long=4   -> long is right
 *   OpenWatcom  (16-bit DOS): int=2, long=4   -> long is right
 *   MinGW / MSVC (Windows):   int=4, long=4   -> either works
 *   gcc / clang (Linux x86-64): int=4, long=8 -> long is WRONG
 *
 * The last row is not hypothetical: on Linux x86-64 `long` is 64 bits, so
 * sk_i32 was silently a 64-bit type there.  Every 64-bit build passed its
 * tests because the values happened to fit, and the error only surfaced
 * when the model layer produced a matvec result that differed from the
 * Windows build by a factor of 55.  Catching it here costs four lines.
 *
 * <limits.h> is in C89, so this works everywhere without stdint.h.
 * ------------------------------------------------------------------- */
#if INT_MAX == 2147483647L
typedef unsigned int  sk_u32;
typedef signed   int  sk_i32;
#elif LONG_MAX == 2147483647L
typedef unsigned long sk_u32;
typedef signed   long sk_i32;
#else
#  error "no 32-bit integer type available on this compiler"
#endif

/* A compile-time proof that the choice was right, so a new compiler that
   breaks the assumption fails at build time instead of at year 30. */
typedef char sk_u32_must_be_4_bytes[(sizeof(sk_u32) == 4) ? 1 : -1];
typedef char sk_i32_must_be_4_bytes[(sizeof(sk_i32) == 4) ? 1 : -1];

typedef unsigned short sk_u16;
typedef signed   short sk_i16;
typedef unsigned char  sk_u8;
typedef signed   char  sk_i8;

#define SK_U32MASK 0xFFFFFFFFUL

/* Wrap to exactly 32 bits, whatever the width of the underlying type. */
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
