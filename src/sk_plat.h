/* sk_plat.h - the one place that decides which platform we are on.
 *
 * This header exists because the same bug shipped twice:
 *
 *   1. sk_io.c had two separate macro lists, one for the feature-test
 *      block and one for the header block.  They drifted, and the file
 *      compiled its POSIX branch on a DOS target.
 *   2. sk_mem.c then re-derived the platform from scratch, used
 *      SK_PLAT_DOS, and that macro only existed inside sk_io.c.  Result:
 *      the DOS build reported "not a DOS target".
 *
 * Both times the cause was the same: the decision was written down more
 * than once.  So it is written down once, here, and every platform file
 * includes this instead of guessing.
 *
 * SKYNET M1 / subsystem: core/platform
 */
#ifndef SK_PLAT_H
#define SK_PLAT_H

#if defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#  define SK_PLAT_WIN 1
#elif defined(__TURBOC__) || defined(__BORLANDC__) || defined(MSDOS) \
   || defined(__DOS__) || (defined(__WATCOMC__) && !defined(_WIN32))
#  define SK_PLAT_DOS 1
#else
#  define SK_PLAT_POSIX 1
#endif

/* POSIX declarations (fileno, fsync, ftruncate) are hidden by glibc when
   -std=c89 defines __STRICT_ANSI__, so ask for them explicitly.  This must
   happen before any system header is included. */
#if defined(SK_PLAT_POSIX)
#  ifndef _POSIX_C_SOURCE
#    define _POSIX_C_SOURCE 200809L
#  endif
#endif

/* Real-mode DOS with BIOS access: INT 21h/67h are reachable. */
#if defined(SK_PLAT_DOS)
#  define SK_HAVE_BIOS 1
/* Which header declares int86() and MK_FP().  This is a compiler detail,
   not a platform decision, but it belongs here for the same reason: it was
   the last platform-shaped #if left outside this file. */
#  if defined(__WATCOMC__)
#    define SK_DOS_USE_I86 1
#  else
#    define SK_DOS_USE_DOS 1
#  endif
#endif

#endif /* SK_PLAT_H */
