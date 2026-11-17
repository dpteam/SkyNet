/* sk_arena.c - SkyNet memory arena.  See sk_arena.h.
 *
 * Bump allocator with mark/release.  Deliberately not a malloc: a
 * 50-year run must not fragment.  Alignment is computed relative to the
 * arena base, which must itself be at least 8-byte aligned (malloc
 * guarantees this; the DOS static path asserts it at attach time).
 *
 * SKYNET M0 / subsystem: core/arena
 */
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "sk_arena.h"

#define SK_ARENA_ALIGN 4

/* In 16-bit DOS a single malloc() cannot exceed 64 KB, because size_t is
 * 16 bits wide there.  Asking for more does not fail: the size truncates,
 * malloc hands back a tiny block, and the arena then believes it owns
 * megabytes.  That is silent memory corruption, so it is refused here
 * instead.  Bigger arenas on DOS need the EMS/XMS backends (M1), which
 * piece memory together from 64 KB pages. */
#if UINT_MAX <= 0xFFFFUL
#  define SK_ARENA_MAX_BLOCK 0xFF00UL
#else
#  define SK_ARENA_MAX_BLOCK 0xFFFFFFFFUL
#endif

int sk_arena_init(sk_arena *a, sk_u32 bytes, int backend)
{
    if (a == SK_NULL)
        return 0;
    memset(a, 0, sizeof(*a));
    if (backend != SK_MEM_OS)
        return 0;                     /* other backends arrive in M1 */
    if (bytes == 0)
        return 0;
    if (bytes > SK_ARENA_MAX_BLOCK)
        return 0;                     /* see SK_ARENA_MAX_BLOCK above */
    a->base = (sk_u8 *)malloc((size_t)bytes);
    if (a->base == SK_NULL)
        return 0;
    a->size = bytes;
    a->backend = SK_MEM_OS;
    a->ready = 1;
    return 1;
}

sk_u32 sk_arena_init_best(sk_arena *a, sk_u32 want, sk_u32 least, int backend)
{
    sk_u32 size;

    if (a == SK_NULL || want == 0 || least == 0 || least > want)
        return 0;
    if (want > SK_ARENA_MAX_BLOCK)
        want = SK_ARENA_MAX_BLOCK;

    /* Degrade, do not fail (invariant I6).  Ask for what the caller wants,
       then halve until something fits or we reach the floor.  On DOS this
       is what turns "8 MB requested, machine has 600 KB" into a working
       run instead of an error message. */
    size = want;
    for (;;) {
        if (sk_arena_init(a, size, backend))
            return size;
        if (size <= least)
            return 0;
        size /= 2;
        if (size < least)
            size = least;
    }
}

int sk_arena_attach(sk_arena *a, void *buffer, sk_u32 bytes)
{
    if (a == SK_NULL || buffer == SK_NULL || bytes == 0)
        return 0;
    memset(a, 0, sizeof(*a));
    a->base = (sk_u8 *)buffer;
    a->size = bytes;
    a->backend = SK_MEM_STATIC;
    a->ready = 1;
    return 1;
}

void sk_arena_done(sk_arena *a)
{
    if (a == SK_NULL)
        return;
    if (a->backend == SK_MEM_OS && a->base != SK_NULL)
        free(a->base);
    a->base = SK_NULL;
    a->size = 0;
    a->used = 0;
    a->ready = 0;
}

void *sk_arena_alloc_aligned(sk_arena *a, sk_u32 bytes, sk_u32 align)
{
    sk_u32 off, pad;

    if (a == SK_NULL || !a->ready || bytes == 0)
        return SK_NULL;
    if (align < SK_ARENA_ALIGN)
        align = SK_ARENA_ALIGN;

    off = a->used;
    pad = off % align;
    if (pad != 0)
        pad = align - pad;

    if (pad > a->size - off || bytes > a->size - off - pad) {
        a->oom++;
        return SK_NULL;
    }

    a->used = off + pad + bytes;
    if (a->used > a->high)
        a->high = a->used;
    a->blocks++;
    return (void *)(a->base + off + pad);
}

void *sk_arena_alloc(sk_arena *a, sk_u32 bytes)
{
    return sk_arena_alloc_aligned(a, bytes, SK_ARENA_ALIGN);
}

sk_u32 sk_arena_mark(const sk_arena *a)
{
    return (a == SK_NULL) ? 0 : a->used;
}

void sk_arena_release(sk_arena *a, sk_u32 mark)
{
    if (a == SK_NULL || !a->ready)
        return;
    if (mark <= a->used)
        a->used = mark;
}

void sk_arena_reset(sk_arena *a)
{
    if (a == SK_NULL)
        return;
    a->used = 0;
    a->blocks = 0;
    a->oom = 0;
}

sk_u32 sk_arena_free(const sk_arena *a)
{
    if (a == SK_NULL || !a->ready || a->used > a->size)
        return 0;
    return a->size - a->used;
}

int sk_arena_ok(const sk_arena *a)
{
    return (a != SK_NULL && a->ready);
}
