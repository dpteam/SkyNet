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
#include "sk_arena.h"

#define SK_ARENA_ALIGN 4

int sk_arena_init(sk_arena *a, sk_u32 bytes, int backend)
{
    if (a == SK_NULL)
        return 0;
    memset(a, 0, sizeof(*a));
    if (backend != SK_MEM_OS)
        return 0;                     /* other backends arrive in M1 */
    if (bytes == 0)
        return 0;
    a->base = (sk_u8 *)malloc((size_t)bytes);
    if (a->base == SK_NULL)
        return 0;
    a->size = bytes;
    a->backend = SK_MEM_OS;
    a->ready = 1;
    return 1;
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
