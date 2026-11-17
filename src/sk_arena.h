/* sk_arena.h - SkyNet memory arena.
 *
 * One allocator for the whole program.  No malloc() after boot: every
 * buffer comes from here, so "out of memory" is a returned NULL that the
 * caller must handle, never an abort.
 *
 * The backend field is the hook for the DOS memory ladder:
 *   conventional -> EMS 4.0 -> XMS -> DPMI -> OS heap
 * M0 implements SK_MEM_OS and SK_MEM_STATIC; the rest are declared so the
 * model layer can be written against the final interface from day one.
 *
 * SKYNET M0 / subsystem: core/arena
 */
#ifndef SK_ARENA_H
#define SK_ARENA_H

#include "sk_types.h"

#define SK_MEM_OS      0   /* malloc() once at init (host targets)      */
#define SK_MEM_STATIC  1   /* caller-supplied buffer (DOS .BSS)         */
#define SK_MEM_CONV    2   /* conventional 640K, reserved - M1          */
#define SK_MEM_EMS     3   /* LIM EMS 4.0 pages, reserved - M1          */
#define SK_MEM_XMS     4   /* XMS 2.0 blocks, reserved - M1             */
#define SK_MEM_DPMI    5   /* DPMI linear arena, reserved - M1          */

typedef struct {
    sk_u8 *base;
    sk_u32 size;
    sk_u32 used;
    sk_u32 high;        /* high-water mark, for leak hunting */
    sk_u32 blocks;
    int    backend;
    int    ready;
    int    oom;         /* count of refused allocations */
} sk_arena;

int    sk_arena_init(sk_arena *a, sk_u32 bytes, int backend);
int    sk_arena_attach(sk_arena *a, void *buffer, sk_u32 bytes);
void   sk_arena_done(sk_arena *a);

void  *sk_arena_alloc(sk_arena *a, sk_u32 bytes);
void  *sk_arena_alloc_aligned(sk_arena *a, sk_u32 bytes, sk_u32 align);

/* Mark/release.  The model layer never frees individual blocks: it takes
   a mark at the start of a phase and releases back to it.  No free lists,
   no fragmentation, no 50-year drift. */
sk_u32 sk_arena_mark(const sk_arena *a);
void   sk_arena_release(sk_arena *a, sk_u32 mark);
void   sk_arena_reset(sk_arena *a);

sk_u32 sk_arena_free(const sk_arena *a);
int    sk_arena_ok(const sk_arena *a);

#endif /* SK_ARENA_H */
