/* sk_mem.h - DOS memory discovery and paged bulk storage.
 *
 * Why this exists: on 16-bit DOS a single malloc() cannot exceed 64 KB,
 * because size_t is 16 bits.  The M0 arena therefore tops out at 65,280
 * bytes no matter how much memory the machine has, and the Tiny baseline
 * model is 380 KB of weights.  EMS 4.0 is how real-mode DOS gets past
 * that: up to 32 MB, paged 16 KB at a time through a 64 KB window in
 * conventional memory.
 *
 * The important design point: EMS memory is NOT linearly addressable.
 * You can see 4 pages at a time.  So this layer does not pretend to be a
 * heap.  sk_arena stays the linear allocator (conventional, DPMI, OS), and
 * sk_pool is a separate paged store with explicit read/write.  Faking a
 * flat address space over a windowed one is how projects end up with
 * pointers that are valid only until the next map().
 *
 * SKYNET M1 / subsystem: core/mem
 */
#ifndef SK_MEM_H
#define SK_MEM_H

#include "sk_types.h"

/* ---- what the machine actually has ---------------------------------- */
typedef struct {
    int    is_dos;

    /* EMS 4.0, via INT 67h */
    int    ems_present;
    int    ems_version;      /* 0x40 for 4.0 */
    sk_u32 ems_total_pages;  /* 16 KB pages */
    sk_u32 ems_free_pages;
    sk_u32 ems_frame_seg;    /* segment of the 64 KB page frame */

    /* XMS 2.0, via INT 2Fh AX=4300h.  Detected, not yet used: the driver
       entry point is a far call, and doing that portably across Turbo C,
       OpenWatcom and DJGPP needs an assembly shim per compiler.  Recorded
       here so the report is honest about what is available. */
    int    xms_present;
    int    xms_version;
    sk_u32 xms_total_kb;
    sk_u32 xms_free_kb;

    /* DPMI, via INT 2Fh AX=1687h.  Same story: detection only for now. */
    int    dpmi_present;
    int    dpmi_version;
} sk_meminfo;

/* Fill in what is available.  Safe to call on any platform. */
void sk_mem_probe(sk_meminfo *mi);

/* Human-readable dump, one line per subsystem. */
void sk_mem_report(const sk_meminfo *mi, void (*emit)(const char *line));

/* ---- paged bulk store ------------------------------------------------ */

#define SK_POOL_EMS     1
#define SK_POOL_LINEAR  2
#define SK_POOL_NONE    0

#define SK_EMS_PAGE     16384UL
#define SK_EMS_SLOTS    4

typedef struct {
    int    backend;
    int    handle;        /* EMS handle */
    sk_u32 pages;         /* 16 KB pages owned */
    sk_u32 bytes;         /* usable bytes = pages * 16384 */
    sk_u32 frame_seg;
    int    mapped;        /* which logical page is in slot 0, or -1 */
    int    ready;
    sk_u32 reads;
    sk_u32 writes;
    sk_u32 maps;
} sk_pool;

/* Open a store of at least `bytes`.  Prefers EMS; falls back to the
   linear arena when running on a host.  Returns 1 on success. */
int  sk_pool_open(sk_pool *p, sk_u32 bytes);

/* Fill a store from a caller-provided buffer (used by the host fallback
   so the same code path can be tested off-DOS). */
int  sk_pool_open_from(sk_pool *p, sk_u32 bytes, const void *src);

int  sk_pool_read(sk_pool *p, sk_u32 off, void *dst, sk_u32 len);
int  sk_pool_write(sk_pool *p, sk_u32 off, const void *src, sk_u32 len);
void sk_pool_close(sk_pool *p);

/* Diagnostics: verify that what was written comes back byte for byte,
   across page boundaries and across the window. */
int  sk_pool_selftest(void);

#endif /* SK_MEM_H */
