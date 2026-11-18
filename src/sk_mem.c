/* sk_mem.c - DOS memory discovery and the EMS paged store.
 *
 * SKYNET M1 / subsystem: core/mem
 */
#include "sk_plat.h"

#include <string.h>
#include <stdio.h>
#include "sk_mem.h"
#include "sk_io.h"

#if defined(SK_DOS_USE_I86)
#  include <i86.h>
#elif defined(SK_DOS_USE_DOS)
#  include <dos.h>
#endif

/* ------------------------------------------------------------------ */
/* BIOS/DOS interrupt access                                          */
/* ------------------------------------------------------------------ */

#if defined(SK_HAVE_BIOS)

/* One place where registers are touched.  Everything above this line is
   portable C89; this is the only code in the memory layer that knows it
   is running on a PC. */
static int sk_bios_int(int num, unsigned ax, unsigned bx, unsigned cx,
                       unsigned dx, unsigned *oax, unsigned *obx,
                       unsigned *ocx, unsigned *odx)
{
    union REGS r;

    memset(&r, 0, sizeof(r));
    r.x.ax = ax;
    r.x.bx = bx;
    r.x.cx = cx;
    r.x.dx = dx;
    int86(num, &r, &r);
    if (oax) *oax = r.x.ax;
    if (obx) *obx = r.x.bx;
    if (ocx) *ocx = r.x.cx;
    if (odx) *odx = r.x.dx;
    return r.x.ax & 0xFF;
}

#endif /* SK_HAVE_BIOS */

/* ------------------------------------------------------------------ */
/* probe                                                              */
/* ------------------------------------------------------------------ */

void sk_mem_probe(sk_meminfo *mi)
{
    if (mi == SK_NULL)
        return;
    memset(mi, 0, sizeof(*mi));

#if defined(SK_HAVE_BIOS)
    mi->is_dos = 1;

    /* EMS: the driver answers INT 67h AH=40h with AH=0 when present.
       A machine can have the vector installed with no driver behind it,
       so the status call is what matters, not the vector. */
    {
        unsigned oax = 0, obx = 0, odx = 0;
        unsigned ah;

        ah = (unsigned)sk_bios_int(0x67, 0x4000, 0, 0, 0, &oax, &obx, 0, &odx);
        if ((oax >> 8) == 0) {
            mi->ems_present = 1;

            /* version: AH=46h -> AL = BCD version */
            sk_bios_int(0x67, 0x4600, 0, 0, 0, &oax, 0, 0, 0);
            mi->ems_version = (int)(oax & 0xFF);

            /* page frame segment: AH=41h -> BX */
            sk_bios_int(0x67, 0x4100, 0, 0, 0, 0, &obx, 0, 0);
            mi->ems_frame_seg = (sk_u32)obx;

            /* free / total pages: AH=42h -> BX = free, DX = total */
            sk_bios_int(0x67, 0x4200, 0, 0, 0, 0, &obx, 0, &odx);
            mi->ems_free_pages = (sk_u32)obx;
            mi->ems_total_pages = (sk_u32)odx;
        }
    }

    /* XMS: INT 2Fh AX=4300h, AL=80h when a driver is installed. */
    {
        unsigned oax = 0;
        sk_bios_int(0x2F, 0x4300, 0, 0, 0, &oax, 0, 0, 0);
        if ((oax & 0xFF) == 0x80) {
            mi->xms_present = 1;
        }
    }

    /* DPMI: INT 2Fh AX=1687h, AX=0 on return when a host is present. */
    {
        unsigned oax = 1, obx = 0;
        sk_bios_int(0x2F, 0x1687, 0, 0, 0, &oax, &obx, 0, 0);
        if (oax == 0) {
            mi->dpmi_present = 1;
            mi->dpmi_version = (int)(obx & 0xFF);
        }
    }
#endif /* SK_HAVE_BIOS */
}

/* ------------------------------------------------------------------ */
/* report                                                             */
/* ------------------------------------------------------------------ */

static void sk_num(char *buf, unsigned long v)
{
    char tmp[16];
    int i, n;

    n = 0;
    if (v == 0) {
        tmp[n++] = '0';
    } else {
        while (v > 0 && n < 15) {
            tmp[n++] = (char)('0' + (int)(v % 10UL));
            v /= 10UL;
        }
    }
    for (i = 0; i < n; i++)
        buf[i] = tmp[n - 1 - i];
    buf[n] = '\0';
}

static void sk_cat(char *dst, const char *a, const char *b)
{
    strcpy(dst, a);
    strcat(dst, b);
}

void sk_mem_report(const sk_meminfo *mi, void (*emit)(const char *line))
{
    char line[96];
    char num[24];

    if (mi == SK_NULL || emit == SK_NULL)
        return;

    if (!mi->is_dos) {
        emit("memory: not a DOS target, using the host allocator");
        return;
    }

    /* EMS */
    if (mi->ems_present) {
        sk_cat(line, "ems   : v", "");
        sk_num(num, (unsigned long)(mi->ems_version >> 4));
        strcat(line, num);
        strcat(line, ".");
        sk_num(num, (unsigned long)(mi->ems_version & 0x0F));
        strcat(line, num);
        strcat(line, "  frame ");
        sk_num(num, mi->ems_frame_seg);
        strcat(line, num);
        strcat(line, "h  free ");
        sk_num(num, mi->ems_free_pages);
        strcat(line, num);
        strcat(line, "/");
        sk_num(num, mi->ems_total_pages);
        strcat(line, num);
        strcat(line, " pages (");
        sk_num(num, mi->ems_free_pages * 16UL);
        strcat(line, num);
        strcat(line, " KB free)");
        emit(line);
    } else {
        emit("ems   : not present");
    }

    /* XMS */
    if (mi->xms_present) {
        emit("xms   : driver present (detected, not used yet)");
    } else {
        emit("xms   : not present");
    }

    /* DPMI */
    if (mi->dpmi_present) {
        sk_cat(line, "dpmi  : host present, version ", "");
        sk_num(num, (unsigned long)mi->dpmi_version);
        strcat(line, num);
        emit(line);
    } else {
        emit("dpmi  : not present");
    }

    /* The number that actually matters for the model. */
    if (mi->ems_present && mi->ems_free_pages > 0) {
        sk_cat(line, "usable: ", "");
        sk_num(num, mi->ems_free_pages * 16UL);
        strcat(line, num);
        strcat(line, " KB of paged memory via EMS");
        emit(line);
    } else {
        emit("usable: 65 KB linear only (the M0 ceiling)");
    }
}

/* ------------------------------------------------------------------ */
/* paged store                                                        */
/* ------------------------------------------------------------------ */

#if defined(SK_HAVE_BIOS)
/* Far pointer into the EMS page frame.  MK_FP is provided by both Turbo C
   (<dos.h>) and OpenWatcom (<i86.h>) and is the correct way to build one:
   casting an integer linear address to a pointer happens to work in the
   large model but silently breaks in the small model, where the pointer
   is near. */
#  define SK_EMS_WINDOW(seg) ((unsigned char *)MK_FP((seg), 0))
#endif

/* The EMS page frame is a PHYSICAL resource: there is one window of four
 * slots in conventional memory, shared by every pool in the process.  So
 * "which page is currently mapped" is global state, not per-pool state.
 *
 * Keeping it per-pool was a real bug with two pools alive at once - the
 * weights in one, the KV cache in another.  Pool A maps page 5, pool B maps
 * page 9, then pool A reads again, sees its own cached "page 5 is mapped",
 * skips the map call, and reads pool B's data.  The forward pass then
 * produced different numbers on DOS than on the host with no error anywhere.
 * Exactly the trap this API was designed to avoid: a pointer whose meaning
 * changes behind your back. */
static int sk_ems_mapped_handle = -1;
static sk_u32 sk_ems_mapped_page = 0xFFFFFFFFUL;

#if !defined(SK_HAVE_BIOS)
/* Without BIOS access there is no window at all, so the bookkeeping above
   is dead weight.  Marked used so the build stays warning-free on hosts. */
static void sk_ems_window_unused(void)
{
    (void)sk_ems_mapped_handle;
    (void)sk_ems_mapped_page;
}
#endif

/* Map logical page `page` of handle `h` into window slot 0.
   Returns 1 on success, 0 on failure.  The caller then reads or writes
   through SK_EMS_WINDOW(frame_seg).  Any such pointer is valid only until
   the next map by ANY pool, which is why the pool API is read/write and
   never hands out a raw pointer. */
#if defined(SK_HAVE_BIOS)
static int sk_pool_map(sk_pool *p, sk_u32 page)
{
    if (p->backend != SK_POOL_EMS)
        return 0;
    if (sk_ems_mapped_handle == p->handle && sk_ems_mapped_page == page)
        return 1;                       /* already in the window */

    {
        unsigned oax = 0;
        unsigned ax = 0x4400;           /* AH=44h, AL=physical page 0 */
        unsigned bx = (unsigned)page;   /* logical page */
        unsigned dx = (unsigned)p->handle;

        sk_bios_int(0x67, ax, bx, 0, dx, &oax, 0, 0, 0);
        if ((oax >> 8) != 0) {
            sk_ems_mapped_handle = -1;  /* window state now unknown */
            return 0;                   /* map refused */
        }
    }
    sk_ems_mapped_handle = p->handle;
    sk_ems_mapped_page = page;
    p->maps++;
    return 1;
}
#endif

int sk_pool_open(sk_pool *p, sk_u32 bytes)
{
    sk_meminfo mi;

    if (p == SK_NULL || bytes == 0)
        return 0;
    memset(p, 0, sizeof(*p));
    p->mapped = -1;

    sk_mem_probe(&mi);
    if (!mi.ems_present || mi.ems_free_pages == 0)
        return 0;

#if defined(SK_HAVE_BIOS)
    {
        sk_u32 need;
        unsigned oax = 0, odx = 0;

        need = (bytes + SK_EMS_PAGE - 1) / SK_EMS_PAGE;
        if (need == 0)
            need = 1;
        if (need > mi.ems_free_pages)
            return 0;                   /* not enough: caller degrades */

        sk_bios_int(0x67, 0x4300, (unsigned)need, 0, 0, &oax, 0, 0, &odx);
        if ((oax >> 8) != 0)
            return 0;                   /* allocation refused */

        p->backend = SK_POOL_EMS;
        p->handle = (int)odx;
        p->pages = need;
        p->bytes = need * SK_EMS_PAGE;
        p->frame_seg = mi.ems_frame_seg;
        p->ready = 1;
        return 1;
    }
#else
    return 0;
#endif
}

int sk_pool_open_from(sk_pool *p, sk_u32 bytes, const void *src)
{
    if (p == SK_NULL || src == SK_NULL || bytes == 0)
        return 0;
    memset(p, 0, sizeof(*p));
    p->mapped = -1;
    p->backend = SK_POOL_LINEAR;
    p->base = (unsigned char *)src;    /* caller owns the memory */
    p->bytes = bytes;
    p->ready = 1;
    return 1;
}

int sk_pool_read(sk_pool *p, sk_u32 off, void *dst, sk_u32 len)
{
    if (p == SK_NULL || !p->ready || dst == SK_NULL)
        return 0;
    if (off > p->bytes || len > p->bytes - off)
        return 0;

    if (p->backend == SK_POOL_LINEAR) {
        memcpy(dst, p->base + off, (size_t)len);
        p->reads++;
        return 1;
    }

#if defined(SK_HAVE_BIOS)
    {
        unsigned char *d = (unsigned char *)dst;

        if (p->backend != SK_POOL_EMS)
            return 0;

        while (len > 0) {
            sk_u32 page = off / SK_EMS_PAGE;
            sk_u32 within = off % SK_EMS_PAGE;
            sk_u32 chunk = SK_EMS_PAGE - within;

            if (chunk > len)
                chunk = len;
            if (!sk_pool_map(p, page))
                return 0;
            memcpy(d, SK_EMS_WINDOW(p->frame_seg) + within, (size_t)chunk);

            d += chunk;
            off += chunk;
            len -= chunk;
            p->reads++;
        }
        return 1;
    }
#else
    return 0;
#endif
}

int sk_pool_write(sk_pool *p, sk_u32 off, const void *src, sk_u32 len)
{
    if (p == SK_NULL || !p->ready || src == SK_NULL)
        return 0;
    if (off > p->bytes || len > p->bytes - off)
        return 0;

    if (p->backend == SK_POOL_LINEAR) {
        memcpy(p->base + off, src, (size_t)len);
        p->writes++;
        return 1;
    }

#if defined(SK_HAVE_BIOS)
    {
        const unsigned char *s = (const unsigned char *)src;

        if (p->backend != SK_POOL_EMS)
            return 0;

        while (len > 0) {
            sk_u32 page = off / SK_EMS_PAGE;
            sk_u32 within = off % SK_EMS_PAGE;
            sk_u32 chunk = SK_EMS_PAGE - within;

            if (chunk > len)
                chunk = len;
            if (!sk_pool_map(p, page))
                return 0;
            memcpy(SK_EMS_WINDOW(p->frame_seg) + within, s, (size_t)chunk);

            s += chunk;
            off += chunk;
            len -= chunk;
            p->writes++;
        }
        return 1;
    }
#else
    return 0;
#endif
}

void sk_pool_close(sk_pool *p)
{
    if (p == SK_NULL)
        return;

#if defined(SK_HAVE_BIOS)
    if (p->ready && p->backend == SK_POOL_EMS) {
        unsigned oax = 0;
        sk_bios_int(0x67, 0x4500, 0, 0, (unsigned)p->handle, &oax, 0, 0, 0);
        /* The window may still show a page of the freed handle. */
        if (sk_ems_mapped_handle == p->handle) {
            sk_ems_mapped_handle = -1;
            sk_ems_mapped_page = 0xFFFFFFFFUL;
        }
    }
#endif
    p->backend = SK_POOL_NONE;
    p->ready = 0;
    p->handle = 0;
    p->pages = 0;
    p->mapped = -1;
#if !defined(SK_HAVE_BIOS)
    sk_ems_window_unused();
#endif
}

/* ------------------------------------------------------------------ */
/* selftest                                                           */
/* ------------------------------------------------------------------ */

/* The pattern is a pure function of the offset, so the test needs no
   source buffer at all - which matters on DOS, where a 48 KB scratch
   array would not fit in the 64 KB data segment next to everything else. */
static unsigned char sk_pool_pattern(sk_u32 off)
{
    return (unsigned char)((off * 31UL + (off >> 8)) & 0xFFUL);
}

int sk_pool_selftest(void)
{
    sk_pool p;
    sk_u32 total;
    sk_u32 off;
    unsigned char buf[257];
    int ok;
    sk_u32 i;

    total = 3UL * SK_EMS_PAGE;          /* crosses two page boundaries */

    if (!sk_pool_open(&p, total)) {
        /* No EMS here.  Not a failure: the host uses the linear arena and
           the DOS report already says what is missing. */
        printf("  pool: skipped (no EMS on this machine)\n");
        return 1;
    }

    ok = 1;
    printf("  pool: %lu KB in %lu EMS pages\n",
           (unsigned long)(p.bytes / 1024UL), (unsigned long)p.pages);

    /* write in one pass, page by page, through a small stack buffer */
    for (off = 0; off < total; off += (sk_u32)sizeof(buf)) {
        sk_u32 n = total - off;
        if (n > (sk_u32)sizeof(buf))
            n = (sk_u32)sizeof(buf);
        for (i = 0; i < n; i++)
            buf[i] = sk_pool_pattern(off + i);
        if (!sk_pool_write(&p, off, buf, n)) {
            printf("  FAIL: write at %lu\n", (unsigned long)off);
            ok = 0;
            break;
        }
    }

    /* read back in odd-sized chunks so every page boundary is crossed at
       a different alignment than it was written */
    if (ok) {
        for (off = 0; off < total; off += 173) {
            sk_u32 n = total - off;
            if (n > 173)
                n = 173;
            if (!sk_pool_read(&p, off, buf, n)) {
                printf("  FAIL: read at %lu\n", (unsigned long)off);
                ok = 0;
                break;
            }
            for (i = 0; i < n; i++) {
                if (buf[i] != sk_pool_pattern(off + i)) {
                    printf("  FAIL: byte %lu is %02X, expected %02X\n",
                           (unsigned long)(off + i), (unsigned int)buf[i],
                           (unsigned int)sk_pool_pattern(off + i));
                    ok = 0;
                    break;
                }
            }
            if (!ok)
                break;
        }
    }

    /* reading past the end must be refused, not wrap */
    if (ok && sk_pool_read(&p, total - 4UL, buf, 8UL))
        ok = 0;

    printf("  pool: %lu maps, %lu reads, %lu writes\n",
           (unsigned long)p.maps, (unsigned long)p.reads,
           (unsigned long)p.writes);

    sk_pool_close(&p);
    return ok;
}
