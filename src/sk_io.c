/* sk_io.c - portable file helpers.  See sk_io.h.
 *
 * SKYNET M0 / subsystem: core/io
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "sk_io.h"

#if defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#  define SK_PLAT_WIN 1
#  include <direct.h>
#  include <io.h>
#  include <windows.h>
#elif defined(__TURBOC__) || defined(__BORLANDC__) || defined(MSDOS)
#  define SK_PLAT_DOS 1
#  include <dir.h>
#  include <io.h>
#else
#  define SK_PLAT_POSIX 1
#  include <unistd.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#endif

/* Durability policy.  See sk_io.h.
 *
 * Measured on this machine: write+fsync+close = 85 ms, write+close = 0.1 ms.
 * Durability is 850 times more expensive than the work it protects, so the
 * policy belongs to the caller and is never guessed here. */
static int sk_io_sync_enabled = SK_SYNC_DEVICE;
static int sk_io_sync_cost = 85;

int sk_io_mkdir(const char *path)
{
#if defined(SK_PLAT_WIN)
    if (_mkdir(path) == 0)
        return 0;
    return -1;
#elif defined(SK_PLAT_DOS)
    if (mkdir(path) == 0)
        return 0;
    return -1;
#else
    if (mkdir(path, 0755) == 0)
        return 0;
    return -1;
#endif
}

int sk_io_mkdirs(const char *path)
{
    char buf[256];
    sk_u32 i, n;

    if (path == SK_NULL)
        return -1;
    n = (sk_u32)strlen(path);
    if (n == 0 || n >= (sk_u32)sizeof(buf))
        return -1;
    memcpy(buf, path, (size_t)(n + 1));

    for (i = 0; i <= n; i++) {
        if (buf[i] == '/' || buf[i] == '\\' || buf[i] == '\0') {
            char keep = buf[i];
            buf[i] = '\0';
            if (i > 0 && buf[i - 1] != ':')
                (void)sk_io_mkdir(buf);
            buf[i] = keep;
        }
    }
    return 0;
}

int sk_io_mkparents(const char *filepath)
{
    char buf[256];
    sk_u32 i;
    int cut;

    if (filepath == SK_NULL)
        return -1;
    if (strlen(filepath) >= sizeof(buf))
        return -1;
    strcpy(buf, filepath);

    cut = -1;
    for (i = 0; buf[i] != '\0'; i++) {
        if (buf[i] == '/' || buf[i] == '\\')
            cut = (int)i;
    }
    if (cut <= 0)
        return 0;                      /* no directory part */
    buf[cut] = '\0';
    return sk_io_mkdirs(buf);
}

int sk_io_join(char *dst, sk_u32 cap, const char *dir, const char *name)
{
    sk_u32 dl, nl;

    if (dst == SK_NULL || dir == SK_NULL || name == SK_NULL || cap == 0)
        return -1;
    dl = (sk_u32)strlen(dir);
    nl = (sk_u32)strlen(name);
    if (dl + nl + 2 > cap)
        return -1;
    memcpy(dst, dir, dl);
    if (dl > 0 && dir[dl - 1] != '/' && dir[dl - 1] != '\\')
        dst[dl++] = '/';
    memcpy(dst + dl, name, nl);
    dst[dl + nl] = '\0';
    return 0;
}

int sk_io_sync(FILE *f)
{
    if (f == SK_NULL)
        return -1;
    if (fflush(f) != 0)
        return -1;
    if (!sk_io_sync_enabled)
        return 0;
#if defined(SK_PLAT_WIN)
    return (_commit(_fileno(f)) == 0) ? 0 : -1;
#elif defined(SK_PLAT_DOS)
    /* Turbo C / plain DOS: no commit primitive available.  The data is
       in the DOS buffers, which is the best we can do without a driver.
       This is why the crash test distinguishes process-kill (safe) from
       power-loss (loses the last unflushed snapshot). */
    return 0;
#else
    return (fsync(fileno(f)) == 0) ? 0 : -1;
#endif
}

void sk_io_sync_mode(int mode)
{
    sk_io_sync_enabled = (mode != SK_SYNC_NONE);
}

int sk_io_sync_cost_ms(void)
{
    return sk_io_sync_cost;
}

int sk_io_truncate(FILE *f, unsigned long size)
{
    if (f == SK_NULL)
        return -1;
    if (fflush(f) != 0)
        return -1;
#if defined(SK_PLAT_WIN)
    if (_chsize(_fileno(f), (long)size) != 0)
        return -1;
#elif defined(SK_PLAT_DOS)
    if (chsize(fileno(f), (long)size) != 0)
        return -1;
#else
    if (ftruncate(fileno(f), (off_t)size) != 0)
        return -1;
#endif
    return 0;
}

unsigned long sk_io_size(FILE *f)
{
    long cur, end;

    if (f == SK_NULL)
        return 0;
    cur = ftell(f);
    if (cur < 0)
        return 0;
    if (fseek(f, 0L, SEEK_END) != 0)
        return 0;
    end = ftell(f);
    if (end < 0)
        end = 0;
    fseek(f, cur, SEEK_SET);
    return (unsigned long)end;
}

int sk_io_remove(const char *path)
{
    if (path == SK_NULL)
        return -1;
    if (remove(path) == 0)
        return 0;
    return -1;
}

int sk_io_rename(const char *from, const char *to)
{
    if (from == SK_NULL || to == SK_NULL)
        return -1;
    if (rename(from, to) == 0)
        return 0;
    return -1;
}

int sk_io_exists(const char *path)
{
    FILE *f;

    if (path == SK_NULL)
        return 0;
    f = fopen(path, "rb");
    if (f == SK_NULL)
        return 0;
    fclose(f);
    return 1;
}

int sk_io_replace(const char *src, const char *dst)
{
    if (src == SK_NULL || dst == SK_NULL)
        return -1;
#if defined(SK_PLAT_WIN)
    if (MoveFileExA(src, dst,
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return 0;
    return -1;
#elif defined(SK_PLAT_DOS)
    /* no atomic replace in the DOS API: two steps, and the caller must
       hold two slots.  The vault does. */
    (void)remove(dst);
    return (rename(src, dst) == 0) ? 0 : -1;
#else
    return (rename(src, dst) == 0) ? 0 : -1;
#endif
}
