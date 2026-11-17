/* sk_io.h - portable file helpers.
 *
 * The only place in SkyNet where platform #ifdefs are allowed to touch
 * the filesystem.  Everything above this layer uses these functions and
 * stays portable C89.
 *
 * SKYNET M0 / subsystem: core/io
 */
#ifndef SK_IO_H
#define SK_IO_H

#include <stdio.h>
#include "sk_types.h"

/* Returns 0 on success, -1 on failure.  An already existing directory is
   a success (DOS has no way to tell, and we do not care). */
int sk_io_mkdir(const char *path);

/* Create every missing component of path.  Used so that
   --dir a/b/c and --out a/b/c/primes.txt both work without the caller
   having to think about it. */
int sk_io_mkdirs(const char *path);

/* Create the parent directory of a file path (may be a no-op). */
int sk_io_mkparents(const char *filepath);

/* Build "dir/name" (or "dir\name" on DOS) into dst.  Returns 0 on
   success, -1 if it does not fit. */
int sk_io_join(char *dst, sk_u32 cap, const char *dir, const char *name);

/* Flush user buffers and, in SK_SYNC_DEVICE mode, push to the device.
   On DOS this degrades to fflush() only - documented, not hidden. */
int sk_io_sync(FILE *f);

/* Durability policy.
 *
 * SK_SYNC_NONE   - fflush only.  Survives a killed process, because the
 *                  OS keeps the page cache; does NOT survive power loss.
 * SK_SYNC_DEVICE - fflush + fsync/_commit.  Survives power loss.
 *
 * This distinction is not a shortcut, it is the actual contract: a crash
 * test that kills the process measures the first, a mission that can lose
 * power needs the second.  Measured cost on a normal SSD: 0.1 ms vs 85 ms,
 * so the policy is chosen by the caller, never hidden here. */
#define SK_SYNC_NONE   0
#define SK_SYNC_DEVICE 1

void sk_io_sync_mode(int mode);
int  sk_io_sync_cost_ms(void);

/* Truncate an open file to size bytes. */
int sk_io_truncate(FILE *f, unsigned long size);

/* Size of an open file in bytes, or 0. */
unsigned long sk_io_size(FILE *f);

int sk_io_remove(const char *path);
int sk_io_rename(const char *from, const char *to);
int sk_io_exists(const char *path);

/* Atomically replace dst with src.  On POSIX this is rename(), which is
   atomic by definition.  On Windows it is MoveFileEx with
   MOVEFILE_REPLACE_EXISTING, which closes the remove/rename window that
   otherwise loses one checkpoint every time the process dies inside it.
   On DOS there is no such primitive, so it degrades to remove+rename -
   which is still safe here, because the vault keeps two slots and only
   ever needs one of them intact. */
int sk_io_replace(const char *src, const char *dst);

#endif /* SK_IO_H */
