/* sk_vault.c - crash-safe state storage.  See sk_vault.h.
 *
 * Write order that makes a crash harmless:
 *   1. write the older slot into STATE.TMP and push it to the device
 *   2. remove the older slot, rename TMP over it
 *   3. only then truncate the journal
 * A crash before 2 leaves a valid slot untouched.  A crash after 2 leaves
 * a valid snapshot and a journal whose records are older than it, which
 * the replay skips.  There is no window in which both slots are bad.
 *
 * SKYNET M0 / subsystem: core/vault
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "sk_vault.h"
#include "sk_io.h"
#include "sk_math.h"

#define SK_SNAP_MAGIC 0x534B5631UL   /* "SKV1" */
#define SK_WAL_MAGIC  0x534B5731UL   /* "SKW1" */
#define SK_VAULT_VERSION 1UL

#define SK_SNAP_HDR 24
#define SK_WAL_HDR  12

static void sk_put32(sk_u8 *p, sk_u32 v)
{
    p[0] = (sk_u8)(v & 0xFFUL);
    p[1] = (sk_u8)((v >> 8) & 0xFFUL);
    p[2] = (sk_u8)((v >> 16) & 0xFFUL);
    p[3] = (sk_u8)((v >> 24) & 0xFFUL);
}

static sk_u32 sk_get32(const sk_u8 *p)
{
    return SK_U32((sk_u32)p[0]
                | ((sk_u32)p[1] << 8)
                | ((sk_u32)p[2] << 16)
                | ((sk_u32)p[3] << 24));
}

static void sk_state_pack(const sk_state *st, sk_u8 *b)
{
    sk_put32(b +  0, st->seq);
    sk_put32(b +  4, st->prime_index);
    sk_put32(b +  8, st->prime_value);
    sk_put32(b + 12, st->supers);
    sk_put32(b + 16, st->rng[0]);
    sk_put32(b + 20, st->rng[1]);
    sk_put32(b + 24, st->rng[2]);
    sk_put32(b + 28, st->rng[3]);
    sk_put32(b + 32, st->arena_used);
    sk_put32(b + 36, st->out_bytes);
    sk_put32(b + 40, st->reserved);
}

static void sk_state_unpack(sk_state *st, const sk_u8 *b)
{
    st->seq         = sk_get32(b +  0);
    st->prime_index = sk_get32(b +  4);
    st->prime_value = sk_get32(b +  8);
    st->supers      = sk_get32(b + 12);
    st->rng[0]      = sk_get32(b + 16);
    st->rng[1]      = sk_get32(b + 20);
    st->rng[2]      = sk_get32(b + 24);
    st->rng[3]      = sk_get32(b + 28);
    st->arena_used  = sk_get32(b + 32);
    st->out_bytes   = sk_get32(b + 36);
    st->reserved    = sk_get32(b + 40);
}

static int sk_read_exact(FILE *f, void *buf, sk_u32 n)
{
    if (n == 0)
        return 1;
    return ((sk_u32)fread(buf, 1, (size_t)n, f) == n) ? 1 : 0;
}

/* ---- snapshot slot ---------------------------------------------------- */

static int sk_vault_write_slot(const char *path, const sk_state *st)
{
    sk_u8 hdr[SK_SNAP_HDR];
    sk_u8 pay[SK_STATE_BYTES];
    FILE *f;
    sk_u32 hcrc;

    sk_state_pack(st, pay);
    sk_put32(hdr +  0, SK_SNAP_MAGIC);
    sk_put32(hdr +  4, SK_VAULT_VERSION);
    sk_put32(hdr +  8, st->seq);
    sk_put32(hdr + 12, (sk_u32)SK_STATE_BYTES);
    sk_put32(hdr + 16, sk_crc32c(pay, (sk_u32)SK_STATE_BYTES));
    hcrc = sk_crc32c(hdr, 20);
    sk_put32(hdr + 20, hcrc);

    f = fopen(path, "wb");
    if (f == SK_NULL)
        return 0;
    if (fwrite(hdr, 1, (size_t)SK_SNAP_HDR, f) != (size_t)SK_SNAP_HDR) {
        fclose(f);
        return 0;
    }
    if (fwrite(pay, 1, (size_t)SK_STATE_BYTES, f) != (size_t)SK_STATE_BYTES) {
        fclose(f);
        return 0;
    }
    if (sk_io_sync(f) != 0) {
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

static int sk_vault_read_slot(const char *path, sk_state *st)
{
    sk_u8 hdr[SK_SNAP_HDR];
    sk_u8 pay[SK_STATE_BYTES];
    FILE *f;
    sk_u32 len, pcrc;

    f = fopen(path, "rb");
    if (f == SK_NULL)
        return 0;
    if (!sk_read_exact(f, hdr, SK_SNAP_HDR)) {
        fclose(f);
        return 0;
    }
    if (sk_get32(hdr + 0) != SK_SNAP_MAGIC) {
        fclose(f);
        return 0;
    }
    if (sk_get32(hdr + 20) != sk_crc32c(hdr, 20)) {
        fclose(f);
        return 0;
    }
    len = sk_get32(hdr + 12);
    if (len != (sk_u32)SK_STATE_BYTES) {
        fclose(f);
        return 0;
    }
    if (!sk_read_exact(f, pay, len)) {
        fclose(f);
        return 0;
    }
    pcrc = sk_get32(hdr + 16);
    if (pcrc != sk_crc32c(pay, len)) {
        fclose(f);
        return 0;
    }
    fclose(f);
    sk_state_unpack(st, pay);
    return 1;
}

/* ---- journal ---------------------------------------------------------- */

/* Apply every record newer than st->seq.  Stops at the first malformed or
   torn record: that is the tail of an interrupted append, and everything
   before it is already applied. */
static int sk_vault_replay(sk_vault *v, sk_state *st)
{
    sk_u8 hdr[SK_WAL_HDR];
    sk_u8 pay[SK_STATE_BYTES];
    sk_state rec;
    FILE *f;
    int applied;

    f = fopen(v->wal_path, "rb");
    if (f == SK_NULL)
        return 0;

    applied = 0;
    for (;;) {
        if (!sk_read_exact(f, hdr, SK_WAL_HDR))
            break;
        if (sk_get32(hdr + 0) != SK_WAL_MAGIC)
            break;
        if (sk_get32(hdr + 4) != (sk_u32)SK_STATE_BYTES)
            break;
        if (!sk_read_exact(f, pay, (sk_u32)SK_STATE_BYTES))
            break;
        if (sk_get32(hdr + 8) != sk_crc32c(pay, (sk_u32)SK_STATE_BYTES))
            break;
        sk_state_unpack(&rec, pay);
        if (rec.seq > st->seq) {
            *st = rec;
            applied++;
        }
    }
    fclose(f);
    return applied;
}

static int sk_vault_log_raw(sk_vault *v, const sk_state *st)
{
    sk_u8 hdr[SK_WAL_HDR];
    sk_u8 pay[SK_STATE_BYTES];
    FILE *f;

    sk_state_pack(st, pay);
    sk_put32(hdr + 0, SK_WAL_MAGIC);
    sk_put32(hdr + 4, (sk_u32)SK_STATE_BYTES);
    sk_put32(hdr + 8, sk_crc32c(pay, (sk_u32)SK_STATE_BYTES));

    f = fopen(v->wal_path, "ab");
    if (f == SK_NULL)
        return 0;
    if (fwrite(hdr, 1, (size_t)SK_WAL_HDR, f) != (size_t)SK_WAL_HDR) {
        fclose(f);
        return 0;
    }
    if (fwrite(pay, 1, (size_t)SK_STATE_BYTES, f) != (size_t)SK_STATE_BYTES) {
        fclose(f);
        return 0;
    }
    sk_io_sync(f);
    fclose(f);
    v->journaled++;
    return 1;
}

/* ---- public ----------------------------------------------------------- */

int sk_vault_open(sk_vault *v, const char *dir)
{
    if (v == SK_NULL)
        return 0;
    memset(v, 0, sizeof(*v));
    if (dir != SK_NULL && dir[0] != '\0') {
        if (strlen(dir) >= (size_t)SK_VAULT_DIRLEN)
            return 0;
        strcpy(v->dir, dir);
    }
    sk_io_mkdirs(v->dir);

    if (sk_io_join(v->path[0], SK_VAULT_PATHLEN, v->dir, "STATE.A") != 0)
        return 0;
    if (sk_io_join(v->path[1], SK_VAULT_PATHLEN, v->dir, "STATE.B") != 0)
        return 0;
    if (sk_io_join(v->wal_path, SK_VAULT_PATHLEN, v->dir, "WAL.LOG") != 0)
        return 0;
    if (sk_io_join(v->tmp_path, SK_VAULT_PATHLEN, v->dir, "STATE.TMP") != 0)
        return 0;

    v->open = 1;
    return 1;
}

void sk_vault_close(sk_vault *v)
{
    if (v == SK_NULL)
        return;
    if (v->wal != SK_NULL) {
        fclose(v->wal);
        v->wal = SK_NULL;
    }
    v->open = 0;
}

int sk_vault_load(sk_vault *v, sk_state *st)
{
    sk_state best, tmp;
    int i, have;

    if (v == SK_NULL || st == SK_NULL || !v->open)
        return 0;

    have = 0;
    for (i = 0; i < SK_VAULT_SLOTS; i++) {
        if (sk_vault_read_slot(v->path[i], &tmp)) {
            v->slot_valid[i] = 1;
            v->slot_seq[i] = tmp.seq;
            if (!have || tmp.seq > best.seq) {
                best = tmp;
                have = 1;
            }
        } else {
            v->slot_valid[i] = 0;
            v->slot_seq[i] = 0;
        }
    }
    if (!have)
        return 0;

    sk_vault_replay(v, &best);
    *st = best;
    return 1;
}

int sk_vault_snapshot(sk_vault *v, const sk_state *st)
{
    sk_state local;
    int target;

    if (v == SK_NULL || st == SK_NULL || !v->open)
        return 0;

    /* Fold in anything the journal knows that the caller does not, so a
       snapshot can never move the state backwards. */
    local = *st;
    sk_vault_replay(v, &local);

    if (!v->slot_valid[0])
        target = 0;
    else if (!v->slot_valid[1])
        target = 1;
    else
        target = (v->slot_seq[0] <= v->slot_seq[1]) ? 0 : 1;

    if (!sk_vault_write_slot(v->tmp_path, &local))
        return 0;

    /* Atomic where the platform allows it: on POSIX and Windows this is a
       single replace, so there is no moment when the slot is missing.  On
       DOS it degrades to remove+rename, which is why two slots exist. */
    if (sk_io_replace(v->tmp_path, v->path[target]) != 0)
        return 0;

    v->slot_valid[target] = 1;
    v->slot_seq[target] = local.seq;
    v->snapshots++;

    /* Journal is now redundant.  Truncate it only after the snapshot is
       safely in place. */
    {
        FILE *f = fopen(v->wal_path, "wb");
        if (f != SK_NULL) {
            sk_io_sync(f);
            fclose(f);
        }
    }
    return 1;
}

int sk_vault_log(sk_vault *v, const sk_state *st)
{
    if (v == SK_NULL || st == SK_NULL || !v->open)
        return 0;
    return sk_vault_log_raw(v, st);
}

sk_u32 sk_vault_best_seq(const sk_vault *v)
{
    sk_u32 s;

    if (v == SK_NULL)
        return 0;
    s = v->slot_valid[0] ? v->slot_seq[0] : 0;
    if (v->slot_valid[1] && v->slot_seq[1] > s)
        s = v->slot_seq[1];
    return s;
}

/* ---- selftest --------------------------------------------------------- */

#define SK_VAULT_TEST_DIR "skvtest"

static void sk_vault_wipe(void)
{
    sk_io_remove(SK_VAULT_TEST_DIR "/STATE.A");
    sk_io_remove(SK_VAULT_TEST_DIR "/STATE.B");
    sk_io_remove(SK_VAULT_TEST_DIR "/WAL.LOG");
    sk_io_remove(SK_VAULT_TEST_DIR "/STATE.TMP");
}

/* Flip one byte inside the payload of a slot, leaving its CRC intact so
   that only the payload checksum can catch it. */
static void sk_vault_bite(const char *path)
{
    FILE *f;
    long pos;

    f = fopen(path, "r+b");
    if (f == SK_NULL)
        return;
    pos = (long)(SK_SNAP_HDR + 8);
    if (fseek(f, pos, SEEK_SET) == 0) {
        int c = fgetc(f);
        if (c != EOF) {
            fseek(f, pos, SEEK_SET);
            fputc((c ^ 0x5A) & 0xFF, f);
        }
    }
    fclose(f);
}

int sk_vault_selftest(void)
{
    sk_vault v;
    sk_state a, b, got;
    FILE *f;
    int ok;

    ok = 1;
    sk_vault_wipe();

    memset(&a, 0, sizeof(a));
    a.seq = 1;
    a.prime_index = 42;
    a.prime_value = 181;
    a.supers = 5;
    a.rng[0] = 0xDEADBEEFUL;
    a.rng[3] = 7;
    a.out_bytes = 1234;

    if (!sk_vault_open(&v, SK_VAULT_TEST_DIR))
        return 0;

    /* 1. an empty vault reports "nothing to recover" */
    if (sk_vault_load(&v, &got))
        ok = 0;

    /* 2. snapshot, then load it back */
    if (!sk_vault_snapshot(&v, &a))
        ok = 0;
    if (!sk_vault_load(&v, &got))
        ok = 0;
    else if (got.prime_index != 42 || got.prime_value != 181 || got.seq != 1)
        ok = 0;

    /* 3. a journal record newer than the snapshot wins */
    b = a;
    b.seq = 2;
    b.prime_index = 99;
    b.prime_value = 523;
    if (!sk_vault_log(&v, &b))
        ok = 0;
    if (!sk_vault_load(&v, &got))
        ok = 0;
    else if (got.prime_index != 99)
        ok = 0;

    /* 4. second snapshot fills the other slot, so both slots exist */
    if (!sk_vault_snapshot(&v, &b))
        ok = 0;
    if (!v.slot_valid[0] || !v.slot_valid[1])
        ok = 0;

    /* 5. damage the NEWER slot: recovery must fall back to the older one
          instead of returning a corrupt state */
    sk_vault_bite(v.path[v.slot_seq[0] >= v.slot_seq[1] ? 0 : 1]);
    {
        sk_vault v2;
        if (!sk_vault_open(&v2, SK_VAULT_TEST_DIR))
            ok = 0;
        else {
            if (!sk_vault_load(&v2, &got))
                ok = 0;
            else if (got.prime_index != 42)   /* the surviving older state */
                ok = 0;
            sk_vault_close(&v2);
        }
    }

    /* 6. a torn journal tail is normal after a kill: ignore it, do not
          fail the whole recovery */
    {
        char path[SK_VAULT_PATHLEN];
        if (sk_io_join(path, SK_VAULT_PATHLEN, SK_VAULT_TEST_DIR, "WAL.LOG") == 0) {
            f = fopen(path, "ab");
            if (f != SK_NULL) {
                fwrite("SKW1\0\0\0\0", 1, 8, f);   /* header cut in half */
                fclose(f);
            }
        }
    }
    if (!sk_vault_load(&v, &got))
        ok = 0;

    sk_vault_close(&v);
    return ok;
}
