/* sk_vault.h - crash-safe state storage.
 *
 * Two snapshot slots plus an append-only journal.  The rule is simple and
 * is the whole 50-year story: a run may lose at most the work done since
 * the last checkpoint, and it must never resume from a corrupt state.
 *
 * Layout inside the state directory:
 *   STATE.A   snapshot slot A
 *   STATE.B   snapshot slot B
 *   WAL.LOG   append-only journal since the newest snapshot
 *   STATE.TMP scratch slot, ignored on recovery
 *
 * Every record is length-prefixed, little-endian, and CRC32C checked.
 * A torn record at the tail of the journal is normal and is discarded.
 * A torn snapshot is never selected, because the other slot is intact.
 *
 * SKYNET M0 / subsystem: core/vault
 */
#ifndef SK_VAULT_H
#define SK_VAULT_H

#include <stdio.h>
#include "sk_types.h"

#define SK_VAULT_SLOTS   2
#define SK_VAULT_DIRLEN  128
#define SK_VAULT_PATHLEN 192

/* The complete resumable state of a SkyNet run.  Serialised as eight
   little-endian 32-bit words so that a state file written by an 8088 is
   readable by x86_64 and the other way round. */
typedef struct {
    sk_u32 seq;          /* monotone record counter                */
    sk_u32 prime_index;  /* Prime Clock: how many primes emitted   */
    sk_u32 prime_value;  /* Prime Clock: last value emitted        */
    sk_u32 supers;       /* super-primes crossed                   */
    sk_u32 rng[4];       /* xoshiro state                          */
    sk_u32 arena_used;   /* arena high-water, diagnostics          */
    sk_u32 out_bytes;    /* durable bytes in the output file       */
    sk_u32 reserved;
} sk_state;

#define SK_STATE_WORDS 11
#define SK_STATE_BYTES (SK_STATE_WORDS * 4)

typedef struct {
    char  dir[SK_VAULT_DIRLEN];
    char  path[SK_VAULT_SLOTS][SK_VAULT_PATHLEN];
    char  wal_path[SK_VAULT_PATHLEN];
    char  tmp_path[SK_VAULT_PATHLEN];
    FILE *wal;
    sk_u32 slot_seq[SK_VAULT_SLOTS];
    int    slot_valid[SK_VAULT_SLOTS];
    sk_u32 snapshots;
    sk_u32 journaled;
    int    open;
} sk_vault;

int  sk_vault_open(sk_vault *v, const char *dir);
void sk_vault_close(sk_vault *v);

/* Load the newest valid snapshot and replay the journal on top of it.
   Returns 1 if a state was recovered, 0 if the vault is empty. */
int  sk_vault_load(sk_vault *v, sk_state *st);

/* Durably replace the older slot, then reset the journal.  The other
   slot keeps the previous good state at all times. */
int  sk_vault_snapshot(sk_vault *v, const sk_state *st);

/* Append one state to the journal.  Cheap; call it more often than
   sk_vault_snapshot(). */
int  sk_vault_log(sk_vault *v, const sk_state *st);

/* Diagnostics. */
sk_u32 sk_vault_best_seq(const sk_vault *v);
int    sk_vault_selftest(void);

#endif /* SK_VAULT_H */
