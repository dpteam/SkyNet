/* main.c - SkyNet, M0.
 *
 * One program, two modules (see PROMPT.md line 25):
 *   run      - runtime: the Prime Clock heartbeat with checkpointing
 *   train    - trainer: ingest files into a dataset manifest
 * plus the two tools that keep the thing honest:
 *   recover  - print what the vault would resume from
 *   selftest - assert every invariant the core depends on
 *
 * SKYNET M0 / subsystem: entry
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sk_types.h"
#include "sk_math.h"
#include "sk_arena.h"
#include "sk_clock.h"
#include "sk_vault.h"
#include "sk_io.h"
#include "sk_selftest.h"

#define SK_VERSION "0.1.0-m0"

typedef struct {
    const char *mode;
    const char *dir;
    const char *out;
    sk_u32 max_primes;
    sk_u32 ckpt_every;
    sk_u32 log_every;
    sk_u32 arena_bytes;
    int    quiet;
    int    durable;
    int    seed_given;
    sk_u32 seed[4];
    int    files;
    char  *filev[64];
} sk_opts;

static void sk_usage(void)
{
    printf("SkyNet %s - a self-contained agent core for retro targets\n", SK_VERSION);
    printf("\n");
    printf("usage: skynet [mode] [options] [files...]\n");
    printf("\n");
    printf("modes\n");
    printf("  run        runtime: prime clock, checkpoint on every super-prime\n");
    printf("  train      ingest the listed files into a dataset manifest\n");
    printf("  recover    show the state the vault would resume from\n");
    printf("  selftest   check every core invariant\n");
    printf("\n");
    printf("options\n");
    printf("  --dir PATH     state directory          (default: state)\n");
    printf("  --out PATH     primes output file       (default: primes.txt)\n");
    printf("  --max N        stop after N primes      (0 = never)\n");
    printf("  --ckpt K       snapshot every K-th super-prime (default 64)\n");
    printf("  --log N        journal every N primes   (default 1024)\n");
    printf("  --mem BYTES    arena size               (default 8388608)\n");
    printf("  --durable      fsync snapshots (survives power loss, ~85ms each)\n");
    printf("  --seed A,B,C,D rng seed (four hex or decimal words)\n");
    printf("  --quiet        do not echo primes to stdout\n");
    printf("  --help         this text\n");
}

static int sk_parse_seed(const char *s, sk_u32 *out)
{
    int i;
    const char *p;
    char buf[16];

    p = s;
    for (i = 0; i < 4; i++) {
        int n = 0;
        while (*p != '\0' && *p != ',' && n < 15)
            buf[n++] = *p++;
        buf[n] = '\0';
        if (n == 0)
            return 0;
        if (buf[0] == '0' && (buf[1] == 'x' || buf[1] == 'X'))
            out[i] = SK_U32(strtoul(buf + 2, SK_NULL, 16));
        else
            out[i] = SK_U32(strtoul(buf, SK_NULL, 10));
        if (*p == ',')
            p++;
    }
    return 1;
}

/* ------------------------------------------------------------------ */

static int cmd_selftest(void)
{
    int fails;

    /* Unbuffered on purpose.  A self-test exists to find crashes, and a
       buffered stdout loses everything when the program dies - which is
       exactly what happened on the 16-bit DOS build: the run produced no
       output at all, so it looked like it never started.  With no buffer,
       the last line printed is the last check that ran. */
    setvbuf(stdout, SK_NULL, _IONBF, 0);

    fails = sk_selftest_run();
    return (fails == 0) ? 0 : 1;
}

static int cmd_recover(const sk_opts *o)
{
    sk_vault v;
    sk_state st;

    sk_math_init();
    if (!sk_vault_open(&v, o->dir)) {
        printf("[sky] cannot open state directory '%s'\n", o->dir);
        return 4;
    }
    if (!sk_vault_load(&v, &st)) {
        printf("[sky] no state in '%s' - nothing to recover\n", o->dir);
        sk_vault_close(&v);
        return 1;
    }
    printf("state directory : %s\n", o->dir);
    printf("record seq      : %lu\n", (unsigned long)st.seq);
    printf("prime index     : %lu\n", (unsigned long)st.prime_index);
    printf("prime value     : %lu\n", (unsigned long)st.prime_value);
    printf("super-primes    : %lu\n", (unsigned long)st.supers);
    printf("output bytes    : %lu\n", (unsigned long)st.out_bytes);
    printf("arena high      : %lu\n", (unsigned long)st.arena_used);
    printf("rng state       : %08lX %08lX %08lX %08lX\n",
           (unsigned long)st.rng[0], (unsigned long)st.rng[1],
           (unsigned long)st.rng[2], (unsigned long)st.rng[3]);
    printf("slots           : A=%s B=%s\n",
           v.slot_valid[0] ? "ok" : "empty",
           v.slot_valid[1] ? "ok" : "empty");
    sk_vault_close(&v);
    return 0;
}

static void sk_state_fill(sk_state *st, const sk_clock *clk, const sk_rng *rng,
                          const sk_arena *arena, unsigned long written)
{
    st->seq++;
    st->prime_index = clk->index;
    st->prime_value = clk->value;
    st->supers = clk->supers;
    st->rng[0] = rng->s[0];
    st->rng[1] = rng->s[1];
    st->rng[2] = rng->s[2];
    st->rng[3] = rng->s[3];
    st->arena_used = arena->high;
    st->out_bytes = SK_U32(written);
}

static int cmd_run(const sk_opts *o)
{
    sk_arena arena;
    sk_clock clk;
    sk_vault vault;
    sk_state st;
    sk_rng rng;
    FILE *out;
    unsigned long written;
    char line[24];
    int recovered;
    sk_u32 got;
    sk_u32 checkpoints;
    sk_u32 journals;

    sk_math_init();

    /* Durability policy: by default we protect against a killed process
       (which is what the crash test measures) but not against power loss,
       because fsync costs 85 ms here against 0.1 ms of real work.  Pass
       --durable on a mission that can actually lose power. */
    sk_io_sync_mode(o->durable ? SK_SYNC_DEVICE : SK_SYNC_NONE);

    /* Degrade rather than fail: on a 640 KB DOS machine the default 8 MB
       request cannot be met, and an error message would be the wrong
       answer.  Halve until something fits, down to 32 KB. */
    got = sk_arena_init_best(&arena, o->arena_bytes, 32UL * 1024UL, SK_MEM_OS);
    if (got == 0) {
        printf("[sky] arena allocation failed, even 32 KB\n");
        return 3;
    }
    if (got != o->arena_bytes)
        printf("[sky] arena: %lu of %lu bytes granted\n",
               (unsigned long)got, (unsigned long)o->arena_bytes);

    sk_clock_init(&clk);
    if (!sk_vault_open(&vault, o->dir)) {
        printf("[sky] cannot open state directory '%s'\n", o->dir);
        return 4;
    }

    memset(&st, 0, sizeof(st));
    recovered = sk_vault_load(&vault, &st);
    if (recovered) {
        sk_clock_seek(&clk, st.prime_value, st.prime_index);
        clk.supers = st.supers;
        sk_rng_seed(&rng, st.rng[0], st.rng[1], st.rng[2], st.rng[3]);
        printf("[sky] resumed at prime #%lu = %lu (seq %lu)\n",
               (unsigned long)st.prime_index, (unsigned long)st.prime_value,
               (unsigned long)st.seq);
    } else {
        if (o->seed_given) {
            sk_rng_seed(&rng, o->seed[0], o->seed[1], o->seed[2], o->seed[3]);
        } else {
            sk_rng_seed(&rng, 0x9E3779B9UL, 0x243F6A88UL,
                        0xB7E15162UL, 0x00000001UL);
        }
        printf("[sky] fresh start\n");
    }

    sk_io_mkparents(o->out);
    out = fopen(o->out, "r+b");
    if (out == SK_NULL)
        out = fopen(o->out, "wb");
    if (out == SK_NULL) {
        printf("[sky] cannot open output '%s'\n", o->out);
        sk_vault_close(&vault);
        return 5;
    }

    if (recovered) {
        unsigned long have = sk_io_size(out);
        if (have < (unsigned long)st.out_bytes) {
            /* The state claims more output than the device kept.  This is
               the power-loss case, not the process-kill case: fall back to
               what actually exists and say so out loud. */
            printf("[sky] warning: output has %lu bytes, state expects %lu\n",
                   have, (unsigned long)st.out_bytes);
            st.out_bytes = SK_U32(have);
        }
        sk_io_truncate(out, (unsigned long)st.out_bytes);
        written = (unsigned long)st.out_bytes;
    } else {
        sk_io_truncate(out, 0);
        written = 0;
    }
    fseek(out, 0L, SEEK_END);

    checkpoints = 0;
    journals = 0;

    for (;;) {
        int n;

        if (o->max_primes != 0 && clk.index >= o->max_primes)
            break;
        if (!sk_clock_next(&clk))
            break;

        n = sprintf(line, "%lu\n", (unsigned long)clk.value);
        fwrite(line, 1, (size_t)n, out);
        written += (unsigned long)n;
        if (!o->quiet)
            fputs(line, stdout);

        if (clk.is_super && o->ckpt_every != 0
            && (clk.supers % o->ckpt_every) == 0) {
            sk_io_sync(out);
            sk_rng_next(&rng);            /* the state must move to prove it */
            sk_state_fill(&st, &clk, &rng, &arena, written);
            if (sk_vault_snapshot(&vault, &st))
                checkpoints++;
        } else if (o->log_every != 0 && (clk.index % o->log_every) == 0) {
            sk_io_sync(out);
            sk_state_fill(&st, &clk, &rng, &arena, written);
            if (sk_vault_log(&vault, &st))
                journals++;
        }
    }

    /* final checkpoint so a clean exit costs nothing on restart */
    sk_io_sync(out);
    sk_state_fill(&st, &clk, &rng, &arena, written);
    sk_vault_snapshot(&vault, &st);
    fclose(out);
    sk_vault_close(&vault);

    printf("\n[sky] stopped at prime #%lu = %lu, %lu super-primes, "
           "%lu checkpoints, %lu journal records, %lu bytes out\n",
           (unsigned long)clk.index, (unsigned long)clk.value,
           (unsigned long)clk.supers, (unsigned long)checkpoints,
           (unsigned long)journals, written);
    printf("[sky] state: %s   output: %s\n", o->dir, o->out);

    sk_arena_done(&arena);
    return 0;
}

/* Ingest mode.  This is the seed of M4: the same walk will feed the
   trainer, and the same CRC is what makes "did this file already go in?"
   answerable without keeping the file. */
static int cmd_train(const sk_opts *o)
{
    int i;
    unsigned long total;
    char path[256];
    FILE *mf;

    sk_math_init();
    total = 0;
    if (sk_io_join(path, 256, o->dir, "DATASET.IDX") != 0) {
        printf("[sky] path too long\n");
        return 6;
    }
    sk_io_mkdirs(o->dir);
    mf = fopen(path, "wb");
    if (mf == SK_NULL) {
        printf("[sky] cannot write %s\n", path);
        return 6;
    }
    fprintf(mf, "SKYNET-DATASET 1\n");

    for (i = 0; i < o->files; i++) {
        FILE *f;
        unsigned char buf[4096];
        size_t got;
        sk_u32 crc;
        unsigned long bytes;

        f = fopen(o->filev[i], "rb");
        if (f == SK_NULL) {
            printf("[sky] skip (unreadable): %s\n", o->filev[i]);
            continue;
        }
        bytes = 0;
        crc = 0;
        for (;;) {
            got = fread(buf, 1, sizeof(buf), f);
            if (got == 0)
                break;
            crc = sk_crc32c_update(crc, buf, (sk_u32)got);
            bytes += (unsigned long)got;
        }
        fclose(f);
        fprintf(mf, "%08lX %lu %s\n", (unsigned long)crc, bytes, o->filev[i]);
        printf("[sky] ingested %8lu bytes  crc %08lX  %s\n",
               bytes, (unsigned long)crc, o->filev[i]);
        total += bytes;
    }
    fclose(mf);
    printf("[sky] %d file(s), %lu bytes, manifest: %s\n",
           o->files, total, path);
    return 0;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    sk_opts o;
    int i;

    memset(&o, 0, sizeof(o));
    o.mode = "run";
    o.dir = "state";
    o.out = "primes.txt";
    o.max_primes = 0;
    o.ckpt_every = 64;
    o.log_every = 1024;
    o.durable = 0;
    o.arena_bytes = 8UL * 1024UL * 1024UL;

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];

        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            sk_usage();
            return 0;
        } else if (strcmp(a, "--quiet") == 0) {
            o.quiet = 1;
        } else if (strcmp(a, "--dir") == 0 && i + 1 < argc) {
            o.dir = argv[++i];
        } else if (strcmp(a, "--out") == 0 && i + 1 < argc) {
            o.out = argv[++i];
        } else if (strcmp(a, "--max") == 0 && i + 1 < argc) {
            o.max_primes = SK_U32(strtoul(argv[++i], SK_NULL, 10));
        } else if (strcmp(a, "--ckpt") == 0 && i + 1 < argc) {
            o.ckpt_every = SK_U32(strtoul(argv[++i], SK_NULL, 10));
        } else if (strcmp(a, "--log") == 0 && i + 1 < argc) {
            o.log_every = SK_U32(strtoul(argv[++i], SK_NULL, 10));
        } else if (strcmp(a, "--mem") == 0 && i + 1 < argc) {
            o.arena_bytes = SK_U32(strtoul(argv[++i], SK_NULL, 10));
        } else if (strcmp(a, "--durable") == 0) {
            o.durable = 1;
        } else if (strcmp(a, "--seed") == 0 && i + 1 < argc) {
            if (sk_parse_seed(argv[++i], o.seed))
                o.seed_given = 1;
        } else if (a[0] == '-') {
            printf("[sky] unknown option %s\n", a);
            sk_usage();
            return 2;
        } else if (o.files == 0
                   && (strcmp(a, "run") == 0 || strcmp(a, "train") == 0
                       || strcmp(a, "recover") == 0
                       || strcmp(a, "selftest") == 0)) {
            o.mode = a;
        } else if (o.files < 64) {
            o.filev[o.files++] = argv[i];
        } else {
            printf("[sky] too many files\n");
            return 2;
        }
    }

    if (strcmp(o.mode, "selftest") == 0)
        return cmd_selftest();
    if (strcmp(o.mode, "recover") == 0)
        return cmd_recover(&o);
    if (strcmp(o.mode, "train") == 0) {
        if (o.files == 0) {
            printf("[sky] train needs at least one file\n");
            return 2;
        }
        return cmd_train(&o);
    }
    if (strcmp(o.mode, "run") != 0) {
        printf("[sky] unknown mode %s\n", o.mode);
        return 2;
    }
    return cmd_run(&o);
}
