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
#include "sk_mem.h"
#include "sk_tensor.h"
#include "sk_forward.h"
#include "sk_train.h"
#include "sk_selftest.h"

#define SK_VERSION "0.1.0-m0"

/* Output of the last forward step.  Static rather than local: 97 Q16.16
   values are 388 bytes, which is fine on a host and not something to put on
   an 8 KB DOS stack next to everything else. */
static sk_fix bt_scores[SK_T_VOCAB];

typedef struct {
    const char *mode;
    const char *dir;
    const char *out;
    sk_u32 max_primes;
    sk_u32 ckpt_every;
    sk_u32 log_every;
    sk_u32 arena_bytes;
    int    creativity;
    int    layers;
    int    hid;
    int    bits;
    int    epochs;
    sk_i32 lr;
    sk_i32 trust;
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
    printf("  mem        report what memory this machine actually has\n");
    printf("  model FILE load a TNY2 model and report what it contains\n");
    printf("  predict FILE TEXT  run the forward pass and print the scores\n");
    printf("  ask FILE QUESTION  generate an answer (default creativity 0)\n");
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

static void sk_emit_stdout(const char *line)
{
    printf("%s\n", line);
}

/* Run the forward pass over a prompt and print the raw score vector.
 *
 * The output format is deliberately identical to tests/ref_tiny.c, which is
 * the original float implementation. Comparing against my own rewrite would
 * only prove that I made the same mistake twice; comparing against the
 * released code proves the fixed-point port is faithful. */
static int cmd_predict(const sk_opts *o)
{
    sk_model m;
    sk_ctx ctx;
    sk_arena arena;
    void *weights;
    void *kvstore;
    sk_u32 need;
    sk_u32 kvneed;
    sk_u32 got;
    int used_ems;
    char why[64];
    const char *path;
    const char *prompt;
    unsigned char hdr[12];
    FILE *fp;
    int i, p, n;

    setvbuf(stdout, SK_NULL, _IONBF, 0);

    /* Not optional: sk_fix_exp() divides by ln(2), and the constants are
       derived in sk_math_init().  Without this call every constant is zero,
       exp() returns saturation, and the whole forward pass collapses to
       zeros - which is exactly what it did the first time. */
    sk_math_init();

    path = (o->files > 0) ? o->filev[0] : o->out;
    prompt = (o->files > 1) ? o->filev[1] : "hello";

    fp = fopen(path, "rb");
    if (fp == SK_NULL) {
        printf("[sky] cannot open model '%s'\n", path);
        return 6;
    }
    if (fread(hdr, 1, 12, fp) != 12 || memcmp(hdr, "TNY2", 4) != 0) {
        fclose(fp);
        printf("[sky] '%s' is not a TNY2 model\n", path);
        return 6;
    }
    fclose(fp);

    need = sk_model_size_bits((sk_u32)((sk_u32)hdr[4] | ((sk_u32)hdr[5] << 8)),
                         (sk_u32)((sk_u32)hdr[6] | ((sk_u32)hdr[7] << 8)),
                         (sk_u32)((sk_u32)hdr[10] | ((sk_u32)hdr[11] << 8)),
                         97UL, 128UL, o->bits);

    /* The KV cache is sized by the model too: per layer, two int8 tables of
       CTX x D plus two arrays of CTX Q16.16 scales.  Ask for both up front -
       sizing the arena for the weights alone leaves no room for the cache
       and the run dies at the first forward step. */
    {
        sk_u32 D = (sk_u32)((sk_u32)hdr[4] | ((sk_u32)hdr[5] << 8));
        sk_u32 L = (sk_u32)((sk_u32)hdr[6] | ((sk_u32)hdr[7] << 8));

        kvneed = L * (2UL * (sk_u32)SK_T_CTX * D + 8UL * (sk_u32)SK_T_CTX);
    }

    /* Weights and KV cache both want to be big.  On DOS they must go to EMS:
       the arena is capped at 64 KB by a 16-bit size_t, and Tiny-2.1 needs
       372 KB of weights.  On a host the arena holds them. */
    got = 0;
    used_ems = 0;
    if (sk_pool_open(&m.pool, need)) {
        printf("[sky] weights in EMS: %lu KB in %lu pages\n",
               (unsigned long)(m.pool.bytes / 1024UL),
               (unsigned long)m.pool.pages);
        got = m.pool.bytes;
        used_ems = 1;
    } else {
        got = sk_arena_init_best(&arena, need + kvneed + 65536UL,
                                 need + kvneed, SK_MEM_OS);
        if (got == 0) {
            printf("[sky] cannot get %lu bytes (need %lu for weights, "
                   "%lu for the cache)\n",
                   (unsigned long)(need + kvneed),
                   (unsigned long)need, (unsigned long)kvneed);
            return 3;
        }
        weights = sk_arena_alloc(&arena, need);
        if (weights == SK_NULL) {
            /* Never fail silently.  This exact branch swallowed the error on
               DOS: the arena degraded to 64 KB, the weight allocation
               returned NULL, and the program exited with no output at all. */
            printf("[sky] arena gave %lu bytes but the weights need %lu\n",
                   (unsigned long)got, (unsigned long)need);
            sk_arena_done(&arena);
            return 3;
        }
        if (!sk_pool_open_from(&m.pool, need, weights)) {
            printf("[sky] cannot open the weight store\n");
            sk_arena_done(&arena);
            return 3;
        }
        printf("[sky] weights in linear memory: %lu KB\n",
               (unsigned long)(need / 1024UL));
    }

    if (!sk_model_load_shaped(&m, path, &m.pool, o->layers, o->hid, o->bits, why, (sk_u32)sizeof(why))) {
        printf("[sky] load failed: %s\n", why);
        sk_arena_done(&arena);
        return 6;
    }

    kvneed = sk_ctx_kv_bytes(&m);
    kvstore = SK_NULL;
    if (!used_ems) {
        /* the arena path needs storage for the cache as well */
        kvstore = sk_arena_alloc(&arena, kvneed);
        if (kvstore == SK_NULL) {
            printf("[sky] cannot get %lu bytes for the KV cache\n",
                   (unsigned long)kvneed);
            sk_arena_done(&arena);
            return 3;
        }
    }
    if (!sk_ctx_open(&ctx, &m, kvstore, kvneed)) {
        printf("[sky] cannot open the context (cache needs %lu bytes)\n",
               (unsigned long)kvneed);
        sk_arena_done(&arena);
        return 3;
    }

    /* Feed the prompt, one letter at a time, exactly like the original. */
    n = (int)strlen(prompt);
    if (n <= 0 || n >= SK_T_CTX) {
        printf("[sky] prompt must be 1..%d letters\n", SK_T_CTX - 1);
        sk_ctx_close(&ctx);
        sk_arena_done(&arena);
        return 2;
    }
    for (p = 0; p < n; p++) {
        int tok = (unsigned char)prompt[p] - ' ' + 2;
        if (tok < 2)
            tok = 2;
        if (tok >= SK_T_VOCAB)
            tok = SK_T_VOCAB - 1;
        if (!sk_forward_step(&ctx, tok, bt_scores)) {
            printf("[sky] forward pass failed at position %d\n", p);
            sk_ctx_close(&ctx);
            sk_arena_done(&arena);
            return 7;
        }
    }

    /* Same format as tests/ref_tiny.c: "scores" then one value per letter.
       Four decimals is plenty for a diff and avoids the int32 overflow that
       (frac * 1000000) would cause. */
    printf("peak %ld%s\n", (long)ctx.peak, ctx.saturated ? " SATURATED" : "");
    printf("scores");
    for (i = 0; i < SK_T_VOCAB; i++) {
        sk_fix v = bt_scores[i];
        long av = (v < 0) ? -(long)v : (long)v;
        long ip = av >> 16;
        long fr = (((av & 0xFFFFL) * 10000L) >> 16);

        printf(" %s%ld.%04ld", (v < 0) ? "-" : "", ip, fr);
    }
    printf("\n");

    sk_ctx_close(&ctx);
    sk_arena_done(&arena);
    return 0;
}

/* Generate an answer.  Shares the setup with cmd_predict; the difference is
 * the decoding loop, which mirrors TINY.C's pick() so the two can be
 * compared letter for letter at creativity 0. */
static int cmd_ask(const sk_opts *o)
{
    sk_model m;
    sk_ctx ctx;
    sk_arena arena;
    sk_rng rng;
    void *weights;
    void *kvstore;
    sk_u32 need;
    sk_u32 kvneed;
    sk_u32 got;
    int used_ems;
    char why[64];
    const char *path;
    const char *question;
    unsigned char hdr[12];
    FILE *fp;
    char answer[SK_T_CTX + 1];
    int n;

    setvbuf(stdout, SK_NULL, _IONBF, 0);
    sk_math_init();

    path = (o->files > 0) ? o->filev[0] : o->out;
    question = (o->files > 1) ? o->filev[1] : "hello";

    fp = fopen(path, "rb");
    if (fp == SK_NULL) {
        printf("[sky] cannot open model '%s'\n", path);
        return 6;
    }
    if (fread(hdr, 1, 12, fp) != 12 || memcmp(hdr, "TNY2", 4) != 0) {
        fclose(fp);
        printf("[sky] '%s' is not a TNY2 model\n", path);
        return 6;
    }
    fclose(fp);

    need = sk_model_size_bits((sk_u32)((sk_u32)hdr[4] | ((sk_u32)hdr[5] << 8)),
                         (sk_u32)((sk_u32)hdr[6] | ((sk_u32)hdr[7] << 8)),
                         (sk_u32)((sk_u32)hdr[10] | ((sk_u32)hdr[11] << 8)),
                         97UL, 128UL, o->bits);
    {
        sk_u32 D = (sk_u32)((sk_u32)hdr[4] | ((sk_u32)hdr[5] << 8));
        sk_u32 L = (sk_u32)((sk_u32)hdr[6] | ((sk_u32)hdr[7] << 8));

        kvneed = L * (2UL * (sk_u32)SK_T_CTX * D + 8UL * (sk_u32)SK_T_CTX);
    }

    used_ems = 0;
    if (sk_pool_open(&m.pool, need)) {
        used_ems = 1;
        got = m.pool.bytes;
    } else {
        got = sk_arena_init_best(&arena, need + kvneed + 65536UL,
                                 need + kvneed, SK_MEM_OS);
        if (got == 0) {
            printf("[sky] cannot get %lu bytes\n",
                   (unsigned long)(need + kvneed));
            return 3;
        }
        weights = sk_arena_alloc(&arena, need);
        if (weights == SK_NULL) {
            printf("[sky] arena gave %lu bytes but weights need %lu\n",
                   (unsigned long)got, (unsigned long)need);
            sk_arena_done(&arena);
            return 3;
        }
        if (!sk_pool_open_from(&m.pool, need, weights)) {
            printf("[sky] cannot open the weight store\n");
            sk_arena_done(&arena);
            return 3;
        }
    }

    if (!sk_model_load_shaped(&m, path, &m.pool, o->layers, o->hid, o->bits, why, (sk_u32)sizeof(why))) {
        printf("[sky] load failed: %s\n", why);
        sk_arena_done(&arena);
        return 6;
    }

    kvstore = SK_NULL;
    if (!used_ems) {
        kvstore = sk_arena_alloc(&arena, kvneed);
        if (kvstore == SK_NULL) {
            printf("[sky] cannot get %lu bytes for the KV cache\n",
                   (unsigned long)kvneed);
            sk_arena_done(&arena);
            return 3;
        }
    }
    if (!sk_ctx_open(&ctx, &m, kvstore, kvneed)) {
        printf("[sky] cannot open the context\n");
        sk_arena_done(&arena);
        return 3;
    }

    sk_rng_seed(&rng, 0x9E3779B9UL, 0x243F6A88UL, 0xB7E15162UL, 1UL);

    n = sk_generate(&ctx, question, answer, (int)sizeof(answer),
                    o->creativity, &rng);
    if (n < 0) {
        printf("[sky] generation failed\n");
        sk_ctx_close(&ctx);
        sk_arena_done(&arena);
        return 7;
    }

    /* Same line format as tests/ref_tiny.c's generate mode, so the two can
       be diffed directly. */
    printf("answer %s\n", answer);

    sk_ctx_close(&ctx);
    sk_arena_done(&arena);
    return 0;
}

/* Train the output projection on a dataset of "question|answer" lines.
 *
 * Reports the loss before and after, because a trainer that cannot show its
 * own loss going down is a trainer nobody should trust. */
static int cmd_train(const sk_opts *o)
{
    sk_model m;
    sk_train tr;
    sk_arena arena;
    void *weights;
    void *tstore;
    sk_u32 need, tneed, got;
    char why[64];
    const char *path;
    const char *data;
    FILE *fp;
    unsigned char hdr[12];
    int epoch, i;
    sk_fix loss_first = 0, loss_last = 0;

    setvbuf(stdout, SK_NULL, _IONBF, 0);
    sk_math_init();

    path = (o->files > 0) ? o->filev[0] : o->out;
    data = (o->files > 1) ? o->filev[1] : "build/dataset.txt";

    fp = fopen(path, "rb");
    if (fp == SK_NULL) {
        printf("[sky] cannot open model '%s'\n", path);
        return 6;
    }
    if (fread(hdr, 1, 12, fp) != 12 || memcmp(hdr, "TNY2", 4) != 0) {
        fclose(fp);
        printf("[sky] '%s' is not a TNY2 model\n", path);
        return 6;
    }
    fclose(fp);

    need = sk_model_size((sk_u32)((sk_u32)hdr[4] | ((sk_u32)hdr[5] << 8)),
                         (sk_u32)((sk_u32)hdr[6] | ((sk_u32)hdr[7] << 8)),
                         (sk_u32)((sk_u32)hdr[10] | ((sk_u32)hdr[11] << 8)),
                         97UL, 128UL);

    got = sk_arena_init_best(&arena, need + (16UL * 1024UL * 1024UL),
                             need + 1024UL, SK_MEM_OS);
    if (got == 0) {
        printf("[sky] cannot get %lu bytes\n", (unsigned long)need);
        return 3;
    }
    weights = sk_arena_alloc(&arena, need);
    if (weights == SK_NULL || !sk_pool_open_from(&m.pool, need, weights)) {
        printf("[sky] cannot allocate the weight store\n");
        sk_arena_done(&arena);
        return 3;
    }
    if (!sk_model_load(&m, path, &m.pool, why, (sk_u32)sizeof(why))) {
        printf("[sky] load failed: %s\n", why);
        sk_arena_done(&arena);
        return 6;
    }

    tneed = sk_train_bytes(&m);
    tstore = sk_arena_alloc(&arena, tneed);
    if (tstore == SK_NULL || !sk_train_open(&tr, &m, tstore, tneed)) {
        printf("[sky] cannot open the trainer (%lu bytes)\n",
               (unsigned long)tneed);
        sk_arena_done(&arena);
        return 3;
    }

    printf("[sky] training %s on %s, lr=%ld/16777216, epochs=%d\n",
           path, data, (long)o->lr, o->epochs);

    for (epoch = 0; epoch < o->epochs; epoch++) {
        FILE *df = fopen(data, "r");
        int count = 0;
        sk_i32 sum = 0;

        if (df == SK_NULL) {
            printf("[sky] cannot open dataset '%s'\n", data);
            sk_train_close(&tr);
            sk_arena_done(&arena);
            return 6;
        }

        while (!feof(df)) {
            char line[256];
            char q[128], a[128];
            int toks[SK_TRAIN_MAXT];
            int ntok = 0, qlen, alen, k, sep;
            sk_fix loss;

            if (fgets(line, (int)sizeof(line), df) == SK_NULL)
                break;
            sep = -1;
            for (k = 0; line[k] != '\0'; k++) {
                if (line[k] == '|') { sep = k; break; }
            }
            if (sep < 0)
                continue;
            strcpy(q, "");
            strcpy(a, "");
            for (k = 0; k < sep && k < 127; k++)
                q[k] = line[k];
            q[k] = '\0';
            {
                int j = 0;
                for (k = sep + 1; line[k] != '\0' && line[k] != '\n'
                     && line[k] != '\r' && j < 127; k++)
                    a[j++] = line[k];
                a[j] = '\0';
            }
            qlen = (int)strlen(q);
            alen = (int)strlen(a);
            if (qlen < 1 || alen < 1)
                continue;
            if (qlen + alen + 2 > SK_TRAIN_MAXT)
                continue;

            /* tokens: question, then newline (1), then answer, then newline */
            for (k = 0; k < qlen; k++) {
                int tk = (unsigned char)q[k] - 30;
                toks[ntok++] = (tk < 2) ? 2 : ((tk >= SK_T_VOCAB) ? SK_T_VOCAB - 1 : tk);
            }
            toks[ntok++] = 1;
            for (k = 0; k < alen; k++) {
                int tk = (unsigned char)a[k] - 30;
                toks[ntok++] = (tk < 2) ? 2 : ((tk >= SK_T_VOCAB) ? SK_T_VOCAB - 1 : tk);
            }
            toks[ntok++] = 1;

            if (ntok > SK_TRAIN_MAXT || ntok < 3)
                continue;

            loss = sk_train_sequence(&tr, toks, ntok, qlen + 1);
            if (loss == 0)
                continue;
            sk_train_grads(&tr, toks, ntok, qlen + 1);
            sk_train_update(&tr, o->lr, o->trust);

            if (count == 0 && epoch == 0)
                loss_first = loss;
            loss_last = loss;
            sum += loss;
            count++;

            if (o->max_primes != 0 && count >= (int)o->max_primes)
                break;
        }
        fclose(df);

        if (count == 0) {
            printf("[sky] dataset '%s' produced no usable examples\n", data);
            break;
        }
        /* The in-epoch mean is not a clean measure: it mixes examples
           measured before an update with examples measured after one, so it
           changes with the learning rate even on the very first epoch. A
           separate pass with no updates is what actually answers "did this
           epoch help" - otherwise "the loss went down" can be an artefact of
           where in the epoch it happened to be sampled. */
        {
            sk_i32 ev = 0;
            int evn = 0;
            FILE *ef = fopen(data, "r");

            if (ef != SK_NULL) {
                char eline[256];
                while (!feof(ef)) {
                    char eq[128], ea[128];
                    int etoks[SK_TRAIN_MAXT], entok = 0;
                    int elen, ealen, ek, esep, ej;
                    sk_fix eloss;

                    if (fgets(eline, (int)sizeof(eline), ef) == SK_NULL)
                        break;
                    esep = -1;
                    for (ek = 0; eline[ek] != 0; ek++) {
                        if (eline[ek] == '|') { esep = ek; break; }
                    }
                    if (esep < 0)
                        continue;
                    for (ek = 0; ek < esep && ek < 127; ek++)
                        eq[ek] = eline[ek];
                    eq[ek] = 0;
                    ej = 0;
                    for (ek = esep + 1; eline[ek] != 0 && eline[ek] != 10
                         && eline[ek] != 13 && ej < 127; ek++)
                        ea[ej++] = eline[ek];
                    ea[ej] = 0;
                    elen = (int)strlen(eq);
                    ealen = (int)strlen(ea);
                    if (elen < 1 || ealen < 1)
                        continue;
                    if (elen + ealen + 2 > SK_TRAIN_MAXT)
                        continue;
                    for (ek = 0; ek < elen; ek++) {
                        int tk = (unsigned char)eq[ek] - 30;
                        etoks[entok++] = (tk < 2) ? 2
                            : ((tk >= SK_T_VOCAB) ? SK_T_VOCAB - 1 : tk);
                    }
                    etoks[entok++] = 1;
                    for (ek = 0; ek < ealen; ek++) {
                        int tk = (unsigned char)ea[ek] - 30;
                        etoks[entok++] = (tk < 2) ? 2
                            : ((tk >= SK_T_VOCAB) ? SK_T_VOCAB - 1 : tk);
                    }
                    etoks[entok++] = 1;
                    eloss = sk_train_sequence(&tr, etoks, entok, elen + 1);
                    if (eloss == 0)
                        continue;
                    ev += eloss;
                    evn++;
                }
                fclose(ef);
            }
            if (evn > 0)
                printf("[sky] epoch %d: %d examples, loss %ld.%04ld, drift %ld\n",
                       epoch + 1, evn,
                       (long)((ev / evn) >> 16),
                       (long)((((ev / evn) & 0xFFFFL) * 10000L) >> 16),
                       (long)sk_train_drift(&tr));
            else
                printf("[sky] epoch %d: %d examples, loss %ld.%04ld\n",
                       epoch + 1, count,
                       (long)((sum / count) >> 16),
                       (long)((((sum / count) & 0xFFFFL) * 10000L) >> 16));
        }
        (void)i;
    }

    if (loss_first != 0 && loss_last != 0) {
        printf("[sky] first example loss %ld.%04ld -> last %ld.%04ld\n",
               (long)(loss_first >> 16),
               (long)(((loss_first & 0xFFFFL) * 10000L) >> 16),
               (long)(loss_last >> 16),
               (long)(((loss_last & 0xFFFFL) * 10000L) >> 16));
    }

    sk_train_close(&tr);
    sk_arena_done(&arena);
    return 0;
}

static int cmd_mem(void)
{
    sk_meminfo mi;

    setvbuf(stdout, SK_NULL, _IONBF, 0);

    sk_mem_probe(&mi);
    sk_mem_report(&mi, sk_emit_stdout);

    /* The number the model layer needs, said plainly. */
    printf("\n");
    if (!mi.is_dos) {
        printf("paged store: not needed on this host (flat arena)\n");
    } else if (mi.ems_present && mi.ems_free_pages > 0) {
        printf("paged store: %lu KB available through the EMS window\n",
               (unsigned long)(mi.ems_free_pages * 16UL));
        printf("the 64 KB linear ceiling no longer applies to weights\n");
    } else {
        printf("paged store: unavailable, arena stays at the 64 KB ceiling\n");
    }
    return 0;
}

/* Load a TNY2 model and report what came out of it.  This is the M2
   checkpoint: if the weights load and the shape is right, the inference
   code has something real to run on instead of a toy. */
static int cmd_model(const sk_opts *o)
{
    const char *path;
    sk_model m;
    sk_arena arena;
    void *storage;
    char why[64];
    sk_u32 need;
    sk_u32 got;
    unsigned char hdr[12];
    FILE *fp;

    setvbuf(stdout, SK_NULL, _IONBF, 0);

    path = (o->files > 0) ? o->filev[0] : o->out;

    fp = fopen(path, "rb");
    if (fp == SK_NULL) {
        printf("[sky] cannot open model '%s'\n", path);
        return 6;
    }
    if (fread(hdr, 1, 12, fp) != 12 || memcmp(hdr, "TNY2", 4) != 0) {
        fclose(fp);
        printf("[sky] '%s' is not a TNY2 model\n", path);
        return 6;
    }
    fclose(fp);

    need = sk_model_size_bits((sk_u32)((sk_u32)hdr[4] | ((sk_u32)hdr[5] << 8)),
                         (sk_u32)((sk_u32)hdr[6] | ((sk_u32)hdr[7] << 8)),
                         (sk_u32)((sk_u32)hdr[10] | ((sk_u32)hdr[11] << 8)),
                         97UL, 128UL, o->bits);
    printf("[sky] model needs %lu bytes of weight storage\n",
           (unsigned long)need);

    /* The weights live in a pool: EMS on DOS (up to 32 MB), a flat buffer
       on a host.  Asking the arena instead would cap the model at 64 KB on
       DOS, which is less than a fifth of Tiny-2.1. */
    got = 0;
    if (sk_pool_open(&m.pool, need)) {
        got = m.pool.bytes;
        printf("[sky] weights in EMS: %lu KB in %lu pages\n",
               (unsigned long)(got / 1024UL), (unsigned long)m.pool.pages);
    } else {
        got = sk_arena_init_best(&arena, need + 65536UL, need, SK_MEM_OS);
        if (got == 0) {
            printf("[sky] cannot get %lu bytes for the weights\n",
                   (unsigned long)need);
            return 3;
        }
        storage = sk_arena_alloc(&arena, need);
        if (storage == SK_NULL) {
            printf("[sky] arena cannot provide %lu bytes\n",
                   (unsigned long)need);
            sk_arena_done(&arena);
            return 3;
        }
        if (!sk_pool_open_from(&m.pool, need, storage)) {
            printf("[sky] cannot open the weight store\n");
            sk_arena_done(&arena);
            return 3;
        }
        printf("[sky] weights in linear memory: %lu KB\n",
               (unsigned long)(need / 1024UL));
    }

    if (!sk_model_load_shaped(&m, path, &m.pool, o->layers, o->hid, o->bits, why, (sk_u32)sizeof(why))) {
        printf("[sky] load failed: %s\n", why);
        sk_arena_done(&arena);
        return 6;
    }

    printf("[sky] loaded  D=%d L=%d H=%d HID=%d HD=%d\n",
           m.D, m.L, m.H, m.HID, m.HD);
    printf("[sky] weights %lu bytes, %lu used, att_div=%ld\n",
           (unsigned long)need, (unsigned long)m.used, (long)m.att_div);
    printf("[sky] tables  emb %lux%lu  pos %lux%lu  f1 %lux%lu  f2 %lux%lu\n",
           (unsigned long)m.emb.rows, (unsigned long)m.emb.cols,
           (unsigned long)m.pos.rows, (unsigned long)m.pos.cols,
           (unsigned long)m.f1[0].rows, (unsigned long)m.f1[0].cols,
           (unsigned long)m.f2[0].rows, (unsigned long)m.f2[0].cols);
    { sk_fix g1row[SK_T_MAXD]; sk_fix gfrow[SK_T_MAXD];
      if (sk_model_gain_row(&m, SK_GAIN_G1, 0, g1row)
          && sk_model_gain_row(&m, SK_GAIN_GF, 0, gfrow))
          printf("[sky] gains   g1[0][0]=%ld gf[0]=%ld\n", (long)g1row[0], (long)gfrow[0]);
      else
          printf("[sky] gains   UNREADABLE\n"); }

    /* Prove the weights landed byte for byte.  The loader checksums the
       weights while parsing the file; recomputing that checksum by reading
       the store back proves the storage path is lossless.  The format is
       therefore parsed in exactly one place: an earlier version of this
       verification walked the file a second time and got the position of
       the g2 gains wrong, which reported a false mismatch on a correct
       loader. */
    {
        sk_u32 crc_back = sk_model_weights_crc(&m);

        printf("[sky] weights crc  load %08lX  store %08lX  %s\n",
               (unsigned long)m.weights_crc, (unsigned long)crc_back,
               (crc_back == m.weights_crc) ? "MATCH" : "MISMATCH");
        if (crc_back != m.weights_crc) {
            sk_arena_done(&arena);
            return 7;
        }
    }
    /* One real matvec, so the loader is proven to produce usable numbers
       and not just to have consumed the right byte count. */
    {
        sk_fix x[SK_T_MAXD];
        sk_fix y[SK_T_MAXD];
        int i;
        int ok;
        sk_fix peak;
        int nz;

        for (i = 0; i < m.D; i++)
            x[i] = SK_FIX_ONE;
        ok = sk_table_matvec(&m.wq[0], x, y);

        peak = 0;
        nz = 0;
        for (i = 0; i < m.D; i++) {
            sk_fix a = (y[i] < 0) ? -y[i] : y[i];
            if (a > peak)
                peak = a;
            if (y[i] != 0)
                nz++;
        }
        printf("[sky] matvec  %s, %d/%d non-zero, peak |y| = %ld.%03ld\n",
               ok ? "ok" : "FAILED", nz, m.D,
               (long)(peak >> 16),
               /* three decimals, not five: (frac * 100000) overflows int32
                  for any frac above ~21474, which printed 4.116 for a true
                  4.78 and looked like a maths bug for a while. */
               (long)(((peak & 0xFFFFL) * 1000L) >> 16));
        if (!ok) {
            sk_arena_done(&arena);
            return 7;
        }
    }

    sk_arena_done(&arena);
    return 0;
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
static int cmd_ingest(const sk_opts *o)
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
    o.creativity = 0;
    o.layers = 0;
    o.hid = 0;
    o.bits = 8;
    o.epochs = 1;
    o.lr = 16777;   /* 0.001 in Q8.24; measured best of 1e-5 .. 0.25 */
    o.trust = 167772;  /* 0.01 in Q8.24: at most 1% of the scale per step */
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
        } else if (strcmp(a, "--epochs") == 0 && i + 1 < argc) {
            o.epochs = (int)strtol(argv[++i], SK_NULL, 10);
        } else if (strcmp(a, "--lr") == 0 && i + 1 < argc) {
            o.lr = (sk_fix)strtol(argv[++i], SK_NULL, 10);
        } else if (strcmp(a, "--bits") == 0 && i + 1 < argc) {
            o.bits = (int)strtol(argv[++i], SK_NULL, 10);
        } else if (strcmp(a, "--hid") == 0 && i + 1 < argc) {
            o.hid = (int)strtol(argv[++i], SK_NULL, 10);
        } else if (strcmp(a, "--layers") == 0 && i + 1 < argc) {
            o.layers = (int)strtol(argv[++i], SK_NULL, 10);
        } else if (strcmp(a, "--creativity") == 0 && i + 1 < argc) {
            o.creativity = (int)strtol(argv[++i], SK_NULL, 10);
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
                       || strcmp(a, "mem") == 0
                       || strcmp(a, "model") == 0
                       || strcmp(a, "predict") == 0
                       || strcmp(a, "ask") == 0
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
    if (strcmp(o.mode, "mem") == 0)
        return cmd_mem();
    if (strcmp(o.mode, "model") == 0)
        return cmd_model(&o);
    if (strcmp(o.mode, "predict") == 0)
        return cmd_predict(&o);
    if (strcmp(o.mode, "ask") == 0)
        return cmd_ask(&o);
    if (strcmp(o.mode, "recover") == 0)
        return cmd_recover(&o);
    if (strcmp(o.mode, "train") == 0)
        return cmd_train(&o);
    if (strcmp(o.mode, "ingest") == 0)
        return cmd_ingest(&o);
    if (strcmp(o.mode, "run") != 0) {
        printf("[sky] unknown mode %s\n", o.mode);
        return 2;
    }
    return cmd_run(&o);
}
