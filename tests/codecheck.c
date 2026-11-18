/* codecheck.c - does a code update actually improve the model, and is the
 * result still a well-formed model?
 *
 * Three claims, and a training loop is only worth having if all three hold:
 *
 *   1. The weights MOVE. Trivially true in principle, and the exact failure
 *      this design exists to avoid is the opposite: a step smaller than the
 *      master's resolution moves nothing and the loss merely stops falling.
 *      So this reports how many elements changed and fails on zero.
 *
 *   2. The loss on the example that was trained on goes DOWN. One example,
 *      one step, no averaging to hide behind. A sign error makes it rise; a
 *      wrong scale makes it diverge. This is the same test tools/train_check
 *      applies to the row scales, applied to the weights.
 *
 *   3. The model is still a model. After the update, every table must expand
 *      into masters and compress back to exactly the codes that are stored -
 *      the invariant proved for an untouched file must survive a write. If
 *      it did not, a trained model would be one that cannot be reloaded into
 *      the same numbers, which is the worst kind of bug: it looks like the
 *      training worked.
 *
 * Build: cc -O2 -Isrc -o build/codecheck tests/codecheck.c <all src files>
 * Run:   build/codecheck tests/fixtures/TINY.BIN tests/fixtures/dataset.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sk_types.h"
#include "sk_math.h"
#include "sk_tensor.h"
#include "sk_forward.h"
#include "sk_train.h"
#include "sk_shadow.h"
#include "sk_arena.h"

#define MAXTOK 64

static int tokenize(const char *line, int *toks, int *answer_from)
{
    char q[128], a[128];
    int ntok = 0, qlen, alen, k, sep, j;

    sep = -1;
    for (k = 0; line[k] != '\0' && line[k] != '\n' && line[k] != '\r'; k++) {
        if (line[k] == '|') { sep = k; break; }
    }
    if (sep < 0)
        return 0;
    for (k = 0; k < sep && k < 127; k++)
        q[k] = line[k];
    q[k] = '\0';
    j = 0;
    for (k = sep + 1; line[k] && line[k] != '\n' && line[k] != '\r' && j < 127; k++)
        a[j++] = line[k];
    a[j] = '\0';

    qlen = (int)strlen(q);
    alen = (int)strlen(a);
    if (qlen < 1 || alen < 1 || qlen + alen + 2 > MAXTOK)
        return 0;

    for (k = 0; k < qlen; k++) {
        int t = (unsigned char)q[k] - 30;
        toks[ntok++] = (t < 2) ? 2 : ((t >= SK_T_VOCAB) ? SK_T_VOCAB - 1 : t);
    }
    toks[ntok++] = 1;
    for (k = 0; k < alen; k++) {
        int t = (unsigned char)a[k] - 30;
        toks[ntok++] = (t < 2) ? 2 : ((t >= SK_T_VOCAB) ? SK_T_VOCAB - 1 : t);
    }
    toks[ntok++] = 1;

    *answer_from = qlen + 1;
    return ntok;
}

/* Every table expands and recompresses to itself. Same claim as
 * `skynet shadow`, checked here because the point is that it survives a
 * write, which `skynet shadow` cannot show - it only ever sees a file that
 * was never trained. */
static int tables_consistent(sk_model *m, sk_i32 *mrow, long *bad_out)
{
    int l, bad = 0;
    long total_bad = 0;

    for (l = -1; l < m->L; l++) {
        sk_table *set[6];
        const char *names[6];
        int nt, t;

        if (l < 0) {
            set[0] = &m->emb;  names[0] = "emb";
            set[1] = &m->pos;  names[1] = "pos";
            nt = 2;
        } else {
            set[0] = &m->wq[l]; names[0] = "q";
            set[1] = &m->wk[l]; names[1] = "k";
            set[2] = &m->wv[l]; names[2] = "v";
            set[3] = &m->wo[l]; names[3] = "o";
            set[4] = &m->f1[l]; names[4] = "f1";
            set[5] = &m->f2[l]; names[5] = "f2";
            nt = 6;
        }

        for (t = 0; t < nt; t++) {
            sk_table *tab = set[t];
            sk_u32 r, j;

            for (r = 0; r < tab->rows; r++) {
                static signed char before[SK_T_MAXHID];
                static signed char after[SK_T_MAXHID];
                sk_fix s0, s1;

                if (!sk_table_scale_read(tab, r, &s0))
                    return 0;
                if (!sk_table_read_row(tab, r, before))
                    return 0;
                if (!sk_shadow_row_load(tab, r, mrow))
                    return 0;
                if (!sk_shadow_row_store(tab, r, mrow))
                    return 0;
                if (!sk_table_scale_read(tab, r, &s1))
                    return 0;
                if (!sk_table_read_row(tab, r, after))
                    return 0;

                if (s0 != s1) {
                    printf("  %s L%d row %lu scale %ld -> %ld\n", names[t], l,
                           (unsigned long)r, (long)s0, (long)s1);
                    total_bad++;
                    bad++;
                }
                for (j = 0; j < tab->cols; j++) {
                    if (before[j] != after[j]) {
                        if (bad < 5)
                            printf("  %s L%d row %lu col %lu code %d -> %d\n",
                                   names[t], l, (unsigned long)r,
                                   (unsigned long)j, (int)before[j],
                                   (int)after[j]);
                        total_bad++;
                        bad++;
                    }
                }
            }
        }
    }
    *bad_out = total_bad;
    return bad == 0;
}

int main(int argc, char **argv)
{
    const char *model = argc > 1 ? argv[1] : "tests/fixtures/TINY.BIN";
    const char *data = argc > 2 ? argv[2] : "tests/fixtures/dataset.txt";
    sk_model m;
    sk_train tr;
    sk_arena arena;
    void *weights, *tstore;
    sk_u32 need, tneed;
    sk_i32 *mrow;
    char why[64];
    unsigned char hdr[12];
    FILE *fp;
    char line[256];
    int toks[MAXTOK], ntok, af;
    sk_fix loss0, loss1;
    sk_i32 lr, trust;
    long bad = 0;
    int moved, moved2;
    sk_u32 crc0, crc1;

    sk_math_init();

    fp = fopen(model, "rb");
    if (!fp) { printf("cannot open %s\n", model); return 2; }
    if (fread(hdr, 1, 12, fp) != 12) { fclose(fp); return 2; }
    fclose(fp);

    need = sk_model_size((sk_u32)(hdr[4] | (hdr[5] << 8)),
                         (sk_u32)(hdr[6] | (hdr[7] << 8)),
                         (sk_u32)(hdr[10] | (hdr[11] << 8)),
                         97UL, 128UL);
    if (!sk_arena_init(&arena, need + (32UL * 1024UL * 1024UL), SK_MEM_OS)) {
        printf("arena failed\n");
        return 2;
    }
    weights = sk_arena_alloc(&arena, need);
    if (!weights || !sk_pool_open_from(&m.pool, need, weights)) return 2;
    if (!sk_model_load(&m, model, &m.pool, why, (sk_u32)sizeof(why))) {
        printf("load failed: %s\n", why);
        return 2;
    }

    mrow = (sk_i32 *)sk_arena_alloc(&arena, (sk_u32)SK_T_MAXHID * 4UL);
    if (mrow == SK_NULL) {
        printf("no room for a row of masters\n");
        return 2;
    }

    tneed = sk_train_bytes(&m, SK_TRAIN_CODES);
    tstore = sk_arena_alloc(&arena, tneed);
    if (!tstore || !sk_train_open(&tr, &m, tstore, tneed, SK_TRAIN_CODES)) {
        printf("trainer open failed\n");
        return 2;
    }

    fp = fopen(data, "r");
    if (!fp) { printf("cannot open %s\n", data); return 2; }
    if (!fgets(line, (int)sizeof(line), fp)) { fclose(fp); return 2; }
    fclose(fp);
    ntok = tokenize(line, toks, &af);
    if (ntok == 0) { printf("unparseable first line\n"); return 2; }

    /* A learning rate big enough for the weights. The rate that suits row
       scales is a thousand times too small here, and the reason is
       arithmetic: dL/dW is around 4e-4 where dL/dscale is around 0.1, and
       the master is not divided by 127 the way a code is. */
    lr = 167772;                    /* 0.01 in Q8.24 */
    trust = 167772;                 /* 1% of the row maximum per step */

    printf("code training on %d tokens, answer from %d\n", ntok, af);
    crc0 = sk_model_weights_crc(&m);
    printf("  weights CRC before %08lX\n", (unsigned long)crc0);

    loss0 = sk_train_sequence(&tr, toks, ntok, af);
    printf("  loss before %.4f\n", (double)loss0 / 65536.0);

    /* ---- claim 1: the weights move ---- */
    sk_train_zero_grads(&tr);
    sk_train_sequence(&tr, toks, ntok, af);
    sk_train_grads(&tr, toks, ntok, af);
    sk_train_update(&tr, lr, trust);
    moved = sk_train_codes_changed(&tr);
    printf("  %d weight elements moved\n", moved);
    if (moved == 0) {
        printf("FAILED: nothing changed, so the step is below the master's"
               " resolution\n");
        return 1;
    }

    /* ---- claim 2: the loss falls ---- */
    loss1 = sk_train_sequence(&tr, toks, ntok, af);
    printf("  loss after  %.4f\n", (double)loss1 / 65536.0);
    if (loss1 >= loss0) {
        printf("FAILED: the loss did not fall (%.4f -> %.4f)\n",
               (double)loss0 / 65536.0, (double)loss1 / 65536.0);
        return 1;
    }

    /* Ten steps, to show it keeps falling rather than getting lucky once. */
    {
        int step;
        sk_fix prev = loss1;

        printf("  ten more steps:");
        for (step = 0; step < 10; step++) {
            sk_train_zero_grads(&tr);
            sk_train_sequence(&tr, toks, ntok, af);
            sk_train_grads(&tr, toks, ntok, af);
            sk_train_update(&tr, lr, trust);
            prev = sk_train_sequence(&tr, toks, ntok, af);
            printf(" %.4f", (double)prev / 65536.0);
        }
        printf("\n");
        if (prev >= loss1) {
            printf("FAILED: ten steps did not improve on one (%.4f -> %.4f)\n",
                   (double)loss1 / 65536.0, (double)prev / 65536.0);
            return 1;
        }
    }

    /* ---- claim 3a: a trained row still uses the whole code range ----
     *
     * sk_shadow_row_store derives the scale as max|master|/127, so after a
     * store the largest code in the row is 127 - unless the row was clamped,
     * which happens only when max|master| < 127 (a row of near-zero weights,
     * where the scale would round to 0 and is raised to 1 instead).
     *
     * Why it matters: a row that stores codes of magnitude 3 while claiming
     * the range of 127 has thrown away four bits. That is what a mis-derived
     * scale looks like, and it is invisible in the loss for hundreds of steps
     * because the row simply contributes less.
     */
    {
        signed char code[SK_T_MAXHID];
        sk_fix scale;
        sk_u32 r, j;
        int last = m.L - 1;
        long clamped = 0, offrange = 0;

        for (r = 0; r < m.f1[last].rows; r++) {
            int maxcode = 0;

            if (!sk_table_scale_read(&m.f1[last], r, &scale)
                || !sk_table_read_row(&m.f1[last], r, code)) {
                printf("FAILED: cannot read f1 row %lu\n",
                       (unsigned long)r);
                return 1;
            }
            for (j = 0; j < m.f1[last].cols; j++) {
                int a = (code[j] < 0) ? -code[j] : code[j];

                if (a > maxcode)
                    maxcode = a;
            }
            if (scale <= 1)
                clamped++;              /* legitimate: near-zero row */
            else if (maxcode != 127)
                offrange++;             /* scale and codes disagree */
        }
        printf("  f1 rows: %ld clamped to scale 1, %ld out of range\n",
               clamped, offrange);
        if (offrange > 0) {
            printf("FAILED: %ld rows store a scale that does not match their"
                   " codes\n", offrange);
            return 1;
        }
    }

    /* ---- claim 3: still a well-formed model ---- */
    crc1 = sk_model_weights_crc(&m);
    printf("  weights CRC after  %08lX%s\n", (unsigned long)crc1,
           (crc0 == crc1) ? "  (unchanged - suspicious)" : "");
    if (crc0 == crc1) {
        printf("FAILED: the CRC did not move, so the tables did not change\n");
        return 1;
    }

    if (!tables_consistent(&m, mrow, &bad)) {
        printf("FAILED: %ld values do not survive a round trip after"
               " training\n", bad);
        return 1;
    }
    printf("  every table still round-trips exactly after 11 updates\n");
    (void)moved2;

    sk_train_close(&tr);
    sk_arena_done(&arena);
    printf("\nok: weights move, loss falls, model still well-formed\n");
    return 0;
}
