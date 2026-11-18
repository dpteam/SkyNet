/* gradcheck.c - numerical verification of the trainer's gradient.
 *
 * This lives in tests/ and uses floating point on purpose. The core has no
 * float (invariant I1) because the model must produce identical results on
 * machines without an FPU; a diagnostic that compares two numbers does not
 * run on the target and gains nothing from being fixed point.
 *
 * What it does: nudge one output scale by a known amount, measure how much
 * the loss moves, and compare with what the backward pass claims. A sign
 * error in a hand-derived gradient looks exactly like a model that will not
 * learn, and no amount of staring at the algebra catches it reliably.
 *
 * Build: cc -O2 -Isrc -o build/gradcheck tests/gradcheck.c src/*.c -lm
 * Run:   build/gradcheck refs/Tiny-2.1-main/dos/TINY.BIN build/dataset.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sk_types.h"
#include "sk_math.h"
#include "sk_tensor.h"
#include "sk_forward.h"
#include "sk_train.h"
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

/* Loss for one sequence, with no gradient accumulation. */
static double loss_of(sk_train *tr, const int *toks, int ntok, int af)
{
    sk_fix l = sk_train_sequence(tr, toks, ntok, af);

    return (double)l / 65536.0;
}

int main(int argc, char **argv)
{
    const char *model = argc > 1 ? argv[1] : "refs/Tiny-2.1-main/dos/TINY.BIN";
    const char *data = argc > 2 ? argv[2] : "build/dataset.txt";
    sk_model m;
    sk_train tr;
    sk_arena arena;
    void *weights, *tstore;
    sk_u32 need, tneed;
    char why[64];
    unsigned char hdr[12];
    FILE *fp;
    int toks[MAXTOK], ntok, af;
    /* 5% of the scale by default. Measured: at 2% one row disagrees by
       50%, at 0.5% and 0.1% the loss difference falls below one LSB of
       Q16.16 and the numeric gradient comes out as exactly zero. The floor
       is the loss representation, not the gradient: ?L = grad * 2*eps*scale,
       and with grad around 0.1 and scale around 0.005 that is 1e-5 at eps
       1% - one LSB. Larger eps is the only way to see the derivative. */
    double eps_frac = (argc > 3) ? atof(argv[3]) : 0.05;
    int rows_checked = 0, rows_ok = 0;
    double worst_rel = 0.0;
    int worst_row = -1;

    sk_math_init();

    fp = fopen(model, "rb");
    if (!fp) { printf("cannot open %s\n", model); return 2; }
    if (fread(hdr, 1, 12, fp) != 12) { fclose(fp); return 2; }
    fclose(fp);

    need = sk_model_size((sk_u32)(hdr[4] | (hdr[5] << 8)),
                         (sk_u32)(hdr[6] | (hdr[7] << 8)),
                         (sk_u32)(hdr[10] | (hdr[11] << 8)),
                         97UL, 128UL);
    if (!sk_arena_init(&arena, need + (24UL * 1024UL * 1024UL), SK_MEM_OS)) {
        printf("arena failed\n");
        return 2;
    }
    weights = sk_arena_alloc(&arena, need);
    if (!weights || !sk_pool_open_from(&m.pool, need, weights)) return 2;
    if (!sk_model_load(&m, model, &m.pool, why, (sk_u32)sizeof(why))) {
        printf("load failed: %s\n", why);
        return 2;
    }

    tneed = sk_train_bytes(&m);
    tstore = sk_arena_alloc(&arena, tneed);
    if (!tstore || !sk_train_open(&tr, &m, tstore, tneed)) {
        printf("trainer open failed\n");
        return 2;
    }

    fp = fopen(data, "r");
    if (!fp) { printf("cannot open %s\n", data); return 2; }
    if (!fgets((char *)hdr, 12, fp)) { fclose(fp); return 2; }
    fclose(fp);

    /* reopen for a full line */
    {
        char line[256];
        FILE *df = fopen(data, "r");
        if (!df || !fgets(line, (int)sizeof(line), df)) {
            printf("empty dataset\n");
            return 2;
        }
        fclose(df);
        ntok = tokenize(line, toks, &af);
        if (ntok == 0) { printf("unparseable first line\n"); return 2; }
    }

    printf("gradient check on %d tokens, answer from %d\n", ntok, af);

    /* Analytic gradient for this sequence.  It must be captured for ALL
       rows BEFORE any perturbation: loss_of() runs the forward pass again,
       which recomputes dlogits, so reading the analytic value inside the
       perturbation loop compares a gradient from one model state against a
       derivative measured at another. That mistake produced a 472%
       "mismatch" on a gradient that was in fact correct. */
    sk_train_zero_grads(&tr);
    sk_train_sequence(&tr, toks, ntok, af);
    sk_train_grads(&tr, toks, ntok, af);

    {
        static double analytic[SK_T_VOCAB];
        static sk_fix scale0[SK_T_VOCAB];
        int r;

        for (r = 0; r < SK_T_VOCAB; r++) {
            analytic[r] = (double)sk_train_grad_row(&tr, r) / 65536.0;
            if (!sk_table_scale_read(&m.out, (sk_u32)r, &scale0[r]))
                scale0[r] = 0;
        }

        for (r = 0; r < SK_T_VOCAB; r++) {
            sk_fix sp, sm;
            double eps, lp, lm, numeric, rel, denom;

            if (scale0[r] <= 0)
                continue;

            eps = eps_frac * ((double)scale0[r] / 16777216.0);
            if (eps < 1e-7)
                continue;

            sp = (sk_fix)(scale0[r] + (sk_i32)(eps * 16777216.0));
            sm = (sk_fix)(scale0[r] - (sk_i32)(eps * 16777216.0));

            sk_table_scale_write(&m.out, (sk_u32)r, sp);
            lp = loss_of(&tr, toks, ntok, af);
            sk_table_scale_write(&m.out, (sk_u32)r, sm);
            lm = loss_of(&tr, toks, ntok, af);
            sk_table_scale_write(&m.out, (sk_u32)r, scale0[r]);

            numeric = (lp - lm) / (2.0 * eps);
            denom = fabs(analytic[r]) > 1e-9 ? fabs(analytic[r]) : 1e-9;
            rel = fabs(numeric - analytic[r]) / denom;

            rows_checked++;
            if (rel < 0.25)
                rows_ok++;
            if (rel > worst_rel) {
                worst_rel = rel;
                worst_row = r;
            }
            if (r < 6 || rel >= 0.25) {
                printf("  row %2d  scale %.6f  analytic %+12.6f  numeric %+12.6f  rel %.1f%%%s\n",
                       r, (double)scale0[r] / 16777216.0, analytic[r], numeric,
                       rel * 100.0, rel < 0.25 ? "" : "   <-- MISMATCH");
            }
        }
    }

    printf("\n%d/%d rows agree within 25%%, worst relative error %.1f%% at row %d\n",
           rows_ok, rows_checked, worst_rel * 100.0, worst_row);
    return (rows_ok == rows_checked && rows_checked > 0) ? 0 : 1;
}
