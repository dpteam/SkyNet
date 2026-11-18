/* gradcheck.c - numerical verification of the trainer's gradient.
 *
 * This lives in tests/ and uses floating point on purpose. The core has no
 * float (invariant I1) because the model must produce identical results on
 * machines without an FPU; a diagnostic that compares two numbers does not
 * run on the target and gains nothing from being fixed point.
 *
 * What it does: nudge one parameter by a known amount, measure how much the
 * loss moves, and compare with what the backward pass claims. A sign error in
 * a hand-derived gradient looks exactly like a model that will not learn, and
 * no amount of staring at the algebra catches it reliably.
 *
 * Two things about this measurement are worth knowing before reading a
 * result, because both of them once looked like "the gradient is wrong":
 *
 * 1. The step cannot be a fixed fraction of the parameter. The loss is
 *    Q16.16, so it is quantised at 1.53e-5, and the RMS norms inside the
 *    forward pass are themselves quantised: the loss is a staircase in every
 *    parameter, with treads of roughly 1e-4. A step that moves the loss by
 *    only a few treads measures the staircase. A step that moves it by a lot
 *    leaves the region where the derivative exists - 25% of an output row
 *    scale is 25% of a logit, and the softmax stops being a local expansion.
 *    So the step is chosen per parameter to move the loss by TARGET_LSB
 *    quanta, and a parameter that would need more than CAP_FRAC of its own
 *    value to do that cannot be measured at all in this arithmetic. Those
 *    rows are reported as unresolved, not as failures.
 *
 * 2. "Numeric differs from analytic" does not mean "the gradient is wrong".
 *    A ReLU switches on over the step and the loss then has a corner there,
 *    where a central difference is not the derivative. One-sided slopes on
 *    either side of the point separate the two cases: they agree for a
 *    smooth function and split at a corner. Rows that split are reported as
 *    nonlinear and are not judged either.
 *
 * Build: cc -O2 -Isrc -o build/gradcheck tests/gradcheck.c src/*.c -lm
 * Run:   build/gradcheck refs/Tiny-2.1-main/dos/TINY.BIN build/dataset.txt
 *
 * Optional arguments, for narrowing a measurement down:
 *   3  TARGET_LSB, quanta of loss the step should move (default 600)
 *   4  verbose, 1 to print every row that does not agree
 *   5  CAP_FRAC, largest step as a fraction of the parameter (default 0.25)
 *   6  number of dataset lines to average the loss over (default 1)
 *   7  SK_P_* kind to restrict to, -1 for all
 *   8  row within that kind to restrict to, -1 for all
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

#define MAXTOK   64
#define MAXLINES 32

typedef struct {
    int toks[MAXTOK];
    int ntok;
    int af;
    int npos;
} sample;

static int tokenize(const char *line, sample *s)
{
    char q[128], a[128];
    int qlen, alen, k, sep, j, ntok;

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

    ntok = 0;
    for (k = 0; k < qlen; k++) {
        int t = (unsigned char)q[k] - 30;
        s->toks[ntok++] = (t < 2) ? 2 : ((t >= SK_T_VOCAB) ? SK_T_VOCAB - 1 : t);
    }
    s->toks[ntok++] = 1;
    for (k = 0; k < alen; k++) {
        int t = (unsigned char)a[k] - 30;
        s->toks[ntok++] = (t < 2) ? 2 : ((t >= SK_T_VOCAB) ? SK_T_VOCAB - 1 : t);
    }
    s->toks[ntok++] = 1;

    s->ntok = ntok;
    s->af = qlen + 1;
    s->npos = ntok - 1 - s->af;
    return (s->npos > 0) ? ntok : 0;
}

/* Position-weighted mean loss over the samples.
 *
 * sk_train_sequence returns the mean over its own positions, so a plain
 * average of those means would give a long line less weight than its length
 * deserves, and the analytic side - which accumulates raw and divides by the
 * total position count - would not be the derivative of what is measured.
 * Weighting by npos makes both sides the mean over all positions. */
static double loss_of(sk_train *tr, const sample *s, int n)
{
    double acc = 0.0;
    long w = 0;
    int i;

    for (i = 0; i < n; i++) {
        double l = (double)sk_train_sequence(tr, s[i].toks, s[i].ntok, s[i].af)
                   / 65536.0;

        acc += l * (double)s[i].npos;
        w += (long)s[i].npos;
    }
    return (w > 0) ? acc / (double)w : 0.0;
}

/* Identifies each trainable parameter and the fixed-point unit it is stored
 * in. Row scales are Q8.24, gains are Q16.16: using one divisor for both made
 * the gains' perturbation 256 times too small, which read as a gradient
 * error of exactly 256 and was a units error in the checker. */
static const struct {
    int kind;
    const char *name;
    double unit;
} kinds[] = {
    { SK_P_OUT, "output projection", 16777216.0 },
    { SK_P_F2,  "f2 (last block)",   16777216.0 },
    { SK_P_F1,  "f1 (last block)",   16777216.0 },
    { SK_P_GF,  "final gains",          65536.0 },
    { SK_P_G2,  "g2 (last block)",      65536.0 }
};
#define NKINDS ((int)(sizeof(kinds) / sizeof(kinds[0])))

int main(int argc, char **argv)
{
    const char *model = argc > 1 ? argv[1] : "tests/fixtures/TINY.BIN";
    const char *data = argc > 2 ? argv[2] : "build/dataset.txt";
    sk_model m;
    sk_train tr;
    sk_arena arena;
    void *weights, *tstore;
    sk_u32 need, tneed;
    char why[64];
    unsigned char hdr[12];
    FILE *fp;
    sample samp[MAXLINES];
    int nsamp = 0, nlines, i, k, r;
    int rows_checked = 0, rows_ok = 0;
    int out_checked = 0, out_ok = 0;
    int unresolved = 0, nonlinear = 0;
    int empty_kinds = 0;
    double worst_rel = 0.0;
    double target_lsb = (argc > 3) ? atof(argv[3]) : 600.0;
    int verbose = (argc > 4) ? atoi(argv[4]) : 0;
    double cap_frac = (argc > 5) ? atof(argv[5]) : 0.25;
    int only_kind = (argc > 7) ? atoi(argv[7]) : -1;
    int only_row = (argc > 8) ? atoi(argv[8]) : -1;
    int dump = (argc > 9) ? atoi(argv[9]) : 0;
    static double ana[SK_T_MAXHID];
    static sk_fix s0[SK_T_MAXHID];

    nlines = (argc > 6) ? atoi(argv[6]) : 1;
    if (nlines < 1)
        nlines = 1;
    if (nlines > MAXLINES)
        nlines = MAXLINES;

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

    tneed = sk_train_bytes(&m, SK_TRAIN_CODES);
    tstore = sk_arena_alloc(&arena, tneed);
    if (!tstore || !sk_train_open(&tr, &m, tstore, tneed,
                             SK_TRAIN_CODES)) {
        printf("trainer open failed\n");
        return 2;
    }

    fp = fopen(data, "r");
    if (!fp) { printf("cannot open %s\n", data); return 2; }
    {
        char line[256];

        while (nsamp < nlines && fgets(line, (int)sizeof(line), fp)) {
            if (tokenize(line, &samp[nsamp]))
                nsamp++;
        }
    }
    fclose(fp);
    if (nsamp == 0) {
        printf("no usable line in %s\n", data);
        return 2;
    }

    printf("gradient check on %d line%s of %s\n", nsamp,
           (nsamp == 1) ? "" : "s", data);
    printf("step sized to move the loss by %.0f quanta of Q16.16, "
           "cap %.0f%% of the parameter, tolerance 25%%\n",
           target_lsb, cap_frac * 100.0);

    /* Analytic gradients, captured for every parameter BEFORE any
       perturbation: loss_of() re-runs the forward pass, so reading a
       gradient inside the loop compares one model state with a derivative
       measured at another. That mistake produced a 472% "mismatch" on a
       correct gradient. */
    /* One line at a time, sequence and then grads: sk_train_grads() reads
       the dlogits left behind by sk_train_sequence(), so two separate loops
       would give every line but the last the dlogits of the last one. That
       is what made a 20-line average look 300 times too small. */
    sk_train_zero_grads(&tr);
    for (i = 0; i < nsamp; i++) {
        sk_train_sequence(&tr, samp[i].toks, samp[i].ntok, samp[i].af);
        sk_train_grads(&tr, samp[i].toks, samp[i].ntok, samp[i].af);
    }
    printf("loss over these samples: %.6f\n", loss_of(&tr, samp, nsamp));

    for (k = 0; k < NKINDS; k++) {
        int n, kc = 0, kok = 0, kun = 0, knl = 0;
        double kworst = 0.0;

        if (only_kind >= 0 && kinds[k].kind != only_kind)
            continue;

        n = (kinds[k].kind == SK_P_OUT) ? SK_T_VOCAB
          : (kinds[k].kind == SK_P_F1)  ? m.HID : m.D;

        for (r = 0; r < n; r++) {
            ana[r] = (double)sk_train_grad_get(&tr, kinds[k].kind, r) / 65536.0;
            if (!sk_train_param_get(&tr, kinds[k].kind, r, &s0[r]))
                s0[r] = 0;
        }

        if (dump && kinds[k].kind == SK_P_F1) {
            int dr, dc;

            printf("ANALYTIC DW_F1\n");
            for (dr = 0; dr < m.HID; dr++)
                for (dc = 0; dc < m.D; dc++)
                    printf("%.12g\n", (double)sk_train_dw_get(&tr, SK_P_DW_F1,
                                                               dr * m.D + dc)
                           / 65536.0);
        }
        if (dump && kinds[k].kind == SK_P_F2) {
            int dr, dc;

            printf("ANALYTIC DW_F2\n");
            for (dr = 0; dr < m.D; dr++)
                for (dc = 0; dc < m.HID; dc++)
                    printf("%.12g\n", (double)sk_train_dw_get(&tr, SK_P_DW_F2,
                                                               dr * m.HID + dc)
                           / 65536.0);
        }

        if (dump) {
            printf("ANALYTIC %s\n", kinds[k].name);
            for (r = 0; r < n; r++)
                printf("%.12g\n", ana[r]);
            continue;               /* dump only, no finite differences */
        }

        for (r = 0; r < n; r++) {
            sk_fix sp, sm;
            double eps, lp, lm, l0, numeric, rel, denom, val, need_eps, curv;

            if (only_row >= 0 && r != only_row)
                continue;
            if (s0[r] <= 0)
                continue;

            /* Step for TARGET_LSB quanta of loss movement. */
            val = (double)s0[r] / kinds[k].unit;
            need_eps = target_lsb / 65536.0 / (2.0 * fabs(ana[r]));
            if (need_eps > cap_frac * val) {
                kun++;
                unresolved++;
                continue;
            }
            if (need_eps < 0.002 * val)
                need_eps = 0.002 * val;      /* still a finite difference */
            eps = need_eps;
            if (eps < 1e-7)
                continue;

            sp = (sk_fix)(s0[r] + (sk_i32)(eps * kinds[k].unit));
            sm = (sk_fix)(s0[r] - (sk_i32)(eps * kinds[k].unit));

            sk_train_param_set(&tr, kinds[k].kind, r, sp);
            lp = loss_of(&tr, samp, nsamp);
            sk_train_param_set(&tr, kinds[k].kind, r, sm);
            lm = loss_of(&tr, samp, nsamp);
            sk_train_param_set(&tr, kinds[k].kind, r, s0[r]);
            l0 = loss_of(&tr, samp, nsamp);

            numeric = (lp - lm) / (2.0 * eps);

            /* One-sided slopes: equal for a smooth function, split at a
               corner. See the note at the top of the file. */
            curv = fabs((lp - l0) - (l0 - lm)) / eps
                   / (2.0 * fabs(numeric) + 1e-12);
            if (curv > 0.5) {
                knl++;
                nonlinear++;
                if (verbose)
                    printf("  %-20s row %3d nonlinear: slopes %+10.6f and"
                           " %+10.6f, centre %+10.6f, analytic %+10.6f\n",
                           kinds[k].name, r, (lp - l0) / eps,
                           (l0 - lm) / eps, numeric, ana[r]);
                continue;
            }

            denom = fabs(ana[r]) > 1e-12 ? fabs(ana[r]) : 1e-12;
            rel = fabs(numeric - ana[r]) / denom;

            kc++;
            if (rel < 0.25)
                kok++;
            if (rel > kworst)
                kworst = rel;

            if (rel >= 0.25 && verbose)
                printf("  %-20s row %3d analytic %+12.6f numeric %+12.6f"
                       "  ratio %+8.4f  slopes %+10.6f %+10.6f"
                       "  rel %.1f%%\n",
                       kinds[k].name, r, ana[r], numeric,
                       (fabs(ana[r]) > 1e-12) ? numeric / ana[r] : 0.0,
                       (lp - l0) / eps, (l0 - lm) / eps, rel * 100.0);
            if (rel >= 0.25 && verbose) {
                printf("      s0 %ld sp %ld sm %ld  lp %.9f l0 %.9f lm %.9f"
                       "  eps %.3e\n", (long)s0[r], (long)sp, (long)sm,
                       lp, l0, lm, eps);
            }
        }

        if (kc == 0)
            empty_kinds++;
        printf("  %-20s %3d/%3d agree, %3d unresolved, %3d nonlinear,"
               " worst %5.1f%%%s\n",
               kinds[k].name, kok, kc, kun, knl, kworst * 100.0,
               (kinds[k].kind == SK_P_OUT) ? "   [in updater]" : "   [measured]");

        rows_checked += kc;
        rows_ok += kok;
        if (kworst > worst_rel)
            worst_rel = kworst;
        if (kinds[k].kind == SK_P_OUT) {
            out_ok = kok;
            out_checked = kc;
        }
    }

    printf("\n%d/%d measurable parameters agree within 25%%, worst %.1f%%\n",
           rows_ok, rows_checked, worst_rel * 100.0);
    if (nonlinear > 0)
        printf("%d parameters have a corner over the step; a finite difference"
               " has no derivative there\n", nonlinear);
    if (unresolved > 0)
        printf("%d parameters would need a step beyond the linear region to be"
               " measurable\n", unresolved);

    /* Every measured row has to agree, and every kind has to contribute at
       least one measured row: a regression that pushes everything into
       "unresolved" must fail rather than report success on nothing. */
    if (dump)
        return 0;                   /* values only, nothing judged */
    if (rows_checked == 0 || rows_ok != rows_checked || empty_kinds > 0) {
        printf("FAILED\n");
        return 1;
    }
    printf("ok (output projection %d/%d)\n", out_ok, out_checked);
    return 0;
}
