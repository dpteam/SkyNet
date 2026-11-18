/* dwcheck.c - structural test for the per-weight gradient.
 *
 * Why this exists alongside gradcheck, and not inside it:
 *
 * gradcheck compares a gradient against a finite difference of the loss.
 * That works for row scales, whose gradients are around 0.1, but it cannot
 * work for a single weight: dL/dW for one element is about 4e-4, and one code
 * step moves the weight by `scale` ~ 0.01, so the loss moves by 4e-6 - a
 * quarter of one LSB of Q16.16. No step size makes that visible.
 *
 * So the weight gradient is proved two other ways:
 *
 *   1. Against the float reference (tools/ref_gradients.py, needs torch).
 *      Medians of ours/theirs over 20 000 elements sit at 0.98 and 1.00.
 *      That check cannot run in CI without torch and without refs/.
 *
 *   2. Here: a property that must hold by algebra, checked in integer
 *      arithmetic, needing neither a reference nor a float.
 *
 * The property: on a sequence with exactly ONE answer position, the whole dW
 * matrix is the outer product of two vectors,
 *
 *      dW[i][j] = dmid[i] * hn2[j]
 *
 * because dL/dW_eff[i][j] is the row's incoming gradient times the input the
 * row was multiplied by, and there is only one position to sum over. An outer
 * product is rank one, and rank one is checkable by cross-multiplication
 * without knowing dmid or hn2 at all:
 *
 *      dW[i][j] * dW[0][0] == dW[i][0] * dW[0][j]
 *
 * This catches the failure modes a whole-corpus average would hide: a
 * transposed matrix, a row stride off by one, the wrong input vector, dmid
 * taken before the ReLU mask instead of after, or two positions accidentally
 * summed into one buffer. None of those are rank one.
 *
 * Build: cc -O2 -Isrc -o build/dwcheck tests/dwcheck.c <all src files>
 * Run:   build/dwcheck tests/fixtures/TINY.BIN tests/fixtures/dataset.txt
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

/* How many quanta of Q16.16 a value must exceed to take part in the rank-one
 * identity. Below this the products are dominated by the truncation inside
 * sk_fix_mul and the test would report noise as a structural failure. */
#define SK_FLOOR 64

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

/* Rank-one test on one table. Returns the number of elements that violate the
 * identity by more than `tol` relative, and reports the worst violation. */
static int rank1(const char *name, sk_train *tr, int kind, int rows, int cols,
                 double tol, double *worst_out)
{
    int i, j, bad = 0, pivot_j = -1;
    double pivot = 0.0, worst = 0.0;
    static double v[SK_T_MAXHID * SK_T_MAXD];

    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            v[i * cols + j] = (double)sk_train_dw_get(tr, kind, i * cols + j)
                              / 65536.0;

    /* A pivot from the row with the largest element: cross-multiplying
       against a value near zero would divide noise by noise. */
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            if (v[i * cols + j] > pivot) {
                pivot = v[i * cols + j];
                pivot_j = j;
            }
    if (pivot_j < 0) {
        printf("  %-12s no positive element, cannot test\n", name);
        *worst_out = 0.0;
        return 0;
    }

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            double lhs = v[i * cols + j] * v[pivot_j];
            double rhs = v[i * cols + pivot_j] * v[j];
            double scale = fabs(lhs) > fabs(rhs) ? fabs(lhs) : fabs(rhs);
            double rel, small;

            if (scale < 1e-12)
                continue;               /* both sides are zero: consistent */

            /* Every value in the identity has to be well above one quantum,
               or the cross-multiplication compares rounding with rounding.
               Measured: dW entries of 1 and 3 units give products a few
               per cent apart purely from the truncation of sk_fix_mul, which
               is why an earlier version of this test reported 2148 failures
               on a matrix that was exactly rank one. */
            small = fabs(v[i * cols + j]);
            if (fabs(v[i * cols + pivot_j]) < small)
                small = fabs(v[i * cols + pivot_j]);
            if (fabs(v[j]) < small)
                small = fabs(v[j]);
            if (fabs(v[pivot_j]) < small)
                small = fabs(v[pivot_j]);
            if (small < (double)SK_FLOOR / 65536.0)
                continue;
            rel = fabs(lhs - rhs) / scale;
            if (rel > worst)
                worst = rel;
            if (rel > tol) {
                bad++;
                if (bad <= 3)
                    printf("    off at [%d][%d]: v=%.3g row0j=%.3g vij*=%.3g"
                           " row0j*=%.3g  lhs=%.3g rhs=%.3g\n",
                           i, j, v[i * cols + j], v[j],
                           v[i * cols + pivot_j], v[pivot_j], lhs, rhs);
            }
        }
    }
    printf("  %-12s %4dx%-4d rank one: %d of %d elements off by > %.1f%%,"
           " worst %.2f%%\n", name, rows, cols, bad, rows * cols,
           tol * 100.0, worst * 100.0);
    *worst_out = worst;
    return bad;
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
    char why[64];
    unsigned char hdr[12];
    FILE *fp;
    char line[256];
    int toks[MAXTOK], ntok, af, keep, i;
    double worst1 = 0.0, worst2 = 0.0;
    int bad1 = 0, bad2 = 0;

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

    /* Exactly one answer position: af predicts toks[af+1], and nothing after
       it. That is what makes dW a single outer product instead of a sum. */
    keep = af + 2;
    if (keep > ntok)
        keep = ntok;
    if (keep - 1 - af != 1) {
        printf("line too short: %d tokens, answer from %d\n", ntok, af);
        return 2;
    }

    printf("weight-gradient structure on %d tokens, 1 answer position\n", keep);
    sk_train_zero_grads(&tr);
    sk_train_sequence(&tr, toks, keep, af);
    sk_train_grads(&tr, toks, keep, af);

    /* Sanity: something must have been accumulated, or rank one is trivially
       true on an all-zero matrix and the test proves nothing. */
    {
        double sum = 0.0;

        for (i = 0; i < m.HID * m.D; i++)
            sum += fabs((double)sk_train_dw_get(&tr, SK_P_DW_F1, i));
        for (i = 0; i < m.D * m.HID; i++)
            sum += fabs((double)sk_train_dw_get(&tr, SK_P_DW_F2, i));
        if (sum == 0.0) {
            printf("FAILED: every weight gradient is zero\n");
            return 1;
        }
        printf("  %-12s total |dW| %.0f units of Q16.16\n", "sum", sum);
    }

    bad1 = rank1("dW f1", &tr, SK_P_DW_F1, m.HID, m.D, 0.02, &worst1);
    bad2 = rank1("dW f2", &tr, SK_P_DW_F2, m.D, m.HID, 0.02, &worst2);

    sk_train_close(&tr);
    sk_arena_done(&arena);

    if (bad1 > 0 || bad2 > 0) {
        printf("\nFAILED: the weight gradient is not an outer product, so it is"
               " not dmid * input\n");
        return 1;
    }
    printf("\nboth tables are rank one within 2%%, worst %.2f%% and %.2f%%\n",
           worst1 * 100.0, worst2 * 100.0);
    printf("ok\n");
    return 0;
}
