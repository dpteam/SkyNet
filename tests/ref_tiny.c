/* ref_tiny.c - the ORIGINAL Tiny-2.1 forward pass, built for the host.
 *
 * This exists to be a ground truth. Validating a fixed-point rewrite
 * against my own float rewrite would only prove that I made the same
 * mistake twice. The functions below are the released implementation's
 * algorithms (matvec, norm, safe_exp, store, step), lifted verbatim from
 * refs/Tiny-2.1-main/dos/TINY.C with the DOS UI removed.
 *
 * It prints the top-N letter scores for a fixed prompt, which the
 * fixed-point implementation in src/sk_forward.c must reproduce.
 *
 * Build: cc -O2 -o build/ref_tiny tests/ref_tiny.c -lm
 * Run:   build/ref_tiny refs/Tiny-2.1-main/dos/TINY.BIN "hello"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VOCAB  97
#define CTX    128
#define MAXD   128
#define MAXHID 512
#define MAXL   8

typedef struct {
    float *scale;
    signed char *w;
    int rows, cols;
} Table;

static int D, L, H, HD, HID;
static float att_div;
static Table emb, pos;
static Table wq[MAXL], wk[MAXL], wv[MAXL], wo[MAXL], f1[MAXL], f2[MAXL];
static float *g1[MAXL], *g2[MAXL], *gf;

static signed char *kc[MAXL], *vc[MAXL];
static float *ks[MAXL], *vs[MAXL];

static float bx[MAXD], bh[MAXD], bq[MAXD], bt[MAXD], ba[MAXD], bm[MAXHID], bs[CTX];

static void *xmalloc(unsigned n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(2); }
    return p;
}

static int read16(FILE *fp)
{
    int lo = fgetc(fp), hi = fgetc(fp);
    if (lo == EOF || hi == EOF) { fprintf(stderr, "short file\n"); exit(2); }
    return lo | (hi << 8);
}

static void load_table(FILE *fp, Table *t, int rows, int cols)
{
    int r;
    t->rows = rows; t->cols = cols;
    t->scale = (float *)xmalloc((unsigned)rows * sizeof(float));
    t->w = (signed char *)xmalloc((unsigned)rows * (unsigned)cols);
    for (r = 0; r < rows; r++) {
        if (fread(&t->scale[r], 4, 1, fp) != 1
            || fread(t->w + (unsigned)r * cols, 1, (unsigned)cols, fp) != (unsigned)cols) {
            fprintf(stderr, "short table\n"); exit(2);
        }
    }
}

static float *load_gains(FILE *fp)
{
    float *g = (float *)xmalloc((unsigned)D * sizeof(float));
    if (fread(g, 4, (unsigned)D, fp) != (unsigned)D) {
        fprintf(stderr, "short gains\n"); exit(2);
    }
    return g;
}

static void load_model(const char *name)
{
    FILE *fp = fopen(name, "rb");
    char magic[4];
    int l;

    if (!fp) { fprintf(stderr, "cannot open %s\n", name); exit(2); }
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "TNY2", 4) != 0) {
        fprintf(stderr, "not TNY2\n"); exit(2);
    }
    D = read16(fp); L = read16(fp); H = read16(fp); HID = read16(fp);
    HD = D / H;
    att_div = (float)sqrt((double)HD);

    load_table(fp, &emb, VOCAB, D);
    load_table(fp, &pos, CTX, D);
    for (l = 0; l < L; l++) {
        g1[l] = load_gains(fp);
        load_table(fp, &wq[l], D, D);
        load_table(fp, &wk[l], D, D);
        load_table(fp, &wv[l], D, D);
        load_table(fp, &wo[l], D, D);
        g2[l] = load_gains(fp);
        load_table(fp, &f1[l], HID, D);
        load_table(fp, &f2[l], D, HID);
    }
    gf = load_gains(fp);
    fclose(fp);

    for (l = 0; l < L; l++) {
        kc[l] = (signed char *)xmalloc((unsigned)CTX * D);
        vc[l] = (signed char *)xmalloc((unsigned)CTX * D);
        ks[l] = (float *)xmalloc(CTX * sizeof(float));
        vs[l] = (float *)xmalloc(CTX * sizeof(float));
    }
}

static void matvec(const Table *t, const float *x, float *y)
{
    int r, c, cols = t->cols;
    const signed char *w = t->w;
    float s;
    for (r = 0; r < t->rows; r++) {
        s = 0.0f;
        for (c = 0; c < cols; c++)
            s += w[c] * x[c];
        y[r] = s * t->scale[r];
        w += cols;
    }
}

static void norm(const float *x, const float *g, float *y)
{
    int i;
    float s = 0.0f;
    for (i = 0; i < D; i++)
        s += x[i] * x[i];
    s = (float)(1.0 / sqrt(s / D + 1e-5));
    for (i = 0; i < D; i++)
        y[i] = x[i] * s * g[i];
}

static float safe_exp(float x)
{
    if (x < -60.0f)
        return 0.0f;
    return (float)exp(x);
}

static float store(signed char *dst, const float *src)
{
    int i;
    float big = 0.0f, v, k;
    for (i = 0; i < D; i++) {
        v = src[i] < 0 ? -src[i] : src[i];
        if (v > big) big = v;
    }
    if (big < 1e-20f)
        big = 1e-20f;
    k = 127.0f / big;
    for (i = 0; i < D; i++) {
        v = src[i] * k;
        dst[i] = (signed char)(v < 0 ? v - 0.5f : v + 0.5f);
    }
    return big / 127.0f;
}

static void step(int tok, int p, float *score)
{
    int i, j, l, h;
    const signed char *e = emb.w + tok * D;
    const signed char *q = pos.w + p * D;
    const signed char *kp, *vp;
    float s, big, sum, es = emb.scale[tok], ps = pos.scale[p];

    for (i = 0; i < D; i++)
        bx[i] = e[i] * es + q[i] * ps;

    for (l = 0; l < L; l++) {
        norm(bx, g1[l], bh);
        matvec(&wk[l], bh, bt);
        ks[l][p] = store(kc[l] + p * D, bt);
        matvec(&wv[l], bh, bt);
        vs[l][p] = store(vc[l] + p * D, bt);
        matvec(&wq[l], bh, bq);
        for (h = 0; h < H; h++) {
            big = -1e30f;
            for (j = 0; j <= p; j++) {
                kp = kc[l] + j * D + h * HD;
                s = 0.0f;
                for (i = 0; i < HD; i++)
                    s += bq[h * HD + i] * kp[i];
                s = s * ks[l][j] / att_div;
                bs[j] = s;
                if (s > big) big = s;
            }
            sum = 0.0f;
            for (j = 0; j <= p; j++) {
                bs[j] = safe_exp(bs[j] - big);
                sum += bs[j];
            }
            for (i = 0; i < HD; i++)
                ba[h * HD + i] = 0.0f;
            for (j = 0; j <= p; j++) {
                if (bs[j] == 0.0f)
                    continue;
                vp = vc[l] + j * D + h * HD;
                s = bs[j] * vs[l][j] / sum;
                for (i = 0; i < HD; i++)
                    ba[h * HD + i] += s * vp[i];
            }
        }
        matvec(&wo[l], ba, bt);
        for (i = 0; i < D; i++)
            bx[i] += bt[i];

        norm(bx, g2[l], bh);
        matvec(&f1[l], bh, bm);
        s = 0.0f;
        for (i = 0; i < HID; i++) {
            if (bm[i] < 0.0f) bm[i] = 0.0f;
            else s += bm[i];
        }
        matvec(&f2[l], bm, bt);
        for (i = 0; i < D; i++)
            bx[i] += bt[i];
    }
    norm(bx, gf, bh);
    matvec(&emb, bh, score);
}

int main(int argc, char **argv)
{
    static float score[VOCAB];
    const char *model = argc > 1 ? argv[1] : "tests/fixtures/TINY.BIN";
    const char *prompt = argc > 2 ? argv[2] : "hello";
    const char *mode = argc > 3 ? argv[3] : "scores";
    int i, p, n;

    load_model(model);
    printf("D=%d L=%d H=%d HID=%d HD=%d\n", D, L, H, HID, HD);

    n = (int)strlen(prompt);
    for (p = 0; p < n; p++) {
        int tok = (unsigned char)prompt[p] - ' ' + 2;
        if (tok < 2) tok = 2;
        if (tok >= VOCAB) tok = VOCAB - 1;
        step(tok, p, score);
    }

    if (strcmp(mode, "generate") == 0) {
        /* Deterministic decoding: creativity 0 means "always the best
           letter", which is the only mode that can be compared between two
           implementations without sharing an RNG. The released TINY.C
           reaches the same state through pick() with creativity <= 0. */
        char out[CTX + 1];
        int letters = 0;

        step(1, p++, score);            /* end of the question */
        for (; p < CTX; p++) {
            int best = 0, tok;
            for (i = 1; i < VOCAB; i++)
                if (score[i] > score[best]) best = i;
            tok = best;
            if (tok <= 1) break;
            out[letters++] = (char)(tok + 30);
            step(tok, p, score);
        }
        out[letters] = 0;
        printf("answer %s\n", out);
        return 0;
    }

    /* Print the whole score vector, in the model's own fixed order, so a
       comparison tool can diff it against the fixed-point build without
       needing to agree on formatting of the top-N list. */
    printf("scores");
    for (i = 0; i < VOCAB; i++)
        printf(" %.6f", score[i]);
    printf("\n");
    return 0;
}
