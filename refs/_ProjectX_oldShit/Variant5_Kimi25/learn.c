/* TinyAI-Learn - Self-contained, single-file neural network trainer
   Compile: tcc learn.c -o learn.exe  (or gcc -Os -s -std=c89)
   Supports: DOS(DJGPP), Win32, POSIX
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --------- Configuration --------- */
#define IN_SIZE   64
#define HID_SIZE  32
#define OUT_SIZE  64
#define LR        0.01f
#define EPOCHS    100
#define SKT_MAGIC "SKT"

/* --------- Platform-specific directory scanning --------- */
#ifdef __DJGPP__  /* DOS with DJGPP (POSIX-compliant) */
#include <dirent.h>
#include <sys/stat.h>
#define HAS_DIR 1
#elif defined(_WIN32)  /* Windows 9x/XP/Win32 */
#include <windows.h>
#define HAS_DIR 1
#else  /* Minimal POSIX or bare fallback */
#ifdef _POSIX_SOURCE
#include <dirent.h>
#include <sys/stat.h>
#define HAS_DIR 1
#endif
#endif

/* --------- Neural Network Structure --------- */
typedef struct {
    int in, hid, out;
    float *w1, *b1, *w2, *b2;
} Net;

/* --------- Math (no math.h) --------- */
float sigm(float x) {
    /* Fast approximation: x / (1 + |x|) mapped to 0..1 */
    if (x < -4.0f) return 0.0f;
    if (x > 4.0f) return 1.0f;
    return 0.5f + 0.5f * x / (1.0f + (x < 0 ? -x : x));
}
float dsigm(float y) { return y * (1.0f - y); }

/* --------- Network Management --------- */
Net* net_new(int in, int hid, int out) {
    Net *n = (Net*)malloc(sizeof(Net));
    n->in=in; n->hid=hid; n->out=out;
    n->w1=(float*)malloc(in*hid*sizeof(float));
    n->b1=(float*)malloc(hid*sizeof(float));
    n->w2=(float*)malloc(hid*out*sizeof(float));
    n->b2=(float*)malloc(out*sizeof(float));
    srand((unsigned)time(NULL));
#define RANDF() ((float)rand()/RAND_MAX - 0.5f)
    { int i; for(i=0;i<in*hid;i++) n->w1[i]=RANDF()*0.5f; }
    { int i; for(i=0;i<hid;i++) n->b1[i]=RANDF()*0.5f; }
    { int i; for(i=0;i<hid*out;i++) n->w2[i]=RANDF()*0.5f; }
    { int i; for(i=0;i<out;i++) n->b2[i]=RANDF()*0.5f; }
    return n;
}
void net_free(Net *n) {
    free(n->w1); free(n->b1); free(n->w2); free(n->b2); free(n);
}

/* --------- IO Utilities --------- */
void gen_skt_name(char *buf) {
    time_t t = time(NULL);
    unsigned long h = ((unsigned long)t) & 0xFFFFFFFF;
    sprintf(buf, "%08lX.SKT", h);
}
void save_skt(Net *n, const char *fn) {
    FILE *f = fopen(fn, "wb");
    char ver = 1;
    if(!f) return;
    fwrite(SKT_MAGIC, 3, 1, f);
    fwrite(&ver, 1, 1, f);
    fwrite(&n->in, 1, 1, f); /* max 255 */
    fwrite(&n->hid, 1, 1, f);
    fwrite(&n->out, 1, 1, f);
    fwrite(n->w1, sizeof(float), n->in*n->hid, f);
    fwrite(n->b1, sizeof(float), n->hid, f);
    fwrite(n->w2, sizeof(float), n->hid*n->out, f);
    fwrite(n->b2, sizeof(float), n->out, f);
    fclose(f);
}

/* --------- Forward & Train --------- */
void forward(Net *n, float *x, float *h, float *y) {
    int i,j;
    for(i=0;i<n->hid;i++) {
        float s = n->b1[i];
        for(j=0;j<n->in;j++) s += x[j] * n->w1[j*n->hid+i];
        h[i] = sigm(s);
    }
    for(i=0;i<n->out;i++) {
        float s = n->b2[i];
        for(j=0;j<n->hid;j++) s += h[j] * n->w2[j*n->out+i];
        y[i] = sigm(s);
    }
}
void train(Net *n, float *x, float *t) {
    float h[HID_SIZE], y[OUT_SIZE], dy[OUT_SIZE], dh[HID_SIZE];
    int i,j;
    forward(n,x,h,y);
    /* Output delta */
    for(i=0;i<n->out;i++) dy[i]=(t[i]-y[i])*dsigm(y[i]);
    /* Hidden delta */
    for(i=0;i<n->hid;i++) {
        float s=0;
        for(j=0;j<n->out;j++) s+=dy[j]*n->w2[i*n->out+j];
        dh[i]=s*dsigm(h[i]);
    }
    /* Update */
    for(i=0;i<n->hid;i++) for(j=0;j<n->out;j++) 
        n->w2[i*n->out+j] += LR * dy[j] * h[i];
    for(i=0;i<n->out;i++) n->b2[i] += LR * dy[i];
    for(i=0;i<n->in;i++) for(j=0;j<n->hid;j++)
        n->w1[i*n->hid+j] += LR * dh[j] * x[i];
    for(i=0;i<n->hid;i++) n->b1[i] += LR * dh[i];
}

/* --------- Training on File --------- */
void train_file(Net *n, const char *fn) {
    FILE *f = fopen(fn, "rb");
    unsigned char buf[IN_SIZE];
    if(!f) return;
    printf("Learning: %s\n", fn);
    while(fread(buf, 1, IN_SIZE, f) == IN_SIZE) {
        float x[IN_SIZE], t[OUT_SIZE];
        int i; for(i=0;i<IN_SIZE;i++) x[i] = buf[i]/255.0f;
        for(i=0;i<OUT_SIZE && i<IN_SIZE;i++) t[i]=x[i];
        train(n,x,t);
    }
    fclose(f);
}

/* --------- Directory Traversal --------- */
#ifdef HAS_DIR
void train_dir(Net *n, const char *path) {
#ifdef _WIN32
    WIN32_FIND_DATA fd;
    HANDLE h;
    char mask[256];
    sprintf(mask, "%s\\*", path);
    h = FindFirstFile(mask, &fd);
    if(h==INVALID_HANDLE_VALUE) return;
    do {
        char full[256];
        if(fd.cFileName[0]=='.') continue;
        sprintf(full, "%s\\%s", path, fd.cFileName);
        if(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            train_dir(n, full);
        } else {
            train_file(n, full);
        }
    } while(FindNextFile(h, &fd));
    FindClose(h);
#else /* DJGPP/POSIX */
    DIR *d = opendir(path);
    struct dirent *e;
    struct stat st;
    char full[512];
    if(!d) return;
    while((e=readdir(d))!=NULL) {
        if(e->d_name[0]=='.') continue;
        sprintf(full, "%s/%s", path, e->d_name);
        if(stat(full, &st)==0) {
            if(S_ISDIR(st.st_mode)) train_dir(n, full);
            else train_file(n, full);
        }
    }
    closedir(d);
#endif
}
#else
void train_dir(Net *n, const char *p) { 
    printf("Directory scan not supported. Use single file.\n"); 
    (void)n; (void)p; 
}
#endif

/* --------- Main --------- */
int main(int argc, char **argv) {
    Net *n = net_new(IN_SIZE, HID_SIZE, OUT_SIZE);
    char outname[13];
    
    printf("TinyAI-Learn 1.0 [%d-%d-%d]\n", IN_SIZE, HID_SIZE, OUT_SIZE);
    
    if(argc>1) {
        printf("Scanning: %s\n", argv[1]);
        train_dir(n, argv[1]);
    } else {
        printf("Usage: %s <directory>\nDemo training on random patterns...\n", argv[0]);
        int e,i; 
        for(e=0;e<100;e++) {
            float x[IN_SIZE], t[OUT_SIZE];
            for(i=0;i<IN_SIZE;i++) { x[i]=(float)rand()/RAND_MAX; t[i]=x[i]; }
            train(n,x,t);
        }
    }
    
    gen_skt_name(outname);
    save_skt(n, outname);
    printf("Saved: %s\n", outname);
    
    net_free(n);
    return 0;
}