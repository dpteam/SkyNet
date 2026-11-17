#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "skt.h"
#include "tinyml.h"

#define LR 0.01f

/* ---------------- model ---------------- */

static void load_model(const char *name){
    FILE *f;
    SKTHeader h;

    f = fopen(name,"rb");
    if(!f){ puts("Model not found"); exit(1); }

    fread(&h,sizeof(h),1,f);
    if(h.magic != SKT_MAGIC){
        puts("Bad SKT file");
        exit(1);
    }

    fread(w1,sizeof(w1),1,f);
    fread(w2,sizeof(w2),1,f);
    fclose(f);
}

/* ---------------- learning ---------------- */

static void learn_file(const char *path){
    FILE *f;
    unsigned char buf[MAX_I];
    float in[MAX_I];
    int i,n;

    f = fopen(path,"rb");
    if(!f) return;

    n = fread(buf,1,MAX_I,f);
    fclose(f);

    for(i=0;i<MAX_I;i++)
        in[i] = (i<n) ? buf[i]/255.0f : 0.0f;

    nn_train(in, LR);
}

/* ---------------- directory scan ---------------- */

#ifdef _WIN32   /* Windows + TinyC */

#include <io.h>

static void scan_dir(const char *path){
    struct _finddata_t fd;
    intptr_t h;
    char mask[260];
    char full[260];

    sprintf(mask, "%s\\*.*", path);

    h = _findfirst(mask, &fd);
    if(h == -1) return;

    do{
        if(fd.name[0] == '.') continue;

        sprintf(full, "%s\\%s", path, fd.name);
        learn_file(full);

    }while(_findnext(h, &fd) == 0);

    _findclose(h);
}

#else          /* DOS / DJGPP */

#include <dos.h>

static void scan_dir(const char *path){
    struct find_t f;
    char mask[260];
    char full[260];

    sprintf(mask, "%s\\*.*", path);

    if(_dos_findfirst(mask, _A_NORMAL, &f) != 0)
        return;

    do{
        if(f.name[0] == '.') continue;

        sprintf(full, "%s\\%s", path, f.name);
        learn_file(full);

    }while(_dos_findnext(&f) == 0);
}

#endif

/* ---------------- agent loop ---------------- */

int main(int argc,char**argv){
    if(argc < 3){
        puts("TinyAI-Runtime <model.skt> <sandbox_dir>");
        return 1;
    }

    load_model(argv[1]);

    puts("TinyAI Runtime started");

    for(;;){
        scan_dir(argv[2]);
        puts("Cycle done");
    }
}
