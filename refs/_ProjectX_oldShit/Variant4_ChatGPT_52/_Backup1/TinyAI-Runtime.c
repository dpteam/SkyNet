#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include "skt.h"
#include "tinyml.h"

#define LR 0.01f

static void load_model(const char *name){
    FILE *f;
    SKTHeader h;

    f=fopen(name,"rb");
    if(!f){ puts("Model not found"); exit(1); }

    fread(&h,sizeof(h),1,f);
    if(h.magic!=SKT_MAGIC){ puts("Bad model"); exit(1); }

    fread(w1,sizeof(w1),1,f);
    fread(w2,sizeof(w2),1,f);
    fclose(f);
}

static void process_file(const char *path){
    FILE *f;
    unsigned char buf[MAX_I];
    float in[MAX_I];
    int i,n;

    f=fopen(path,"rb");
    if(!f) return;

    n=fread(buf,1,MAX_I,f);
    fclose(f);

    for(i=0;i<MAX_I;i++)
        in[i]=(i<n)?buf[i]/255.0f:0;

    nn_train(in,LR);
}

static void scan_dir(const char *path){
    DIR *d;
    struct dirent *e;
    char full[260];

    d=opendir(path);
    if(!d) return;

    while((e=readdir(d))){
        if(e->d_name[0]=='.') continue;
        sprintf(full,"%s/%s",path,e->d_name);
        process_file(full);
    }
    closedir(d);
}

int main(int argc,char**argv){
    if(argc<3){
        puts("TinyAI-Runtime <model.skt> <sandbox_dir>");
        return 1;
    }

    load_model(argv[1]);

    /* автономный цикл */
    for(;;){
        scan_dir(argv[2]);
        puts("Learning cycle done");
    }
}
