#include <stdio.h>
#include <stdlib.h>
#include "skt.h"
#include "tinyml.h"

#define LR 0.05f

static void learn_file(const char *name){
    FILE *f;
    unsigned char buf[MAX_I];
    float in[MAX_I];
    int i,n;

    f=fopen(name,"rb");
    if(!f) return;

    n=fread(buf,1,MAX_I,f);
    fclose(f);

    for(i=0;i<MAX_I;i++)
        in[i]=(i<n)?buf[i]/255.0f:0;

    nn_train(in,LR);
}

int main(int argc,char**argv){
    FILE *f;
    SKTHeader h;
    int i,j;

    if(argc<2){
        puts("TinyAI-Learn <file>");
        return 1;
    }

    /* init weights */
    for(i=0;i<MAX_I;i++)
        for(j=0;j<MAX_H;j++)
            w1[i][j]=(float)(i*j%7)/50.0f;

    for(i=0;i<MAX_H;i++)
        for(j=0;j<MAX_O;j++)
            w2[i][j]=(float)(i+j%5)/50.0f;

    learn_file(argv[1]);

    h.magic=SKT_MAGIC;
    h.inputs=MAX_I;
    h.hidden=MAX_H;
    h.outputs=MAX_O;
    h.samples=1;

    f=fopen("A1B2C3D4.SKT","wb");
    fwrite(&h,sizeof(h),1,f);
    fwrite(w1,sizeof(w1),1,f);
    fwrite(w2,sizeof(w2),1,f);
    fclose(f);

    puts("Model saved: A1B2C3D4.SKT");
    return 0;
}
