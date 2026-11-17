#ifndef TINYML_H
#define TINYML_H

#include <math.h>

#define MAX_I 256
#define MAX_H 32
#define MAX_O 256

static float w1[MAX_I][MAX_H];
static float w2[MAX_H][MAX_O];

static float sigmoid(float x){
    if(x < -10) return 0;
    if(x > 10) return 1;
    return 1.0f/(1.0f+exp(-x));
}

static void nn_forward(float *in,float *h,float *o){
    int i,j,k;

    for(j=0;j<MAX_H;j++){
        h[j]=0;
        for(i=0;i<MAX_I;i++)
            h[j]+=in[i]*w1[i][j];
        h[j]=sigmoid(h[j]);
    }

    for(k=0;k<MAX_O;k++){
        o[k]=0;
        for(j=0;j<MAX_H;j++)
            o[k]+=h[j]*w2[j][k];
        o[k]=sigmoid(o[k]);
    }
}

static void nn_train(float *in,float lr){
    float h[MAX_H],o[MAX_O];
    float err;
    int j,k;

    nn_forward(in,h,o);

    for(k=0;k<MAX_O;k++){
        err=in[k]-o[k];
        for(j=0;j<MAX_H;j++)
            w2[j][k]+=lr*err*h[j];
    }
}

#endif
