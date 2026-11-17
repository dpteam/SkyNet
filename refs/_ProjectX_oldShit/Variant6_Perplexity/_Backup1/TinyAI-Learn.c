#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef signed short w_t; // 16-bit weights для компактности
#define L1 2 // input+1 bias
#define H1 3 // hidden+1
#define O1 1 // output+1

w_t w1[L1*H1], w2[H1*O1]; // ~100 байт
w_t b1[H1], b2[O1];

unsigned char hex[16]="0123456789ABCDEF";
w_t itof(float f){return (w_t)(f*2048);} // scale to int16
float foti(w_t i){return i/2048.0;}
float sig(float x){return x/(1+abs(x));} // fast sigmoid[web:24]

void matmul(const w_t* w, const float* in, float* out, int rows, int cols){
    for(int i=0;i<rows;i++){
        float sum=0;
        for(int j=0;j<cols;j++) sum += foti(w[i*cols+j]) * in[j];
        out[i] = sig(sum);
    }
}

void backprop(float* in, float target, float lr){
    float out[H1+1], delta_o[O1], delta_h[H1];
    float hout[O1]; matmul(w2, out, hout, O1, H1+1); // forward hidden->out
    float oout = hout[0]; delta_o[0] = (target - oout) * oout * (1-oout);
    
    for(int i=0;i<H1;i++) delta_h[i] = delta_o[0] * foti(w2[i*O1]) * out[i]*(1-out[i]);
    
    // update w2, b2
    for(int i=0;i<H1;i++) w2[i*O1] += itof(lr * delta_o[0] * out[i]);
    b2[0] += itof(lr * delta_o[0]);
    
    // hidden forward
    float hsum[H1]; matmul(w1, in, hsum, H1, L1);
    for(int i=0;i<H1;i++) out[i] = sig(hsum[i]);
    out[H1] = 1.0; // bias
    
    // update w1, b1
    for(int i=0;i<H1;i++) for(int j=0;j<L1;j++) w1[i*L1+j] += itof(lr * delta_h[i] * in[j]);
    for(int i=0;i<H1;i++) b1[i] += itof(lr * delta_h[i]);
}

int learn_dir(const char* path){ // dummy data from dir: file sizes mod 2 as XOR
    FILE* f = fopen(path,"rb"); if(!f) return 0;
    fseek(f,0,SEEK_END); long sz=ftell(f); fclose(f);
    static float data[4][L1] = {{0,0},{0,1},{1,0},{1,1}}; // XOR
    float targ[4] = {0,1,1,0};
    float* in = data[sz%4]; backprop(in, targ[sz%4], 0.1);
    return 1;
}

int recurse_dir(char* cwd, char* path){ // DOS/*nix compatible recursive[web:12]
    strcpy(cwd, path);
    FILE* p = popen("dir /b /s *.txt 2>nul || find . -name '*.txt' 2>/dev/null", "r"); // hybrid
    if(!p) return 0;
    char buf[256]; while(fgets(buf,sizeof(buf),p)) learn_dir(buf);
    pclose(p); return 1;
}

int main(int argc, char** argv){
    if(argc<2){printf("usage: learn <dir>\n");return 1;}
    memset(w1,0,sizeof(w1)); memset(w2,0,sizeof(w2)); // init 0
    
    char cwd[512]; recurse_dir(cwd, argv[1]); // learn 1000 iter equiv
    
    time_t t=time(0); unsigned h = t ^ (t>>16) ^ (t>>8); // 32bit->8hex hash
    char fname[13]; sprintf(fname,"%02X%02X%02X%02X.SKT", (h>>24)&0xFF,(h>>16)&0xFF,(h>>8)&0xFF,h&0xFF);
    
    FILE* f = fopen(fname,"wb");
    fwrite(w1,sizeof(w1),1,f); fwrite(b1,sizeof(b1),1,f);
    fwrite(w2,sizeof(w2),1,f); fwrite(b2,sizeof(b2),1,f);
    fclose(f); printf("Model saved: %s\n",fname); // ~300 bytes[web:26]
    return 0;
}
