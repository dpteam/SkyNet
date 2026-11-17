#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef signed short w_t;
w_t w1[6], w2[3], b1[3], b2[1];

float foti(w_t i){return i/2048.0f;}
float sig(float x){return x/(1.0f+abs(x));}

void forward(float i1, float i2, float* h, float* out){
    for(int j=0; j<3; j++){
        float sum = foti(b1[j]);
        sum += foti(w1[j*2+0]) * i1;
        sum += foti(w1[j*2+1]) * i2;
        h[j] = sig(sum);
    }
    float sum = foti(b2[0]);
    for(int j=0; j<3; j++) sum += foti(w2[j]) * h[j];
    *out = sig(sum);
}

void backprop(float i1, float i2, float target, float lr){
    float h[3], out;
    float in[2] = {i1, i2};
    
    forward(i1, i2, h, &out);
    
    float delta_out = (target - out) * out * (1.0f - out);
    
    for(int j=0; j<3; j++){
        float delta_h = delta_out * foti(w2[j]) * h[j] * (1.0f - h[j]);
        w2[j] += (w_t)(lr * delta_out * h[j] * 2048.0f);
        
        for(int k=0; k<2; k++){
            w1[j*2 + k] += (w_t)(lr * delta_h * in[k] * 2048.0f);
        }
        b1[j] += (w_t)(lr * delta_h * 2048.0f);
    }
    b2[0] += (w_t)(lr * delta_out * 2048.0f);
}

void learn_file(const char* fname){
    FILE* f = fopen(fname, "rb");
    if(!f) return;
    
    char buf[1024]; size_t n;
    while((n = fread(buf, 1, sizeof(buf), f)) > 0){
        for(int i=0; i<n; i++){
            float in1 = (buf[i] & 0xF0) ? 1.0f : 0.0f;
            float in2 = (buf[i] & 0x0F) ? 1.0f : 0.0f;
            float target = ((int)in1 ^ (int)in2) ? 1.0f : 0.0f;
            backprop(in1, in2, target, 0.02f);  // медленнее, точнее
        }
    }
    fclose(f);
}

int main(int argc, char** argv){
    if(argc < 2){ printf("usage: learn.exe <file>\n"); return 1; }
    
    // Инициализация НЕ нулевыми весами!
    for(int i=0; i<6; i++) w1[i] = (rand()%1000)-500;
    for(int i=0; i<3; i++) w2[i] = (rand()%1000)-500;
    for(int i=0; i<3; i++) b1[i] = (rand()%1000)-500;
    b2[0] = (rand()%1000)-500;
    
    srand(time(0));
    learn_file(argv[1]);
    
    // ✅ ФИКС fwrite - пишем БАЙТЫ, не элементы!
    time_t t = time(0);
    unsigned h = ((unsigned)t ^ (t>>16)) & 0xFFFFFFFF;
    char fname[13];
    sprintf(fname,"%02X%02X%02X%02X.SKT", 
        (h>>24)&0xFF,(h>>16)&0xFF,(h>>8)&0xFF,h&0xFF);
    
    FILE* f = fopen(fname, "wb");
    fwrite(w1, 1, sizeof(w1), f);  // ✅ 12 байт
    fwrite(b1, 1, sizeof(b1), f);  // ✅ 6 байт  
    fwrite(w2, 1, sizeof(w2), f);  // ✅ 6 байт
    fwrite(b2, 1, sizeof(b2), f);  // ✅ 2 байта
    fclose(f);
    
    printf("Model saved: %s (26 bytes, XOR-trained)\n", fname);
    return 0;
}
