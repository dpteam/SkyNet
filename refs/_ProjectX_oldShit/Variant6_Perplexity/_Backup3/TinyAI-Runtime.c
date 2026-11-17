#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <math.h>

typedef signed short w_t;
#define L1 2
#define H1 3
#define O1 1

w_t w1[6], w2[3], b1[3], b2[1];  // L1*H1=6, H1*O1=3

float foti(w_t i){return i/2048.0;}
float sig(float x){return x>0 ? x/(1+x) : -x/(1-x);}  // fast sigmoid[web:24]

void matmul(const w_t* w, const float* in, float* out, int rows, int cols){
    for(int i=0;i<rows;i++){
        float sum = foti(b1[i]);  // bias first
        for(int j=0;j<cols;j++) sum += foti(w[i*cols+j]) * in[j];
        out[i] = sig(sum);
    }
}

void backprop(float* in, float target, float lr){  // simplified
    float h[H1], hout[O1], delta_o, delta_h[3];
    
    // forward
    matmul(w1, in, h, H1, L1);
    matmul(w2, h, hout, O1, H1);  // reuse matmul for layer2 (bias in b2)
    
    delta_o = (target - hout[0]) * hout[0] * (1 - hout[0]);
    
    // update output layer
    for(int i=0;i<H1;i++){
        w2[i] += (w_t)(lr * delta_o * h[i] * 2048);
    }
    b2[0] += (w_t)(lr * delta_o * 2048);
    
    // hidden deltas
    for(int i=0;i<H1;i++){
        delta_h[i] = delta_o * foti(w2[i]) * h[i] * (1 - h[i]);
        for(int j=0;j<L1;j++){
            w1[i*L1 + j] += (w_t)(lr * delta_h[i] * in[j] * 2048);
        }
        b1[i] += (w_t)(lr * delta_h[i] * 2048);
    }
}

float predict(float i1, float i2){
    float in[2] = {i1, i2};
    float h[3]; matmul(w1, in, h, H1, L1);
    float hout[1]; 
    float sum = foti(b2[0]);
    for(int i=0;i<H1;i++) sum += foti(w2[i]) * h[i];
    return sig(sum);
}

int load_skt(const char* fname){
    FILE* f = fopen(fname, "rb");
    if(!f) return 0;
    fread(w1, sizeof(w1), 1, f);
    fread(b1, sizeof(b1), 1, f);
    fread(w2, sizeof(w2), 1, f);
    fread(b2, sizeof(b2), 1, f);
    fclose(f);
    return 1;
}

void self_learn(const char* path){
    FILE* f = fopen(path, "rb");
    if(!f) return;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fclose(f);
    float in[2] = {(float)(sz % 100)/100.0, (float)(sz % 10)/10.0};
    backprop(in, (sz % 2 == 0) ? 0.0 : 1.0, 0.1);
}

int recurse_learn(char* buf, const char* base){
    strncpy(buf, base, 250);
    FILE* p = popen(strstr(base,"\\") ? "dir /b /s *.txt 2>nul" : "find . -name '*.txt' 2>/dev/null", "r");
    if(!p) return 0;
    char line[256];
    while(fgets(line, sizeof(line), p)){
        line[strlen(line)-1]=0;  // \n
        self_learn(line);
    }
    pclose(p);
    return 1;
}

void agent_loop(){
    printf("TinyAI Agent: pred x y | learn dir | repro | http url\n");
    char cmd[300], cwd[260];
    while(1){
        printf("> ");
        if(!fgets(cmd, sizeof(cmd), stdin)) break;
        cmd[strlen(cmd)-1]=0;
        
        if(sscanf(cmd, "pred %f %f", &cmd[0], &cmd[1])==2){  // hack parse
            float a=0,b=0; sscanf(cmd,"%f %f",&a,&b); printf("=%.3f\n", predict(a,b));
        }
        else if(strncmp(cmd, "learn ", 6)==0){
            recurse_learn(cwd, cmd+6);
            printf("Learned from %s\n", cmd+6);
        }
        else if(strcmp(cmd, "repro")==0){
            // self-copy (Windows/DOS)
            system("copy TinyAI-Runtime.exe TinyAI-Agent.exe >nul");
            printf("Reproduced: TinyAI-Agent.exe (now self-learn!)\n");
        }
        else if(strstr(cmd, "http")){
            char dl[300]; sprintf(dl, "curl -o dl.txt \"%s\" || wget -O dl.txt \"%s\"", cmd, cmd);
            system(dl);
            self_learn("dl.txt");
            printf("Learned from web: %s\n", cmd);
        }
    }
}

int main(int argc, char** argv){
    char* model = argc>1 ? argv[1] : "default.SKT";
    if(!load_skt(model)){
        printf("No model '%s', init random\n", model);
        srand(time(0));
        for(int i=0;i<sizeof(w1)/sizeof(w_t);i++) w1[i]= (rand()%4096)-2048;
        // аналогично для других
    }
    
    if(argc>2 && strcmp(argv[2],"agent")==0) agent_loop();
    else{
        printf("Test XOR: predict(0 0)=%.2f (should ~0), (1 1)=%.2f (~0)\n",
               predict(0,0), predict(1,1));
    }
    return 0;
}
