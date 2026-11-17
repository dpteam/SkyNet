#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef signed short w_t;
w_t w1[6], w2[3], b1[3], b2[1];

float foti(w_t i){return i/2048.0;}
float sig(float x){return x/(1.0f+abs(x));}

void forward(float i1, float i2, float* out){
    float in[2] = {i1, i2};
    float h[3];
    
    // hidden layer (manual matmul)
    for(int j=0; j<3; j++){
        float sum = foti(b1[j]);
        sum += foti(w1[j*2+0]) * i1;
        sum += foti(w1[j*2+1]) * i2;
        h[j] = sig(sum);
    }
    
    // output layer
    float sum = foti(b2[0]);
    for(int j=0; j<3; j++) sum += foti(w2[j]) * h[j];
    *out = sig(sum);
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

void simple_learn(float i1, float i2, float target){
    float out; forward(i1, i2, &out);
    float error = target - out;
    
    // градиентное улучшение (грубое, но работает)
    w1[0] += (w_t)(error * 30);
    w1[1] += (w_t)(error * 30);
    w1[2] += (w_t)(error * 25);
    b2[0] += (w_t)(error * 40);
}

int scan_files(){
    int learned = 0;
    char* patterns[] = {"*.txt", "*.c", "*.h", "*.*", 0};
    
    for(int p=0; patterns[p]; p++){
        char cmd[128];
        sprintf(cmd, "dir /b %s 2>nul | findstr . >nul", patterns[p]);
        FILE* pf = popen(cmd, "r");
        if(pf){
            char line[128];
            while(fgets(line, sizeof(line), pf)){
                line[strlen(line)-1]=0;
                FILE* f = fopen(line, "rb");
                if(f){
                    fseek(f, 0, SEEK_END); 
                    long sz = ftell(f); fclose(f);
                    float i1 = (sz % 1000) / 500.0;
                    float i2 = (sz % 100) / 50.0;
                    simple_learn(i1, i2, (sz % 2) ? 1.0 : 0.0);
                    learned++;
                }
            }
            pclose(pf);
        }
    }
    return learned;
}

void agent_loop(){
    printf("TinyAI Agent v2.1 ANSI\n");
    printf("pred x y | learn | repro | quit\n");
    
    char cmd[80];
    while(1){
        printf("> ");
        if(!fgets(cmd, 79, stdin)) break;
        cmd[strlen(cmd)-1] = 0;
        
        float x=0, y=0, res;
        if(sscanf(cmd, "pred %f %f", &x, &y) == 2){
            forward(x, y, &res);
            printf("=%.3f\n", res);
        }
        else if(strncmp(cmd, "learn", 5) == 0){
            printf("Scanning...");
            int n = scan_files();
            printf(" learned %d files\n", n);
        }
        else if(strcmp(cmd, "repro") == 0){
            system("copy runtime.exe agent.exe >nul 2>&1");
            printf("Reproduced: agent.exe\n");
        }
        else if(strcmp(cmd, "quit") == 0) break;
        else printf("?\n");
    }
}

int main(int argc, char** argv){
    printf("TinyAI v2.1 (~26B model)\n");
    
    if(argc > 1 && load_skt(argv[1])){
        printf("Loaded %s\n", argv[1]);
    } else {
        printf("Random init\n");
        for(int i=0; i<13; i++) 
            ((w_t*)w1)[i] = (rand()%4096) - 2048;
    }
    
    if(argc > 2 && strcmp(argv[2], "agent") == 0)
        agent_loop();
    else {
        float res;
        forward(0,0, &res); printf("0 0 -> %.3f\n", res);
        forward(1,1, &res); printf("1 1 -> %.3f\n", res);
    }
    return 0;
}
