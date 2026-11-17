#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef signed short w_t;
w_t w1[6], w2[3], b1[3], b2[1];

float foti(w_t i){return i/2048.0;}
float sig(float x){return x/(1.0f+abs(x));}

void forward(float i1, float i2, float* out){
    float h[3];
    for(int j=0; j<3; j++){
        float sum = foti(b1[j]);
        sum += foti(w1[j*2]) * i1;
        sum += foti(w1[j*2+1]) * i2;
        h[j] = sig(sum);
    }
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
    for(int i=0; i<6; i++) w1[i] += (w_t)(error * 10);
    for(int i=0; i<3; i++) w2[i] += (w_t)(error * 20);
}

int scan_files(){
    FILE* p = popen("dir /b *.txt *.exe *.c 2>nul", "r");
    if(!p) return 0;
    
    int count = 0;
    char line[128];
    while(fgets(line, sizeof(line), p)){
        line[strlen(line)-1] = 0;
        FILE* f = fopen(line, "rb");
        if(f){
            fseek(f, 0, SEEK_END); long sz = ftell(f);
            fclose(f);
            float i1 = (sz%10)/5.0f;
            float i2 = (sz%3)/1.5f;
            simple_learn(i1, i2, (sz%2)?1.0f:0.0f);
            count++;
        }
    }
    pclose(p);
    return count;
}

void agent_loop(){
    printf("TinyAI Agent v3.1\n");
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
        else if(strncmp(cmd, "learn", 5)==0){
            printf("Learning...");
            int n = scan_files();
            printf(" %d files\n", n);
        }
        else if(strcmp(cmd, "repro")==0){
            system("copy runtime.exe agent.exe >nul 2>&1");
            printf("Reproduced!\n");
        }
        else if(strcmp(cmd, "quit")==0) break;
    }
}

int main(int argc, char** argv){
    if(argc > 1 && load_skt(argv[1]))
        printf("Loaded %s\n", argv[1]);
    else {
        printf("No model, default\n");
        for(int i=0; i<13; i++) ((char*)w1)[i] = 50;
    }
    
    if(argc > 2 && strcmp(argv[2], "agent")==0)
        agent_loop();
    else {
        float res;
        forward(0,1,&res); printf("0 1=%.3f\n", res);
        forward(1,0,&res); printf("1 0=%.3f\n", res);
    }
    return 0;
}
