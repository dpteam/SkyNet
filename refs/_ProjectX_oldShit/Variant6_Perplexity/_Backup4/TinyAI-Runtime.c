#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dos.h>  // DOS INT21h

typedef signed short w_t;
w_t w1[6], w2[3], b1[3], b2[1];  // фиксированные веса

float foti(w_t i){return i/2048.0;}
float sig(float x){return x/(1.0f+abs(x));}

void forward(float i1, float i2, float* out){
    float in[2] = {i1, i2};
    float h[3];
    
    // hidden layer
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
    fread(w1, 6*2, 1, f);  // 6 весов
    fread(b1, 3*2, 1, f);
    fread(w2, 3*2, 1, f);
    fread(b2, 1*2, 1, f);
    fclose(f);
    return 1;
}

void simple_learn(float i1, float i2, float target){
    float out; forward(i1, i2, &out);
    float error = target - out;
    
    // crude gradient update (работает!)
    for(int j=0; j<6; j++) w1[j] += (w_t)(error * 20);
    for(int j=0; j<3; j++) w2[j] += (w_t)(error * 30);
    b2[0] += (w_t)(error * 40);
}

// DOS INT21h directory scan (без popen!)
int dos_files(const char* path){
    char findbuf[128] = "*.*";
    if(path) strcpy(findbuf, path);
    
    struct find_t ft;
    if(_dos_findfirst(findbuf, _A_NORMAL|_A_SUBDIR, &ft) != 0) return 0;
    
    do {
        if(!(ft.attrib & _A_SUBDIR)){  // файл, не папка
            FILE* f = fopen(ft.name, "rb");
            if(f){
                fseek(f, 0, SEEK_END); 
                long sz = ftell(f); fclose(f);
                float i1 = (sz % 1000) / 500.0;
                float i2 = (sz % 100) / 50.0;
                simple_learn(i1, i2, (sz % 2) ? 1.0 : 0.0);
            }
        }
    } while(_dos_findnext(&ft) == 0);
    return 1;
}

void agent_loop(){
    printf("TinyAI Agent v2.0 DOS-ready\n");
    printf("pred x y | learn | repro | quit\n");
    
    char cmd[80];
    while(1){
        printf("> ");
        if(!fgets(cmd, 79, stdin)) break;
        cmd[strlen(cmd)-1] = 0;  // ugh
        
        float x=0, y=0, res;
        if(sscanf(cmd, "pred %f %f", &x, &y) == 2){
            forward(x, y, &res);
            printf("=%.3f\n", res);
        }
        else if(strncmp(cmd, "learn", 5) == 0){
            printf("Scanning files...");
            int n = dos_files(0);  // current dir
            printf(" learned %d files\n", n);
        }
        else if(strcmp(cmd, "repro") == 0){
            fcloseall();  // чистим
            system("copy runtime.exe agent.exe >nul");
            printf("Reproduced: agent.exe\n");
        }
        else if(strcmp(cmd, "quit") == 0) break;
    }
}

int main(int argc, char** argv){
    printf("TinyAI Runtime (~8088 ready)\n");
    
    // init или load
    if(argc > 1 && load_skt(argv[1])){
        printf("Loaded %s (26 bytes)\n", argv[1]);
    } else {
        printf("No model, random init\n");
        for(int i=0; i<13; i++) ((char*)w1)[i] = rand() % 256 - 128;
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
