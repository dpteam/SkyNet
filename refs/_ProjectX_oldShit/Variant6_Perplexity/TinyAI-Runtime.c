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
        sum += foti(w1[j*2+0]) * i1;
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
    
    // ✅ ТОЧНАЯ СИНХРОНИЗАЦИЯ с Learn: fwrite(1, sizeof(), f)
    fread(w1, 1, sizeof(w1), f);    // 12 байт
    fread(b1, 1, sizeof(b1), f);    // 6 байт  
    fread(w2, 1, sizeof(w2), f);    // 6 байт
    fread(b2, 1, sizeof(b2), f);    // 2 байта
    
    fclose(f);
    return 1;
}

void simple_learn(float i1, float i2, float target){
    float out; forward(i1, i2, &out);
    float error = target - out;
    for(int i=0; i<6; i++) w1[i] += (w_t)(error * 20);
    b2[0] += (w_t)(error * 30);
}

int scan_files(){
    // DOS/Windows dir pipe
    FILE* p = popen("dir /b *.txt *.c *.exe *.SKT 2>nul", "r");
    if(!p) return 0;
    
    int count = 0;
    char line[128];
    while(fgets(line, sizeof(line), p)){
        line[strlen(line)-1] = 0;  // rm \n
        
        FILE* f = fopen(line, "rb");
        if(f){
            fseek(f, 0, SEEK_END); 
            long sz = ftell(f); 
            fclose(f);
            
            float i1 = (sz % 10) / 5.0f;
            float i2 = (sz % 3) / 1.5f;
            simple_learn(i1, i2, (sz % 2) ? 1.0f : 0.0f);
            count++;
        }
    }
    pclose(p);
    return count;
}

void agent_loop(){
    printf("TinyAI Agent v3.2 (XOR Толстого!)\n");
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
            printf("Learning...");
            int n = scan_files();
            printf(" %d files\n", n);
        }
        else if(strcmp(cmd, "repro") == 0){
            system("copy runtime.exe agent.exe >nul 2>&1 || cp runtime.exe agent");
            printf("Reproduced: agent.exe\n");
        }
        else if(strcmp(cmd, "quit") == 0) break;
        else printf("?\n");
    }
}

int main(int argc, char** argv){
    printf("TinyAI v3.2 (~8088)\n");
    
    if(argc > 1 && load_skt(argv[1])){
        printf("Loaded %s (26B)\n", argv[1]);
    } else {
        printf("No model - random\n");
        srand(123);
        for(int i=0; i<13; i++) ((w_t*)w1)[i] = (rand()%4096)-2048;
    }
    
    if(argc > 2 && strcmp(argv[2],"agent") == 0)
        agent_loop();
    else {
        float res;
        forward(0,1,&res); printf("XOR 0 1=%.3f\n", res);
        forward(1,0,&res); printf("XOR 1 0=%.3f\n", res);
        forward(1,1,&res); printf("XOR 1 1=%.3f\n", res);
    }
    return 0;
}
