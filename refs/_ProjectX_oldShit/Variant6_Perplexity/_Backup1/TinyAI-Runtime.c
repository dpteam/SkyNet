#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

typedef signed short w_t;
#define L1 2 #define H1 3 #define O1 1
w_t w1[L1*H1], w2[H1*O1], b1[H1], b2[O1];
float sig(float x){return x/(1+abs(x));}
void matmul(const w_t* w, const float* in, float* out, int rows, int cols){ /* same as above */ }
float predict(float i1, float i2){ // agent predict
    float in[L1] = {i1,i2,1.0}; float h[H1]; matmul(w1, in, h, H1, L1); float hout[H1+1]; 
    for(int j=0;j<H1;j++) hout[j]=sig(foti(b1[j])+h[j]); hout[H1]=1.0;
    float out[O1]; matmul(w2, hout, out, O1, H1+1); return out[0]+foti(b2[0]);
}

int load_skt(const char* fname){
    FILE* f=fopen(fname,"rb"); if(!f) return 0;
    fread(w1,sizeof(w1),1,f); fread(b1,sizeof(b1),1,f);
    fread(w2,sizeof(w2),1,f); fread(b2,sizeof(b2),1,f); fclose(f); return 1;
}

void self_learn_dir(char* cwd, char* path){ // augment weights from new data
    strcpy(cwd,path); FILE* p = popen("dir /b /s * 2>nul || ls -R 2>/dev/null","r");
    char buf[256]; while(fgets(buf,sizeof(buf),p)){
        float sz = (float)strlen(buf); backprop((float[]){sz/100,1.0,1.0}, sz>10?1:0, 0.05); // simple
    } pclose(p);
}

void agent_mode(){ // autonomous: predict, act, learn, self-reproduce
    printf("TinyAI Agent v1.0 (predict 0-1, learn dir, repro)\n");
    char cmd[256]; while(1){
        float inp1=0,inp2=0; printf("> "); if(!fgets(cmd,sizeof(cmd),stdin)) break;
        if(sscanf(cmd,"%f %f",&inp1,&inp2)==2) printf("pred: %.2f\n", predict(inp1,inp2));
        else if(strncmp(cmd,"learn ",6)==0) self_learn_dir(cmd,""); // self-learn
        else if(strcmp(cmd,"repro\n")==0){ // self-reproduce: copy+mutate
            FILE* s=fopen("TinyAI-Runtime.exe","rb"); FILE* d=fopen("TinyAI-Agent.exe","wb");
            char buf[1024]; size_t n; while((n=fread(buf,1,1024,s))>0) fwrite(buf,1,n,d);
            fclose(s); fclose(d); printf("Reproduced: TinyAI-Agent.exe\n");
        }
        // internet/files: dummy wget/curl via system("wget...")
        if(strstr(cmd,"http")){ system(cmd); self_learn_dir(cmd,"downloaded"); }
    }
}

int main(int argc, char** argv){
    if(argc>1 && load_skt(argv[1])) printf("Loaded %s\n",argv[1]);
    else{ printf("No model, random agent\n"); memset(w1,1,sizeof(w1)); } // default
    
    if(argc>2 && strcmp(argv[2],"agent")==0) agent_mode();
    else printf("Runtime ready. pred(0.5 0.5)=%.2f\n",predict(0.5,0.5));
    return 0;
}
