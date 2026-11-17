/* TinyAI-Runtime - Autonomous self-learning agent
   Picks .SKT files automatically, learns from local FS, interacts
   Compile: tcc runtime.c -o runtime.exe
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

#define IN_SIZE   64
#define HID_SIZE  32
#define OUT_SIZE  64
#define LR        0.005f  /* Lower LR for self-learning */
#define SKT_MAGIC "SKT"

/* Platform detection */
#ifdef __DJGPP__
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#define HAS_DIR 1
#elif defined(_WIN32)
#include <windows.h>
#define HAS_DIR 1
#else
#ifdef _POSIX_SOURCE
#include <dirent.h>
#include <sys/stat.h>
#define HAS_DIR 1
#endif
#endif

typedef struct { int in,hid,out; float *w1,*b1,*w2,*b2; } Net;

/* Math */
float sigm(float x) {
    if(x<-4.0f)return 0.0f; if(x>4.0f)return 1.0f;
    return 0.5f+0.5f*x/(1.0f+(x<0?-x:x));
}
float dsigm(float y){return y*(1.0f-y);}

/* Network ops */
Net* net_new(int in,int hid,int out){
    Net*n=malloc(sizeof(Net)); n->in=in;n->hid=hid;n->out=out;
    n->w1=malloc(in*hid*4);n->b1=malloc(hid*4);
    n->w2=malloc(hid*out*4);n->b2=malloc(out*4);
    srand((unsigned)time(NULL));
    int i; for(i=0;i<in*hid;i++)n->w1[i]=((float)rand()/RAND_MAX-0.5f)*0.1f;
    for(i=0;i<hid;i++)n->b1[i]=0;
    for(i=0;i<hid*out;i++)n->w2[i]=((float)rand()/RAND_MAX-0.5f)*0.1f;
    for(i=0;i<out;i++)n->b2[i]=0;
    return n;
}
void net_free(Net*n){free(n->w1);free(n->b1);free(n->w2);free(n->b2);free(n);}

Net* load_skt(const char*fn){
    FILE*f=fopen(fn,"rb"); char m[4]; unsigned char ver,in,hid,out;
    Net*n; int i;
    if(!f)return NULL;
    fread(m,3,1,f); m[3]=0;
    if(strcmp(m,SKT_MAGIC)!=0){fclose(f);return NULL;}
    fread(&ver,1,1,f); fread(&in,1,1,f); fread(&hid,1,1,f); fread(&out,1,1,f);
    n=net_new(in,hid,out);
    fread(n->w1,4,in*hid,f); fread(n->b1,4,hid,f);
    fread(n->w2,4,hid*out,f); fread(n->b2,4,out,f);
    fclose(f); return n;
}
void save_skt(Net*n,const char*fn){
    FILE*f=fopen(fn,"wb"); unsigned char ver=1;
    fwrite(SKT_MAGIC,3,1,f); fwrite(&ver,1,1,f);
    fwrite(&n->in,1,1,f); fwrite(&n->hid,1,1,f); fwrite(&n->out,1,1,f);
    fwrite(n->w1,4,n->in*n->hid,f); fwrite(n->b1,4,n->hid,f);
    fwrite(n->w2,4,n->hid*n->out,f); fwrite(n->b2,4,n->out,f);
    fclose(f);
}

void forward(Net*n,float*x,float*h,float*y){
    int i,j;
    for(i=0;i<n->hid;i++){
        float s=n->b1[i];
        for(j=0;j<n->in;j++)s+=x[j]*n->w1[j*n->hid+i];
        h[i]=sigm(s);
    }
    for(i=0;i<n->out;i++){
        float s=n->b2[i];
        for(j=0;j<n->hid;j++)s+=h[j]*n->w2[j*n->out+i];
        y[i]=sigm(s);
    }
}
void train_step(Net*n,float*x,float*t){ /* single step backprop */
    float h[64],y[64],dy[64],dh[64]; /* fixed max for simplicity */
    int i,j;
    forward(n,x,h,y);
    for(i=0;i<n->out;i++)dy[i]=(t[i]-y[i])*dsigm(y[i]);
    for(i=0;i<n->hid;i++){
        float s=0; for(j=0;j<n->out;j++)s+=dy[j]*n->w2[i*n->out+j];
        dh[i]=s*dsigm(h[i]);
    }
    for(i=0;i<n->hid;i++)for(j=0;j<n->out;j++)n->w2[i*n->out+j]+=LR*dy[j]*h[i];
    for(i=0;i<n->out;i++)n->b2[i]+=LR*dy[i];
    for(i=0;i<n->in;i++)for(j=0;j<n->hid;j++)n->w1[i*n->hid+j]+=LR*dh[j]*x[i];
    for(i=0;i<n->hid;i++)n->b1[i]+=LR*dh[i];
}

/* Self-learning from file (autonomous improvement) */
void assimilate(Net*n,const char*fn){
    FILE*f=fopen(fn,"rb");
    unsigned char b[64];
    int cnt=0;
    if(!f)return;
    printf("[Assimilating %s]\n",fn);
    while(fread(b,1,64,f)==64 && cnt<50){ /* limit per file to prevent lock */
        float x[64],t[64]; int i;
        for(i=0;i<64;i++)x[i]=b[i]/255.0f;
        for(i=0;i<OUT_SIZE;i++)t[i]=x[i];
        train_step(n,x,t);
        cnt++;
    }
    fclose(f);
}

#ifdef HAS_DIR
void scan_assimilate(Net*n,const char*root){
#ifdef _WIN32
    WIN32_FIND_DATA fd; HANDLE h; char mask[256];
    sprintf(mask,"%s\\*",root);
    h=FindFirstFile(mask,&fd);
    if(h==INVALID_HANDLE_VALUE)return;
    do{
        char p[256];
        if(fd.cFileName[0]=='.')continue;
        sprintf(p,"%s\\%s",root,fd.cFileName);
        if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)scan_assimilate(n,p);
        else if(strstr(fd.cFileName,".txt")||strstr(fd.cFileName,".c")||strstr(fd.cFileName,".h"))
            assimilate(n,p);
    }while(FindNextFile(h,&fd));
    FindClose(h);
#else
    DIR*d=opendir(root); struct dirent*e; struct stat s; char p[512];
    if(!d)return;
    while((e=readdir(d))!=NULL){
        if(e->d_name[0]=='.')continue;
        sprintf(p,"%s/%s",root,e->d_name);
        if(stat(p,&s)==0){
            if(S_ISDIR(s.st_mode))scan_assimilate(n,p);
            else if(strstr(e->d_name,".txt")||strstr(e->d_name,".c")||strstr(e->d_name,".h"))
                assimilate(n,p);
        }
    }
    closedir(d);
#endif
}
#else
void scan_assimilate(Net*n,const char*r){printf("No FS scan support\n");(void)n;(void)r;}
#endif

/* Interactive agent mode */
void interact(Net*n){
    char line[128];
    float h[64],y[64];
    printf("Agent ready. Input (Enter=quit):\n");
    while(fgets(line,sizeof(line),stdin)){
        if(line[0]=='\n')break;
        float x[64]={0}; int i,l=strlen(line);
        if(l>64)l=64;
        for(i=0;i<l;i++)x[i]=((unsigned char)line[i])/255.0f;
        forward(n,x,h,y);
        printf("Response: ");
        for(i=0;i<l;i++){
            int c=(int)(y[i]*255.0);
            if(c>=32&&c<127)putchar(c);else putchar('?');
        }
        printf("\n");
    }
}

/* Autonomous reproduction (save evolved state) */
void reproduce(Net*n){
    char name[13];
    unsigned long h=(unsigned long)time(NULL);
    sprintf(name,"%08lX.SKT",h&0xFFFFFFFF);
    save_skt(n,name);
    printf("[Reproduced to %s]\n",name);
}

int main(int argc,char**argv){
    Net*n=NULL; char target[256]="";
    
    printf("TinyAI-Runtime 1.0 [Autonomous Agent]\n");
    
    /* Load model: arg or auto-detect */
    if(argc>1){
        n=load_skt(argv[1]);
        if(n)printf("Loaded: %s\n",argv[1]);
    }else{
#ifdef HAS_DIR
#ifdef _WIN32
        WIN32_FIND_DATA fd; HANDLE h=FindFirstFile("*.SKT",&fd);
        if(h!=INVALID_HANDLE_VALUE){strcpy(target,fd.cFileName);FindClose(h);}
#else
        DIR*d=opendir("."); struct dirent*e;
        if(d){while((e=readdir(d))!=NULL)if(strstr(e->d_name,".SKT")){strcpy(target,e->d_name);break;} closedir(d);}
#endif
#endif
        if(target[0]){n=load_skt(target); if(n)printf("Auto-loaded: %s\n",target);}
    }
    
    if(!n){
        printf("No model found, spawning genesis network...\n");
        n=net_new(IN_SIZE,HID_SIZE,OUT_SIZE);
    }
    
    /* Autonomous learning phase (self-modification) */
    printf("Entering autonomous learning phase...\n");
    scan_assimilate(n,".");
    reproduce(n); /* Save improved version */
    
    /* Interactive mode */
    interact(n);
    
    /* Final save */
    reproduce(n);
    net_free(n);
    return 0;
}