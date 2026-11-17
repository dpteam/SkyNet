/* TinyAI-Runtime.c - Universal Version */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- Cross-platform Sleep --- */
#ifdef _WIN32
  #include <windows.h>
  #define DO_SLEEP(x) Sleep(x * 1000)
#else
  #include <unistd.h>
  #define DO_SLEEP(x) sleep(x)
#endif

/* --- Cross-platform Directory Listing (Minimal) --- */
#ifdef _WIN32
  #include <io.h>
  #define FIND_DATA struct _finddata_t
  #define FIND_FIRST(p, d) _findfirst(p, d)
  #define FIND_NEXT(h, d) _findnext(h, d)
  #define FIND_CLOSE(h) _findclose(h)
  #define IS_DIR(d) (d.attrib & _A_SUBDIR)
  #define FNAME(d) d.name
#else
  #include <dirent.h>
  #define FIND_DATA struct dirent*
  /* Для POSIX реализуем обертку под стиль Windows итерации, но проще свой цикл */
#endif

typedef short fixed;
#define F_ONE 256
#define IN_SZ 64
#define HID_SZ 32
#define OUT_SZ 16
#define TOTAL_W ((IN_SZ * HID_SZ) + HID_SZ + (HID_SZ * OUT_SZ) + OUT_SZ)

fixed weights[TOTAL_W];
fixed in[IN_SZ], hid[HID_SZ], out[OUT_SZ];

fixed sigmoid(fixed x) {
    if (x > 8 * F_ONE) return F_ONE;
    if (x < -8 * F_ONE) return 0;
    return (F_ONE / 2) + (x / 2);
}

void forward() {
    int i, j, idx = 0;
    long sum;
    for (i = 0; i < HID_SZ; i++) {
        sum = 0;
        for (j = 0; j < IN_SZ; j++) sum += in[j] * weights[idx++];
        hid[i] = sigmoid((fixed)(sum >> 8) + weights[idx++]);
    }
    for (i = 0; i < OUT_SZ; i++) {
        sum = 0;
        for (j = 0; j < HID_SZ; j++) sum += hid[j] * weights[idx++];
        out[i] = sigmoid((fixed)(sum >> 8) + weights[idx++]);
    }
}

int load_model(const char *name) {
    FILE *f = fopen(name, "rb");
    if (!f) return 0;
    fread(weights, sizeof(fixed), TOTAL_W, f);
    fclose(f);
    return 1;
}

void find_model(char *buf) {
    #ifdef _WIN32
        struct _finddata_t c_file;
        intptr_t hFile;
        if ((hFile = _findfirst("*.SKT", &c_file)) != -1L) {
            strcpy(buf, c_file.name);
            _findclose(hFile);
        } else buf[0] = 0;
    #else
        DIR *d = opendir(".");
        struct dirent *dir;
        buf[0] = 0;
        if (d) {
            while ((dir = readdir(d)) != NULL) {
                if (strstr(dir->d_name, ".SKT")) {
                    strcpy(buf, dir->d_name);
                    break;
                }
            }
            closedir(d);
        }
    #endif
}

int main() {
    char model_name[32];
    int tick = 0;

    find_model(model_name);
    if (!model_name[0]) { puts("No .SKT model found!"); return 1; }
    
    printf("Runtime Active. Model: %s\n", model_name);
    if (!load_model(model_name)) return 1;

    while (1) {
        /* Сканирование текущей папки */
        #ifdef _WIN32
            struct _finddata_t c_file;
            intptr_t hFile;
            if ((hFile = _findfirst("*", &c_file)) != -1L) {
                do {
                    if (c_file.name[0] == '.') continue;
                    
                    memset(in, 0, sizeof(in));
                    for (int k=0; k<IN_SZ; k++) in[k] = c_file.name[k % strlen(c_file.name)];
                    
                    forward();
                    
                    if (out[0] > 128) {
                        FILE *log = fopen("agent_log.txt", "a");
                        if (log) { fprintf(log, "Win Action: %s\n", c_file.name); fclose(log); }
                    }
                    if (out[1] > 128) weights[tick % TOTAL_W] += (rand() % 2) - 1;
                } while (_findnext(hFile, &c_file) == 0);
                _findclose(hFile);
            }
        #else
            DIR *d = opendir(".");
            struct dirent *dir;
            if (d) {
                while ((dir = readdir(d)) != NULL) {
                    if (dir->d_name[0] == '.') continue;
                    memset(in, 0, sizeof(in));
                    for (int k=0; k<IN_SZ; k++) in[k] = dir->d_name[k % strlen(dir->d_name)];
                    
                    forward();
                    
                    if (out[0] > 128) {
                        FILE *log = fopen("agent_log.txt", "a");
                        if (log) { fprintf(log, "Unix Action: %s\n", dir->d_name); fclose(log); }
                    }
                    if (out[1] > 128) weights[tick % TOTAL_W] += (rand() % 2) - 1;
                }
                closedir(d);
            }
        #endif

        if (tick % 60 == 0) {
            FILE *f = fopen(model_name, "wb");
            if (f) { fwrite(weights, sizeof(fixed), TOTAL_W, f); fclose(f); }
        }
        
        tick++;
        DO_SLEEP(1);
    }
    return 0;
}