/* TinyAI-Learn.c - Universal Version */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- Cross-platform Directory Listing --- */
#ifdef _WIN32
  #include <io.h> /* Для _findfirst, _findnext */
  #define SCAN_DIR(path, callback, data) scan_dir_win(path, callback, data)
  void scan_dir_win(const char *path, void (*cb)(const char*, void*), void *data) {
    struct _finddata_t c_file;
    intptr_t hFile;
    char search[512];
    sprintf(search, "%s/*", path);
    
    if ((hFile = _findfirst(search, &c_file)) == -1L) return;
    do {
        if (c_file.name[0] == '.') continue;
        char full[512];
        sprintf(full, "%s/%s", path, c_file.name);
        if (c_file.attrib & _A_SUBDIR) {
            scan_dir_win(full, cb, data); /* Рекурсия */
        } else {
            cb(full, data);
        }
    } while (_findnext(hFile, &c_file) == 0);
    _findclose(hFile);
  }
#else
  /* Для DJGPP, Linux, Unix */
  #include <dirent.h>
  #include <sys/stat.h>
  #define SCAN_DIR(path, callback, data) scan_dir_posix(path, callback, data)
  void scan_dir_posix(const char *path, void (*cb)(const char*, void*), void *data) {
    DIR *d = opendir(path);
    if (!d) return;
    struct dirent *dir;
    char full[512];
    while ((dir = readdir(d)) != NULL) {
        if (dir->d_name[0] == '.') continue;
        sprintf(full, "%s/%s", path, dir->d_name);
        
        /* Проверка на папку (упрощенно) */
        DIR *sub = opendir(full);
        if (sub) { closedir(sub); scan_dir_posix(full, cb, data); }
        else cb(full, data);
    }
    closedir(d);
  }
#endif

/* --- TinyAI Core (Fixed Point 16-bit) --- */
typedef short fixed;
#define F_ONE 256
#define F_LEARN 2

#define IN_SZ 64
#define HID_SZ 32
#define OUT_SZ 16
#define TOTAL_W ((IN_SZ * HID_SZ) + HID_SZ + (HID_SZ * OUT_SZ) + OUT_SZ)

fixed weights[TOTAL_W];
fixed in[IN_SZ], hid[HID_SZ], out[OUT_SZ], target[OUT_SZ];

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

void backward() {
    int i, j, idx = (IN_SZ * HID_SZ) + HID_SZ;
    fixed err, delta;
    for (i = 0; i < OUT_SZ; i++) {
        err = target[i] - out[i];
        delta = (err * F_LEARN) >> 8;
        for (j = 0; j < HID_SZ; j++) weights[idx + j] += hid[j] * delta >> 8;
        weights[idx + HID_SZ] += delta;
        idx += (HID_SZ + 1);
    }
}

/* --- File Processing --- */
void process_file(const char *path, void *data) {
    int *epoch = (int*)data;
    FILE *f = fopen(path, "rb");
    int c, i = 0;
    if (!f) return;

    memset(in, 0, sizeof(in));
    while ((c = fgetc(f)) != EOF) in[i++ % IN_SZ] ^= (short)c + (i * 31);
    fclose(f);

    /* Цель: хеш имени файла */
    memset(target, 0, sizeof(target));
    for (i = 0; i < OUT_SZ; i++) target[i] = path[i % strlen(path)];

    forward();
    backward();
}

void get_sketch_name(char *buf) {
    unsigned int t = (unsigned int)time(NULL);
    t = (t * 1103515245 + 12345);
    sprintf(buf, "%08X.SKT", t);
}

int main(int argc, char **argv) {
    char sketch_name[16];
    int epoch, i;

    if (argc < 2) { puts("Usage: tinyai-learn <folder>"); return 1; }

    srand((unsigned int)time(NULL));
    for (i = 0; i < TOTAL_W; i++) weights[i] = (rand() % F_ONE) - (F_ONE / 2);

    printf("Learning from: %s\n", argv[1]);
    
    for (epoch = 0; epoch < 3; epoch++) {
        printf("Epoch %d...\n", epoch+1);
        SCAN_DIR(argv[1], process_file, &epoch);
    }

    get_sketch_name(sketch_name);
    FILE *out_f = fopen(sketch_name, "wb");
    if (out_f) {
        fwrite(weights, sizeof(fixed), TOTAL_W, out_f);
        fclose(out_f);
        printf("Model saved: %s\n", sketch_name);
    }
    return 0;
}