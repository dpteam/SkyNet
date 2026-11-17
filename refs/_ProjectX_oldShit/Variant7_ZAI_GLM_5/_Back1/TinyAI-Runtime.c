/* TinyAI-Runtime.c - Компиляция: tcc -o tinyai-run.exe TinyAI-Runtime.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h> /* Для mkdir/FS ops */

#define IN_SZ 64
#define HID_SZ 32
#define OUT_SZ 16
#define TOTAL_W ((IN_SZ * HID_SZ) + HID_SZ + (HID_SZ * OUT_SZ) + OUT_SZ)

typedef short fixed;
#define F_ONE 256

fixed weights[TOTAL_W];
fixed in[IN_SZ], hid[HID_SZ], out[OUT_SZ];

/* Сигмоида */
fixed sigmoid(fixed x) {
    if (x > 8 * F_ONE) return F_ONE;
    if (x < -8 * F_ONE) return 0;
    return (F_ONE / 2) + (x / 2);
}

/* Прямой проход (идентичен Learn) */
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

/* Загрузка модели */
int load_model(const char *name) {
    FILE *f = fopen(name, "rb");
    if (!f) return 0;
    fread(weights, sizeof(fixed), TOTAL_W, f);
    fclose(f);
    return 1;
}

/* Поиск файла .SKT в папке */
void find_model(char *buf) {
    DIR *d = opendir(".");
    struct dirent *dir;
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            int len = strlen(dir->d_name);
            if (len > 4 && strcmp(dir->d_name + len - 4, ".SKT") == 0) {
                strcpy(buf, dir->d_name);
                closedir(d);
                return;
            }
        }
        closedir(d);
    }
    buf[0] = 0;
}

/* Главная петля агента */
int main() {
    char model_name[32];
    char file_buf[512];
    DIR *d;
    struct dirent *dir;
    int tick = 0;

    find_model(model_name);
    if (!model_name[0]) { puts("No .SKT model found!"); return 1; }
    
    printf("Runtime Active. Model: %s\n", model_name);
    if (!load_model(model_name)) return 1;

    /* Автономный цикл */
    while (1) {
        /* 1. Сенсор: Чтение файловой системы */
        d = opendir(".");
        if (!d) { sleep(1); continue; }
        
        int found_interest = 0;
        while ((dir = readdir(d)) != NULL) {
            if (dir->d_name[0] == '.' || strstr(dir->d_name, ".SKT")) continue;
            
            memset(in, 0, sizeof(in));
            /* Формируем вход из имени файла */
            for (int k = 0; k < IN_SZ; k++) {
                if (dir->d_name[k]) in[k] = dir->d_name[k];
                else break;
            }
            
            forward();
            
            /* Интерпретация выхода */
            /* out[0] > 128 -> Действие (создать файл) */
            /* out[1] > 128 -> Самообучение (модификация весов) */
            
            if (out[0] > 128) {
                printf("Agent Action: Creating log for %s\n", dir->d_name);
                FILE *log = fopen("agent_log.txt", "a");
                if (log) { fprintf(log, "Processed: %s\n", dir->d_name); fclose(log); }
            }
            
            if (out[1] > 128) {
                /* Онлайн-обучение: подкрепление связи (рефлекс) */
                weights[tick % TOTAL_W] += (rand() % 2) - 1; 
            }
        }
        closedir(d);
        
        /* 3. Самовоспроизведение / Сохранение обновленных весов */
        if (tick % 60 == 0) { /* Раз в минуту (условно) */
            FILE *f = fopen(model_name, "wb");
            if (f) {
                fwrite(weights, sizeof(fixed), TOTAL_W, f);
                fclose(f);
                printf("Model memory updated.\n");
            }
            
            /* Попытка выхода в сеть (заглушка для старых систем) */
            /* В DOS это могло бы быть открытие COM порта, тут просто месседж */
            printf("Agent checking external interfaces...\n");
        }
        
        tick++;
        sleep(1); /* Чтобы не грузить CPU на 100% */
    }
    return 0;
}