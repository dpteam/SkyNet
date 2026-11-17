/* TinyAI-Learn.c - Компиляция: tcc -o tinyai-learn.exe TinyAI-Learn.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h> /* Для *nix/Win32/DJGPP. Для Turbo C под DOS заменить на <dos.h> и findfirst/findnext */

/* Архитектура: 64 входа -> 32 скрытых -> 16 выходов. Сжимаем данные до "образа". */
#define IN_SZ 64
#define HID_SZ 32
#define OUT_SZ 16
#define TOTAL_W ((IN_SZ * HID_SZ) + HID_SZ + (HID_SZ * OUT_SZ) + OUT_SZ)

/* Fixed Point Q8.8: 1.0 = 256. Используем short (16-bit) для совместимости с 8088. */
typedef short fixed;
#define F_ONE 256
#define F_LEARN 2   /* Скорость обучения */

fixed weights[TOTAL_W];
fixed in[IN_SZ], hid[HID_SZ], out[OUT_SZ], target[OUT_SZ];

/* Сигмоида для fixed point (аппроксимация) */
fixed sigmoid(fixed x) {
    if (x > 8 * F_ONE) return F_ONE;
    if (x < -8 * F_ONE) return 0;
    return (F_ONE / 2) + (x / 2); /* Линейная аппроксимация для скорости 8088 */
}

/* Прямой проход */
void forward() {
    int i, j, idx = 0;
    long sum; /* long (32-bit) для аккумулирования суммы */
    
    /* Вход -> Скрытый */
    for (i = 0; i < HID_SZ; i++) {
        sum = 0;
        for (j = 0; j < IN_SZ; j++) sum += in[j] * weights[idx++];
        hid[i] = sigmoid((fixed)(sum >> 8) + weights[idx++]); /* + bias */
    }
    /* Скрытый -> Выход */
    for (i = 0; i < OUT_SZ; i++) {
        sum = 0;
        for (j = 0; j < HID_SZ; j++) sum += hid[j] * weights[idx++];
        out[i] = sigmoid((fixed)(sum >> 8) + weights[idx++]);
    }
}

/* Обратное распространение (упрощенное) */
void backward() {
    int i, j, idx = (IN_SZ * HID_SZ) + HID_SZ; /* Начало весов скрытый->выход */
    fixed err, delta;
    
    /* Скрытый -> Выход */
    for (i = 0; i < OUT_SZ; i++) {
        err = target[i] - out[i];
        delta = (err * F_LEARN) >> 8; /* Градиент */
        for (j = 0; j < HID_SZ; j++) weights[idx + j] += hid[j] * delta >> 8;
        weights[idx + HID_SZ] += delta; /* Bias */
        idx += (HID_SZ + 1);
    }
    /* Вход -> Скрытый (простое обновление без полного backprop для экономии кода) */
    /* В минимальной версии пропускаем глубокий backprop, оставляя обучение линейным */
}

/* Хеш-функция имени файла */
void get_sketch_name(char *buf) {
    unsigned int t = (unsigned int)time(NULL);
    t = (t * 1103515245 + 12345) & 0xFFFFFFFF;
    sprintf(buf, "%08X.SKT", t);
}

/* Загрузка данных из файла в входной вектор */
void file_to_input(const char *path) {
    FILE *f = fopen(path, "rb");
    int c, i = 0;
    if (!f) return;
    memset(in, 0, sizeof(in));
    /* Читаем байты и распределяем по входам (хеширование содержимого) */
    while ((c = fgetc(f)) != EOF) {
        in[i++ % IN_SZ] ^= (short)c + (i * 31);
    }
    fclose(f);
}

/* Главная функция обучения */
int main(int argc, char **argv) {
    DIR *d;
    struct dirent *dir;
    char path[512];
    char sketch_name[16];
    int iter = 0;

    if (argc < 2) { puts("Usage: tinyai-learn <folder>"); return 1; }

    /* Инициализация весов случайными значениями */
    srand((unsigned int)time(NULL));
    for (iter = 0; iter < TOTAL_W; iter++) weights[iter] = (rand() % F_ONE) - (F_ONE / 2);

    d = opendir(argv[1]);
    if (!d) { puts("Error opening dir"); return 1; }

    printf("Learning from: %s\n", argv[1]);
    /* Эпохи обучения (проход по файлам несколько раз) */
    for (iter = 0; iter < 3; iter++) {
        rewinddir(d);
        while ((dir = readdir(d)) != NULL) {
            if (dir->d_name[0] == '.') continue;
            sprintf(path, "%s/%s", argv[1], dir->d_name);
            
            file_to_input(path);
            /* Цель обучения: совпадение выхода с хешем имени файла */
            memset(target, 0, sizeof(target));
            for (int k = 0; k < OUT_SZ; k++) target[k] = dir->d_name[k % strlen(dir->d_name)];
            
            forward();
            backward();
        }
        printf("Epoch %d done.\n", iter + 1);
    }
    closedir(d);

    /* Сохранение модели */
    get_sketch_name(sketch_name);
    FILE *out_f = fopen(sketch_name, "wb");
    if (out_f) {
        fwrite(weights, sizeof(fixed), TOTAL_W, out_f);
        fclose(out_f);
        printf("Model saved: %s\n", sketch_name);
    }
    return 0;
}