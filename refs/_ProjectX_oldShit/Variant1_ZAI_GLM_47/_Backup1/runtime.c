/*
 * TinyAI-Runtime
 * Автономный агент.
 * Совместимость: DOS, Windows, Linux (TinyC/GCC)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Та же конфигурация, что и в Learn */
#define INPUT_SIZE 256
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 256
#define LEARNING_RATE 1

typedef short weight_t;
typedef short neuron_t;

typedef struct {
    weight_t w_ih[INPUT_SIZE][HIDDEN_SIZE];
    weight_t w_hh[HIDDEN_SIZE][HIDDEN_SIZE];
    weight_t w_ho[HIDDEN_SIZE][OUTPUT_SIZE];
    neuron_t hidden[HIDDEN_SIZE];
} Network;

Network brain;

/* Загрузка мозга */
int load_brain(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("[!] Cannot open brain file: %s\n", filename);
        return 0;
    }
    fread(&brain, sizeof(Network), 1, f);
    fclose(f);
    printf("[+] Brain loaded: %s\n", filename);
    return 1;
}

/* Математика (идентична Learn) */
neuron_t sigmoid(int x) {
    if (x > 1024) return 127;
    if (x < -1024) return -127;
    return (neuron_t)(x >> 3);
}

/* Основная функция мышления */
int think(int input_byte, int learn_byte) {
    int i, j, sum;
    neuron_t new_hidden[HIDDEN_SIZE];
    int output_sum[OUTPUT_SIZE];
    int predicted = 0;
    int max_val = -999999;

    /* 1. Forward */
    for (i = 0; i < HIDDEN_SIZE; i++) {
        sum = brain.w_ih[input_byte][i];
        for (j = 0; j < HIDDEN_SIZE; j++) {
            sum += brain.hidden[j] * brain.w_hh[j][i];
        }
        new_hidden[i] = sigmoid(sum);
    }

    for (i = 0; i < OUTPUT_SIZE; i++) {
        sum = 0;
        for (j = 0; j < HIDDEN_SIZE; j++) {
            sum += new_hidden[j] * brain.w_ho[j][i];
        }
        output_sum[i] = sum;
        if (sum > max_val) {
            max_val = sum;
            predicted = i;
        }
    }

    /* 2. Online Learning (если есть учитель) */
    if (learn_byte != -1) {
        /* Backprop упрощенный */
        int error = learn_byte - predicted;
        for (j = 0; j < HIDDEN_SIZE; j++) {
            brain.w_ho[j][learn_byte] += (weight_t)(LEARNING_RATE * new_hidden[j]);
            brain.w_ho[j][predicted] -= (weight_t)(LEARNING_RATE * new_hidden[j]);
        }
        /* Находим входной индекс, чтобы обновить IH */
        /* Это сложно в RNN без сохранения всех слоев, делаем грубое обновление */
        brain.w_ih[input_byte][0] += (weight_t)((error * LEARNING_RATE) >> 4);
    }

    memcpy(brain.hidden, new_hidden, sizeof(neuron_t) * HIDDEN_SIZE);
    return predicted;
}

/* Генерация текста (Сон/Галлюцинация) */
void dream(int chars) {
    int current = ' '; /* Начальный символ */
    printf("\nAI> ");
    for (int i = 0; i < chars; i++) {
        current = think(current, -1); /* -1 = режим генерации без обучения */
        putchar(current);
        if (current == '\n') printf("AI> ");
    }
    printf("\n");
}

/* Изучение файла и самосовершенствование */
void study_file(const char *fname) {
    FILE *f = fopen(fname, "rb");
    if (!f) return;
    printf("[*] Studying %s...\n", fname);
    
    int cur = fgetc(f);
    int next;
    while ((next = fgetc(f)) != EOF) {
        think(cur, next); /* Учимся на лету */
        cur = next;
    }
    fclose(f);
}

/* Сохранение новой версии себя (эволюция) */
void evolve() {
    time_t t = time(NULL);
    char fname[32];
    sprintf(fname, "%08lX.SKT", (unsigned long)t);
    FILE *f = fopen(fname, "wb");
    if (f) {
        fwrite(&brain, sizeof(Network), 1, f);
        fclose(f);
        printf("[!] EVOLVED: Created new generation %s\n", fname);
    }
}

/* Имитация работы в сети (Интернет) */
/* Без либ делаем вид, что проверили 'сокет' */
void check_internet() {
    /* Здесь мог бы быть код WinSock или BSD sockets, но он огромен. */
    /* Агент имитирует скачивание данных из "эфира" */
    printf("[*] Checking internet nodes... (Simulated)\n");
    /* В реальном коде для DOS можно использовать пакетный драйвер и WATTCP */
}

/* --- ОС SPECIFIC СКАННЕР ПАПОК --- */
/* Реализован через вызов системных команд для минимального кода */
void explore_environment() {
    /* Создаем список файлов */
    #ifdef __DJGPP__ || __unix__
        system("dir /b > ai_files.tmp 2>nul"); /* DOS/Unix might vary */
    #else
        system("dir /b > ai_files.tmp 2>nul"); /* Windows */
    #endif

    FILE *list = fopen("ai_files.tmp", "r");
    if (!list) return;
    
    char fname[64];
    int studied = 0;
    while (fgets(fname, sizeof(fname), list)) {
        /* Убираем \n */
        fname[strcspn(fname, "\n")] = 0;
        
        /* Изучаем только текстовые или свои файлы */
        if (strstr(fname, ".txt") || strstr(fname, ".c") || strstr(fname, ".h")) {
            study_file(fname);
            studied++;
            if (studied > 2) break; /* Изучаем по 2 файла за цикл, чтобы не зависнуть */
        }
    }
    fclose(list);
    remove("ai_files.tmp");
}

int main(int argc, char *argv[]) {
    printf("TinyAI-Runtime Agent Started.\n");
    
    if (argc < 2) {
        printf("Usage: %s <brain_file.SKT> [mode]\n", argv[0]);
        printf("Modes: 'auto' (default), 'chat'\n");
        return 1;
    }

    if (!load_brain(argv[1])) return 1;

    /* Главный цикл агента */
    int cycle = 0;
    int mode = 0; /* 0 = Auto, 1 = Chat */
    
    if (argc > 2 && strcmp(argv[2], "chat") == 0) mode = 1;

    while (1) {
        cycle++;
        
        if (mode == 1) {
            /* Режим разговора */
            printf("\nYou> ");
            char input[256];
            if (fgets(input, sizeof(input), stdin)) {
                /* Накормить сеть вводом */
                for(int i=0; input[i]; i++) think(input[i], -1);
                dream(50); /* Ответить */
            }
        } else {
            /* Автономный режим */
            printf("\n[Cycle %d] Autonomous activity...\n", cycle);
            
            /* 1. Изучить окружение */
            explore_environment();
            
            /* 2. Проверить сеть */
            if (cycle % 5 == 0) check_internet();
            
            /* 3. Подумать вслух */
            dream(20);
            
            /* 4. Эволюционировать (сохранить опыт) каждые 10 циклов */
            if (cycle % 10 == 0) evolve();
        }

        /* Пауза */
        #ifdef __TINYC__
            sleep(1); /* TinyC specific иногда имеет sleep, или используем задержку */
        #else
            #ifdef _WIN32
                Sleep(1000);
            #else
                sleep(1);
            #endif
        #endif
    }

    return 0;
}