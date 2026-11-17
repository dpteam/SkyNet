/*
 * TinyAI-Runtime (Header Fix)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Подключаем правильные заголовки для задержки времени */
#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

/* Настройки сети */
#define INPUT_SIZE 256
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 256
#define LEARNING_RATE 3 /* Увеличили для скорости обучения */

typedef short weight_t;
typedef short neuron_t;

typedef struct {
    weight_t w_ih[INPUT_SIZE][HIDDEN_SIZE];
    weight_t w_hh[HIDDEN_SIZE][HIDDEN_SIZE];
    weight_t w_ho[HIDDEN_SIZE][OUTPUT_SIZE];
    neuron_t hidden[HIDDEN_SIZE];
} Network;

Network brain;

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

/* Новая функция активации: Квази-Tanh.
 * Держит значения в пределах байта (-128..127), но сохраняет градиент. */
neuron_t activate(int x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (neuron_t)x;
}

/* Основной цикл мышления */
int think(int input_byte, int learn_byte) {
    int i, j, sum;
    neuron_t new_hidden[HIDDEN_SIZE];
    int output_sum[OUTPUT_SIZE];
    int predicted = 0;
    int max_val = -999999;

    /* 1. Расчет скрытого слоя */
    for (i = 0; i < HIDDEN_SIZE; i++) {
        sum = brain.w_ih[input_byte][i];
        /* Рекурсия: влияние прошлого */
        for (j = 0; j < HIDDEN_SIZE; j++) {
            sum += brain.hidden[j] * brain.w_hh[j][i];
        }
        new_hidden[i] = activate(sum);
    }

    /* 2. Расчет выхода */
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

    /* 3. Обучение (Backpropagation через скрытый слой) */
    if (learn_byte != -1) {
        int error = learn_byte - predicted;
        
        /* Обновляем веса Hidden->Output (награждаем правильный, штрафуем ошибочный) */
        for (j = 0; j < HIDDEN_SIZE; j++) {
            brain.w_ho[j][learn_byte] += (weight_t)(LEARNING_RATE * new_hidden[j]);
            if (predicted != learn_byte)
                 brain.w_ho[j][predicted] -= (weight_t)(LEARNING_RATE * new_hidden[j]);
        }

        /* Обновляем веса Input->Hidden и Hidden->Hidden 
           (Обновляем только входной нейрон и один случайный скрытый для экономии ресурсов) */
        int rnd_hidden = rand() % HIDDEN_SIZE;
        
        brain.w_ih[input_byte][rnd_hidden] += (weight_t)((error * LEARNING_RATE) >> 3);
        brain.w_hh[rnd_hidden][rnd_hidden] += (weight_t)((error * LEARNING_RATE) >> 4);
    }

    memcpy(brain.hidden, new_hidden, sizeof(neuron_t) * HIDDEN_SIZE);
    return predicted;
}

/* Генерация текста с фильтрацией мусора */
void dream(int chars) {
    int current = ' ';
    printf("\nAI> ");
    for (int i = 0; i < chars; i++) {
        current = think(current, -1);
        
        /* --- ИСПРАВЛЕНИЕ ГЛЮКОВ ВЫВОДА --- */
        /* Если AI выдал мусор (управляющие коды или > 127), заменяем на пробел */
        if (current < 32 || current > 126) {
            current = ' '; 
        }
        
        putchar(current);
        if (current == '\n') printf("AI> ");
    }
    printf("\n");
}

/* Изучение файла с повторениями (Ускоренное обучение) */
void study_file(const char *fname) {
    FILE *f = fopen(fname, "rb");
    if (!f) return;
    
    /* Проходимся по файлу несколько раз, чтобы лучше запомнить */
    int passes = 3; 
    int p;
    
    for(p = 0; p < passes; p++) {
        fseek(f, 0, SEEK_SET);
        int cur = fgetc(f);
        int next;
        while ((next = fgetc(f)) != EOF) {
            think(cur, next);
            cur = next;
        }
    }
    fclose(f);
    printf("[*] Studied %s (%d passes)\n", fname, passes);
}

void evolve() {
    time_t t = time(NULL);
    char fname[32];
    sprintf(fname, "%08lX.SKT", (unsigned long)t);
    FILE *f = fopen(fname, "wb");
    if (f) {
        fwrite(&brain, sizeof(Network), 1, f);
        fclose(f);
        printf("[!] EVOLVED: New generation %s\n", fname);
    }
}

void explore_environment() {
    /* Команда для получения списка файлов */
    #ifdef _WIN32
        system("dir /b > ai_files.tmp 2>nul");
    #else
        system("ls > ai_files.tmp 2>/dev/null");
    #endif

    FILE *list = fopen("ai_files.tmp", "r");
    if (!list) return;
    
    char fname[64];
    int studied = 0;
    while (fgets(fname, sizeof(fname), list)) {
        fname[strcspn(fname, "\n")] = 0;
        
        /* Учим только свои исходники и текст */
        if (strstr(fname, ".c") || strstr(fname, ".h") || strstr(fname, ".txt")) {
            study_file(fname);
            studied++;
            /* Увеличили лимит изучаемых файлов за цикл */
            if (studied > 5) break; 
        }
    }
    fclose(list);
    remove("ai_files.tmp");
}

void check_internet() {
    printf("[*] Checking internet nodes... (Simulated)\n");
}

int main(int argc, char *argv[]) {
    printf("TinyAI-Runtime Agent V2 Started.\n");
    
    if (argc < 2) {
        printf("Usage: %s <brain_file.SKT> [mode]\n", argv[0]);
        return 1;
    }

    if (!load_brain(argv[1])) return 1;

    int cycle = 0;
    int mode = 0; 
    
    if (argc > 2 && strcmp(argv[2], "chat") == 0) mode = 1;

    while (1) {
        cycle++;
        
        if (mode == 1) {
            printf("\nYou> ");
            char input[256];
            if (fgets(input, sizeof(input), stdin)) {
                for(int i=0; input[i]; i++) {
                    if(input[i] >= 32 && input[i] <= 126) think(input[i], -1);
                }
                dream(50);
            }
        } else {
            /* Автономный режим */
            printf("\n[Cycle %d] Autonomous activity...\n", cycle);
            
            explore_environment();
            
            if (cycle % 5 == 0) check_internet();
            
            dream(40); /* Немного больше текста */
            
            if (cycle % 10 == 0) evolve();
        }

        #ifdef _WIN32
            Sleep(1000);
        #else
            sleep(1);
        #endif
    }

    return 0;
}