/*
 * TinyAI-Runtime V4 (Stable & Creative)
 * - Исправлен бесконечный цикл переводов строк.
 * - Добавлен "Шум" (Temperature) для разнообразия.
 * - Очистка памяти перед генерацией.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

#define INPUT_SIZE 256
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 256
#define LEARNING_RATE 3

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
    printf("[+] Brain loaded: %s (Fixed Size: %d bytes)\n", filename, sizeof(Network));
    return 1;
}

neuron_t activate(int x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (neuron_t)x;
}

/* 
 * think_mode: 
 * 0 = Точный расчет (как при обучении)
 * 1 = Творческий (с шумом)
 */
int think(int input_byte, int learn_byte, int think_mode) {
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
        new_hidden[i] = activate(sum);
    }

    for (i = 0; i < OUTPUT_SIZE; i++) {
        sum = 0;
        for (j = 0; j < HIDDEN_SIZE; j++) {
            sum += new_hidden[j] * brain.w_ho[j][i];
        }
        
        /* --- ТВОРЧЕСКИЙ ШУМ (Temperature) --- */
        /* Добавляем случайность, чтобы не застревать на одном символе */
        if (think_mode == 1) {
            sum += (rand() % 10) - 5; 
        }
        
        output_sum[i] = sum;
        if (sum > max_val) {
            max_val = sum;
            predicted = i;
        }
    }

    /* 2. Backpropagation (Только если учимся) */
    if (learn_byte != -1) {
        int error = learn_byte - predicted;
        int rnd_h = rand() % HIDDEN_SIZE;
        
        for (j = 0; j < HIDDEN_SIZE; j++) {
            brain.w_ho[j][learn_byte] += (weight_t)(LEARNING_RATE * new_hidden[j]);
            if (predicted != learn_byte)
                 brain.w_ho[j][predicted] -= (weight_t)(LEARNING_RATE * new_hidden[j]);
        }
        brain.w_ih[input_byte][rnd_h] += (weight_t)((error * LEARNING_RATE) >> 3);
        brain.w_hh[rnd_h][rnd_h] += (weight_t)((error * LEARNING_RATE) >> 4);
    }

    memcpy(brain.hidden, new_hidden, sizeof(neuron_t) * HIDDEN_SIZE);
    return predicted;
}

void dream(int chars) {
    int current = ' ';
    int newline_counter = 0; /* Счетчик переносов строк */

    /* --- САМОЕ ВАЖНОЕ: Очищаем память перед сном --- */
    memset(brain.hidden, 0, sizeof(brain.hidden));

    printf("\nAI> ");
    
    for (int i = 0; i < chars; i++) {
        /* Генерируем с режимом 1 (С шумом) */
        current = think(current, -1, 1);
        
        /* --- ИНТЕЛЛЕКТ: Фильтр пустых строк --- */
        if (current == '\n' || current == '\r') {
            newline_counter++;
            /* Если подряд 3 переноса строки - меняем на пробел */
            if (newline_counter >= 3) {
                current = ' ';
                newline_counter = 0;
            }
        } else {
            newline_counter = 0;
        }

        /* Фильтр непечатных символов (кроме перевода строки) */
        if (current < 32 && current != 10 && current != 13) {
            current = ' '; 
        }
        
        putchar(current);
        fflush(stdout);
        
        if (current == '\n') {
            printf("AI> ");
            fflush(stdout);
        }
    }
    printf("\n");
}

/* Обучение (используем think_mode 0) */
void study_file(const char *fname) {
    FILE *f = fopen(fname, "rb");
    if (!f) return;
    int passes = 2; 
    int p;
    for(p = 0; p < passes; p++) {
        fseek(f, 0, SEEK_SET);
        int cur = fgetc(f);
        int next;
        while ((next = fgetc(f)) != EOF) {
            think(cur, next, 0); /* Режим точного обучения */
            cur = next;
        }
    }
    fclose(f);
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
    /* Не используем самообучение в этой версии, чтобы сохранить знания о книге */
}

void check_internet() {
    /* Заглушка */
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL)); /* Инициализация генератора случайных чисел */
    
    printf("TinyAI-Runtime V4 Started.\n");
    #ifdef _WIN32
        printf("[Hint] Run 'chcp 866' for correct Russian text.\n");
    #endif
    
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
            fflush(stdout);
            
            char input[1024]; 
            if (fgets(input, sizeof(input), stdin)) {
                /* Ввод кормим сети точно (mode 0) */
                for(int i=0; input[i]; i++) {
                    if(input[i] != '\n' && input[i] != '\r') {
                        think(input[i], -1, 0);
                    }
                }
                dream(60); /* Ответ творческий (mode 1) */
            }
        } else {
            /* Автономный режим */
            printf("\n[Cycle %d] Autonomous activity...\n", cycle);
            explore_environment();
            if (cycle % 5 == 0) check_internet();
            dream(50); 
            if (cycle % 15 == 0) evolve();
        }

        #ifdef _WIN32
            Sleep(1500);
        #else
            sleep(1);
        #endif
    }

    return 0;
}