/*
 * TinyAI-Runtime V5 (CRASH FIX)
 * - Исправлен краш на русском языке (Signed Char bug).
 * - Добавлена защита от отрицательных индексов (& 0xFF).
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
    printf("[+] Brain loaded: %s (Safe Mode)\n", filename);
    return 1;
}

neuron_t activate(int x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (neuron_t)x;
}

/* 
 * think_mode: 0 = Точный (обучение), 1 = Творческий (генерация)
 */
int think(int input_byte, int learn_byte, int think_mode) {
    int i, j, sum;
    neuron_t new_hidden[HIDDEN_SIZE];
    int output_sum[OUTPUT_SIZE];
    int predicted = 0;
    int max_val = -999999;

    /* --- ЗАЩИТА ОТ КРАША --- */
    /* Приводим любой вход (даже отрицательный от русского символа) к 0..255 */
    int idx = input_byte & 0xFF; 

    /* 1. Forward */
    for (i = 0; i < HIDDEN_SIZE; i++) {
        sum = brain.w_ih[idx][i]; /* Используем безопасный индекс */
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
        
        if (think_mode == 1) {
            sum += (rand() % 20) - 10; /* Чуть больше шума для разнообразия */
        }
        
        output_sum[i] = sum;
        if (sum > max_val) {
            max_val = sum;
            predicted = i;
        }
    }

    /* 2. Backpropagation (Только если учимся) */
    if (learn_byte != -1) {
        int target_idx = learn_byte & 0xFF;
        int error = target_idx - predicted;
        int rnd_h = rand() % HIDDEN_SIZE;
        
        for (j = 0; j < HIDDEN_SIZE; j++) {
            brain.w_ho[j][target_idx] += (weight_t)(LEARNING_RATE * new_hidden[j]);
            if (predicted != target_idx)
                 brain.w_ho[j][predicted] -= (weight_t)(LEARNING_RATE * new_hidden[j]);
        }
        brain.w_ih[idx][rnd_h] += (weight_t)((error * LEARNING_RATE) >> 3);
        brain.w_hh[rnd_h][rnd_h] += (weight_t)((error * LEARNING_RATE) >> 4);
    }

    memcpy(brain.hidden, new_hidden, sizeof(neuron_t) * HIDDEN_SIZE);
    return predicted;
}

void dream(int chars) {
    int current = ' ';
    int newline_counter = 0;
    memset(brain.hidden, 0, sizeof(brain.hidden)); /* Сброс памяти */

    printf("\nAI> ");
    
    for (int i = 0; i < chars; i++) {
        current = think(current, -1, 1);
        
        /* Фильтр пустых строк */
        if (current == '\n' || current == '\r') {
            newline_counter++;
            if (newline_counter >= 3) {
                current = ' ';
                newline_counter = 0;
            }
        } else {
            newline_counter = 0;
        }

        /* Фильтр непечатных символов */
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

/* Обучение */
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
            think(cur, next, 0);
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
    /* Отключено */
}

void check_internet() {
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    
    printf("TinyAI-Runtime V5 (CrashFixed) Started.\n");
    #ifdef _WIN32
        printf("[Hint] Ensure 'chcp 866' is active.\n");
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
                for(int i=0; input[i]; i++) {
                    if(input[i] != '\n' && input[i] != '\r') {
                        /* Передаем как есть, & 0xFF сработает внутри think */
                        think(input[i], -1, 0);
                    }
                }
                dream(60);
            }
        } else {
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