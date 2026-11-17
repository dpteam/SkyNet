/*
 * TinyAI-Runtime V8 (Sanitizer)
 * - Запрет новых строк (чтобы не ломать консольный вывод).
 * - Блокировка символов 'AI>', чтобы агент не галлюцинировал интерфейс.
 * - Top-3 Random выбор для разнообразия.
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

/* Структура для хранения кандидатов на выбор */
typedef struct {
    int id;
    int score;
} Candidate;

void init_console() {
    #ifdef _WIN32
        SetConsoleCP(65001);
        SetConsoleOutputCP(65001);
        /* Set font via API is complex, relying on user to set Lucida/Consolas manually if needed */
        printf("[Console] UTF-8 Ready.\n");
    #endif
}

int load_brain(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("[!] Cannot open brain file: %s\n", filename);
        return 0;
    }
    fread(&brain, sizeof(Network), 1, f);
    fclose(f);
    printf("[+] Brain loaded: %s (Sanitizer ON)\n", filename);
    return 1;
}

neuron_t activate(int x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (neuron_t)x;
}

/* 
 * think_mode:
 * 0 = Deterministic (Обучение)
 * 1 = Stochastic Top-3 (Генерация)
 */
int think(int input_byte, int learn_byte, int mode) {
    int i, j, sum;
    neuron_t new_hidden[HIDDEN_SIZE];
    int output_sum[OUTPUT_SIZE];
    
    int idx = input_byte & 0xFF; 

    /* 1. Forward */
    for (i = 0; i < HIDDEN_SIZE; i++) {
        sum = brain.w_ih[idx][i];
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
        output_sum[i] = sum;
    }

    /* 2. Выбор ответа (Selection) */
    int predicted = 0;
    
    if (mode == 0) {
        /* Обучение: Берем лучший */
        int max_val = -999999;
        for (i = 0; i < OUTPUT_SIZE; i++) {
            if (output_sum[i] > max_val) {
                max_val = output_sum[i];
                predicted = i;
            }
        }
    } else {
        /* Генерация: Top-3 Random */
        /* Ищем 3 лучших кандидата */
        Candidate c1 = {0, -999999};
        Candidate c2 = {0, -999999};
        Candidate c3 = {0, -999999};
        
        for (i = 0; i < OUTPUT_SIZE; i++) {
            int s = output_sum[i];
            /* Вставка в топ-3 */
            if (s > c1.score) {
                c3 = c2; c2 = c1; c1.id = i; c1.score = s;
            } else if (s > c2.score) {
                c3 = c2; c2.id = i; c2.score = s;
            } else if (s > c3.score) {
                c3.id = i; c3.score = s;
            }
        }
        
        /* Выбираем случайного из топ-3 */
        int r = rand() % 3;
        if (r == 0) predicted = c1.id;
        else if (r == 1) predicted = c2.id;
        else predicted = c3.id;
    }

    /* 3. Backpropagation */
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
    
    /* НЕ сбрасываем память */
    
    printf("\nAI> ");
    
    for (int i = 0; i < chars; i++) {
        current = think(current, -1, 1);
        
        /* --- ФИЛЬТР САНИТАРИИ --- */
        
        /* 1. Блокируем символы промпта, чтобы не ломать консоль */
        if (current == 'A' || current == 'I' || current == '>') {
            current = '.';
        }
        
        /* 2. Блокируем переносы строк */
        if (current == '\n' || current == '\r') {
            current = ','; /* Заменяем на запятую */
        }
        
        /* 3. Блокируем непечатные символы (меньше 32) */
        if (current < 32) {
            current = ' ';
        }
        
        /* Вывод */
        putchar(current);
        fflush(stdout);
    }
    printf("\n");
}

void study_file(const char *fname) {
    FILE *f = fopen(fname, "rb");
    if (!f) return;
    int passes = 1; 
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
        printf("[!] EVOLVED: %s\n", fname);
    }
}

void explore_environment() {
}

void check_internet() {
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    init_console();
    
    printf("TinyAI-Runtime V8 (Sanitized) Started.\n");
    
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
                        think(input[i], -1, 0);
                    }
                }
                dream(150); /* Длинный ответ */
            }
        } else {
            printf("\n[Cycle %d] Thinking...\n", cycle);
            explore_environment();
            if (cycle % 5 == 0) check_internet();
            dream(100); 
            if (cycle % 15 == 0) evolve();
        }

        #ifdef _WIN32
            Sleep(1000);
        #else
            sleep(1);
        #endif
    }

    return 0;
}