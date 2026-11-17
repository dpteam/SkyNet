/*
 * TinyAI-Runtime V7 (Finally Talking)
 * 1. Сохраняет контекст пользователя (не стирает память).
 * 2. Увеличен "Шум" (Temperature) x100, чтобы бороться с пробелами.
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

void init_console() {
    #ifdef _WIN32
        SetConsoleCP(65001);
        SetConsoleOutputCP(65001);
        printf("[Console] UTF-8 Mode Active.\n");
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
    printf("[+] Brain loaded: %s (Ready)\n", filename);
    return 1;
}

neuron_t activate(int x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (neuron_t)x;
}

/* 
 * mode:
 * 0 = Точный (обучение/ввод)
 * 1 = Креативный (генерация с высоким шумом)
 */
int think(int input_byte, int learn_byte, int mode) {
    int i, j, sum;
    neuron_t new_hidden[HIDDEN_SIZE];
    int output_sum[OUTPUT_SIZE];
    int predicted = 0;
    int max_val = -999999;
    
    /* Безопасный индекс */
    int idx = input_byte & 0xFF; 

    /* 1. Forward Pass */
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
        
        /* --- ГЛАВНОЕ ИСПРАВЛЕНИЕ: Большой Шум --- */
        if (mode == 1) {
            /* Суммы могут быть около 100,000. Шум +/- 2000 достаточно, чтобы сбить выбор */
            sum += (rand() % 4000) - 2000;
        }
        
        output_sum[i] = sum;
        if (sum > max_val) {
            max_val = sum;
            predicted = i;
        }
    }

    /* 2. Backpropagation (Обучение) */
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

    /* ВАЖНО: НЕ СБРАСЫВАЕМ ПАМЯТЬ (memset убран) */
    /* Теперь агент продолжает мысль, которая осталась в hidden после ввода */

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

        /* Если это пробел, иногда заменяем на случайную букву (чтобы избежать пустых строк) */
        if (current == ' ' && (rand() % 10) == 0) {
             current = 'a' + (rand() % 26); /* Латинские буквы как стимул */
        }
        
        if (current != 0) {
            putchar(current);
        }
        fflush(stdout);
        
        if (current == '\n') {
            printf("AI> ");
            fflush(stdout);
        }
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
    
    printf("TinyAI-Runtime V7 (Active) Started.\n");
    
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
                        think(input[i], -1, 0); /* Заполняем память контекстом */
                    }
                }
                dream(100); /* Длинный ответ */
            }
        } else {
            printf("\n[Cycle %d] Thinking...\n", cycle);
            explore_environment();
            if (cycle % 5 == 0) check_internet();
            dream(60); 
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