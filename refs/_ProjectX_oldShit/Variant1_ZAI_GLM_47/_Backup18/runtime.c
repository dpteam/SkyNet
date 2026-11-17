/*
 * TinyAI-Runtime V16 (The Explorer / Runner-Up Mode)
 * - "Нейронный Штраф": Если сеть выдала букву 'И', она НЕЛЬЗЯ выбрать 'И' снова.
 * - Это заставляет перебирать вторые и третьи по силе варианты.
 * - Результат: Очень разнообразный текст без потери структуры.
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
#define PENALTY 50000 /* Штраф для предыдущей буквы (очень много) */

typedef short weight_t;
typedef short neuron_t;

typedef struct {
    weight_t w_ih[INPUT_SIZE][HIDDEN_SIZE];
    weight_t w_hh[HIDDEN_SIZE][HIDDEN_SIZE];
    weight_t w_ho[HIDDEN_SIZE][OUTPUT_SIZE];
    neuron_t hidden[HIDDEN_SIZE];
} Network;

Network brain;

/* Глобальная память для запрета повторов */
int g_last_pred = 0;

typedef struct {
    int id;
    int score;
} Candidate;

void init_console() {
    #ifdef _WIN32
        SetConsoleOutputCP(866);
        SetConsoleCP(866);
        printf("[Console] Russian DOS (CP 866).\n");
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
    printf("[+] Brain loaded: %s (Explorer Mode)\n", filename);
    return 1;
}

neuron_t activate(int x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (neuron_t)x;
}

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

    /* 2. Selection */
    int predicted = 0;
    Candidate c1 = {0, -999999};
    Candidate c2 = {0, -999999};
    Candidate c3 = {0, -999999};
    
    for (i = 0; i < OUTPUT_SIZE; i++) {
        int s = output_sum[i];
        
        /* --- ЛОГИКА ИССЛЕДОВАТЕЛЯ --- */
        if (mode == 1) {
            /* Небольшой шум */
            s += (rand() % 4000) - 2000;
            
            /* ГЛАВНЫЙ ТРЮК: Убиваем оценку предыдущей буквы */
            /* Сеть не сможет выбрать g_last_pred, если есть что-то хоть на 1 очко лучше */
            if (i == g_last_pred) {
                s -= PENALTY;
            }
            
            /* Небольшой бонус кириллице (128-255) */
            if (i > 128) s += 3000;
        }
        
        if (s > c1.score) { c3 = c2; c2 = c1; c1.id = i; c1.score = s; }
        else if (s > c2.score) { c3 = c2; c2.id = i; c2.score = s; }
        else if (s > c3.score) { c3.id = i; c3.score = s; }
    }
    
    int r = rand() % 3;
    if (r == 0) predicted = c1.id;
    else if (r == 1) predicted = c2.id;
    else predicted = c3.id;

    /* 3. Backprop */
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
    g_last_pred = 0; /* Сброс запрета перед сном */
    
    printf("\nAI> ");
    
    for (int i = 0; i < chars; i++) {
        current = think(current, -1, 1);
        
        /* Обновляем запрет */
        g_last_pred = current;
        
        /* Очистка */
        if (current == 10 || current == 13) current = ' ';
        if (current < 32) current = ' ';
        
        putchar(current);
        fflush(stdout);
    }
    printf("\n");
}

void study_file(const char *fname) {
}

void evolve() {
}

void explore_environment() {
}

void check_internet() {
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    init_console();
    
    printf("TinyAI-Runtime V16 (Explorer) Started.\n");
    
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
                    think(input[i], -1, 0);
                }
                dream(150);
            }
        } else {
            printf("\n[Cycle %d] Thinking...\n", cycle);
            dream(100);
        }

        #ifdef _WIN32
            Sleep(1000);
        #else
            sleep(1);
        #endif
    }

    return 0;
}