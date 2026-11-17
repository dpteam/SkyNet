/*
 * TinyAI-Learn (FIXED: Real directory scanning)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- КОНФИГУРАЦИЯ --- */
#define INPUT_SIZE 256
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 256
#define LEARNING_RATE 2

typedef short weight_t;
typedef short neuron_t;

typedef struct {
    weight_t w_ih[INPUT_SIZE][HIDDEN_SIZE];
    weight_t w_hh[HIDDEN_SIZE][HIDDEN_SIZE];
    weight_t w_ho[HIDDEN_SIZE][OUTPUT_SIZE];
    neuron_t hidden[HIDDEN_SIZE];
} Network;

Network net;

void init_net(Network *net) {
    int i, j;
    srand((unsigned int)time(NULL));
    for(i = 0; i < INPUT_SIZE; i++)
        for(j = 0; j < HIDDEN_SIZE; j++)
            net->w_ih[i][j] = (weight_t)(rand() % 256 - 128);
    
    for(i = 0; i < HIDDEN_SIZE; i++)
        for(j = 0; j < HIDDEN_SIZE; j++)
            net->w_hh[i][j] = (weight_t)(rand() % 256 - 128);
            
    for(i = 0; i < HIDDEN_SIZE; i++)
        for(j = 0; j < OUTPUT_SIZE; j++)
            net->w_ho[i][j] = (weight_t)(rand() % 256 - 128);

    for(i = 0; i < HIDDEN_SIZE; i++) net->hidden[i] = 0;
}

/* Активация (быстрая, без плавающей точки) */
neuron_t activate(int x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (neuron_t)x;
}

/* Обучение на одном файле */
void train_file(Network *net, const char *fname) {
    FILE *f = fopen(fname, "rb");
    if (!f) return;
    
    printf("Learning: %s\n", fname);
    
    int cur, next;
    int count = 0;
    cur = fgetc(f);
    
    while ((next = fgetc(f)) != EOF) {
        /* Прямой проход + Обучение */
        int i, j, sum;
        neuron_t new_hidden[HIDDEN_SIZE];
        int output_sum[OUTPUT_SIZE];
        int predicted = 0;
        int max_val = -999999;

        /* 1. Forward */
        for (i = 0; i < HIDDEN_SIZE; i++) {
            sum = net->w_ih[cur][i];
            for (j = 0; j < HIDDEN_SIZE; j++) {
                sum += net->hidden[j] * net->w_hh[j][i];
            }
            new_hidden[i] = activate(sum);
        }

        for (i = 0; i < OUTPUT_SIZE; i++) {
            sum = 0;
            for (j = 0; j < HIDDEN_SIZE; j++) {
                sum += new_hidden[j] * net->w_ho[j][i];
            }
            output_sum[i] = sum;
            if (sum > max_val) {
                max_val = sum;
                predicted = i;
            }
        }

        /* 2. Backpropagation (Обучение) */
        if (predicted != next) {
            int error = next - predicted;
            
            /* Обновляем HO */
            for (j = 0; j < HIDDEN_SIZE; j++) {
                net->w_ho[j][next] += (weight_t)(LEARNING_RATE * new_hidden[j]);
                net->w_ho[j][predicted] -= (weight_t)(LEARNING_RATE * new_hidden[j]);
            }
            
            /* Обновляем IH и HH (Упрощенно для скорости) */
            int rnd_h = rand() % HIDDEN_SIZE;
            net->w_ih[cur][rnd_h] += (weight_t)((error * LEARNING_RATE) >> 3);
            net->w_hh[rnd_h][rnd_h] += (weight_t)((error * LEARNING_RATE) >> 4);
        }

        /* Обновляем состояние */
        memcpy(net->hidden, new_hidden, sizeof(neuron_t) * HIDDEN_SIZE);
        cur = next;
        
        count++;
        if (count % 10000 == 0) { printf("."); fflush(stdout); } /* Прогресс-бар */
    }
    fclose(f);
    printf(" Done.\n");
}

void save_model(Network *net, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) { perror("Error saving"); return; }
    fwrite(net, sizeof(Network), 1, f);
    fclose(f);
    printf("[+] Model saved to: %s (%d bytes)\n", filename, sizeof(Network));
}

/* Получение списка файлов (Windows/Unix) */
void scan_and_train(Network *net, const char *path) {
    char command[256];
    char list_file[] = "ai_learn_list.tmp";
    
    /* Создаем список всех файлов в папке */
    #ifdef _WIN32
        sprintf(command, "dir /s /b \"%s\" > %s 2>nul", path, list_file);
    #else
        sprintf(command, "find \"%s\" -type f > %s 2>/dev/null", path, list_file);
    #endif
    
    system(command);
    
    FILE *list = fopen(list_file, "r");
    if (!list) {
        printf("[!] Could not generate file list. Is path correct?\n");
        return;
    }
    
    char fname[512];
    int files_count = 0;
    while (fgets(fname, sizeof(fname), list)) {
        /* Убираем перенос строки */
        fname[strcspn(fname, "\r\n")] = 0;
        train_file(net, fname);
        files_count++;
    }
    
    fclose(list);
    remove(list_file);
    printf("[+] Processed %d files.\n", files_count);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <folder_path> [output_name]\n", argv[0]);
        return 1;
    }

    init_net(&net);
    printf("TinyAI-Learn Started. Target: %s\n", argv[1]);

    /* Основное обучение */
    scan_and_train(&net, argv[1]);

    char outname[32];
    if (argc > 2) strncpy(outname, argv[2], 31);
    else {
        time_t t = time(NULL);
        sprintf(outname, "%08lX.SKT", (unsigned long)t);
    }
    
    save_model(&net, outname);
    return 0;
}