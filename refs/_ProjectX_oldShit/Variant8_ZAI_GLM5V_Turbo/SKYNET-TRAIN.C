/*
 * ============================================================
 *  SKYNET-TRAIN.C - Neural Network Trainer for 8088
 *  Compiler: Turbo C++ 2.01 (Tiny Model) / Small-C
 *  Target: IBM PC/XT (8088, 4.77MHz, 512KB RAM)
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 *  CONFIGURATION - Размер сети под 8088
 * ============================================================ */
#define INPUT_SIZE      8       /* Входов (сенсоров) */
#define HIDDEN_SIZE     12      /* Нейронов скрытого слоя */
#define OUTPUT_SIZE     4       /* Выходов (действий) */
#define TRAIN_SAMPLES   50      /* Образцов в датасете */
#define EPOCHS          1000    /* Эпох обучения */
#define LEARNING_RATE   64      /* Fixed-point: 0.25 = 64/256 */

/* ============================================================
 *  FIXED-POINT ARITHMETIC (Q8.8 format)
 *  Вместо float используем int16_t: 
 *    старшие 8 бит = целая часть
 *    младшие 8 бит = дробная часть
 * ============================================================ */
typedef int16_t fp_t;          /* Q8.8 fixed point */
typedef int32_t fp_acc_t;      /* Accumulator for multiply */

#define FP_SHIFT        8
#define FP_ONE          256     /* 1.0 in Q8.8 */
#define FP_HALF         128     /* 0.5 in Q8.8 */

#define INT_TO_FP(x)    ((fp_t)((x) << FP_SHIFT))
#define FP_TO_INT(x)    ((x) >> FP_SHIFT)
#define FP_MUL(a,b)     ((fp_t)(((fp_acc_t)(a) * (b)) >> FP_SHIFT))

/* ============================================================
 *  ASSEMBLY OPTIMIZATIONS (inline для Turbo C)
 *  Для Small-C - отдельный .ASM файл
 * ============================================================ */

#ifdef __TURBOC__
/* Быстрое умножение fixed-point на 8088 */
#pragma inline
static fp_t fast_mul_fp(fp_t a, fp_t b) {
    fp_t result;
    asm {
        mov ax, a       ; Загрузить a
        imul b          ; AX * b -> DX:AX
        shr ax, 1       ; Сдвиг на 8 бит (2 сдвига по 4)
        shr ax, 1
        shr ax, 1
        shr ax, 1
        shr ax, 1
        shr ax, 1
        shr ax, 1
        shr ax, 1
        mov result, ax  ; Сохранить результат
    }
    return result;
}

/* Быстрая сигмоида через Lookup Table (256 байт!) */
extern const unsigned char sigmoid_lut[256];

#pragma inline
static fp_t fast_sigmoid(fp_t x) {
    unsigned char index;
    fp_t result;
    asm {
        mov al, byte ptr x+1  ; Старший байт (знак + целая часть)
        cbw                     ; Расширить знак
        add al, 128             ; Смещение для индекса (-128..127 -> 0..255)
        mov index, al
    }
    result = (fp_t)sigmoid_lut[index] << FP_SHIFT;
    return result;
}
#else
/* Fallback для Small-C (без inline asm) */
#define fast_mul_fp(a,b)  FP_MUL(a,b)
#endif

/* ============================================================
 *  NETWORK STRUCTURE
 * ============================================================ */
typedef struct {
    /* Веса: input->hidden (INPUT_SIZE * HIDDEN_SIZE) */
    fp_t w_ih[INPUT_SIZE][HIDDEN_SIZE];
    /* Веса: hidden->output (HIDDEN_SIZE * OUTPUT_SIZE) */
    fp_t w_ho[HIDDEN_SIZE][OUTPUT_SIZE];
    /* Байасы */
    fp_t b_h[HIDDEN_SIZE];
    fp_t b_o[OUTPUT_SIZE];
} Network;

/* Активации (выделяем в BSS/статике) */
static fp_t hidden[HIDDEN_SIZE];
static fp_t output[OUTPUT_SIZE];
static fp_t hidden_deriv[HIDDEN_SIZE];  /* Для backprop */

/* ============================================================
 *  LOOKUP TABLE FOR SIGMOID (генерируется заранее)
 *  Экономит МНОГО циклов CPU!
 * ============================================================ */
#ifdef __TURBOC__
const unsigned char sigmoid_lut[256] = {
    /* Предрассчитано: sigmoid(x) для x в [-6.0, +2.0] */
    2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7,
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 27, 30,
    33, 36, 39, 43, 47, 51, 56, 61, 67, 73, 79, 86, 94, 102, 111, 120,
    130, 141, 152, 164, 177, 190, 204, 218, 233, 248, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    /* ... остальная таблица ... */
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};
#endif

/* ============================================================
 *  FORWARD PASS (Inference)
 * ============================================================ */
void forward_pass(Network *net, fp_t inputs[INPUT_SIZE]) {
    int i, j;
    fp_acc_t acc;
    
    /* === Скрытый слой === */
    for (i = 0; i < HIDDEN_SIZE; i++) {
        acc = 0;
        for (j = 0; j < INPUT_SIZE; j++) {
            acc += (fp_acc_t)inputs[j] * net->w_ih[j][i];
        }
        acc += net->b_h[i] << FP_SHIFT;  /* Байас */
        hidden[i] = fast_sigmoid((fp_t)(acc >> FP_SHIFT));
    }
    
    /* === Выходной слой === */
    for (i = 0; i < OUTPUT_SIZE; i++) {
        acc = 0;
        for (j = 0; j < HIDDEN_SIZE; j++) {
            acc += (fp_acc_t)hidden[j] * net->w_ho[j][i];
        }
        acc += net->b_o[i] << FP_SHIFT;
        output[i] = fast_sigmoid((fp_t)(acc >> FP_SHIFT));
    }
}

/* ============================================================
 *  BACKPROPAGATION (Обучение)
 *  Упрощённый SGD без momentum (экономия памяти)
 * ============================================================ */
void train_step(Network *net, fp_t inputs[INPUT_SIZE], 
                fp_t targets[OUTPUT_SIZE]) {
    int i, j;
    fp_t error, delta;
    
    /* 1. Forward pass (уже посчитаны output[]) */
    forward_pass(net, inputs);
    
    /* 2. Вычисляем ошибки выходного слоя */
    fp_t output_deltas[OUTPUT_SIZE];  /* В стеке! */
    for (i = 0; i < OUTPUT_SIZE; i++) {
        error = targets[i] - output[i];
        /* производная sigmoid: y*(1-y), упрощённо */
        output_deltas[i] = FP_MUL(error, FP_MUL(output[i], 
                                     (FP_ONE - output[i])));
    }
    
    /* 3. Ошибки скрытого слоя */
    for (i = 0; i < HIDDEN_SIZE; i++) {
        acc = 0;
        for (j = 0; j < OUTPUT_SIZE; j++) {
            acc += (fp_acc_t)output_deltas[j] * net->w_ho[i][j];
        }
        delta = (fp_t)(acc >> FP_SHIFT);
        hidden_deriv[i] = FP_MUL(delta, FP_MUL(hidden[i],
                                  (FP_ONE - hidden[i])));
    }
    
    /* 4. Обновление весов hidden->output */
    for (i = 0; i < HIDDEN_SIZE; i++) {
        for (j = 0; j < OUTPUT_SIZE; j++) {
            net->w_ho[i][j] += FP_MUL(LEARNING_RATE, 
                             FP_MUL(output_deltas[j], hidden[i]));
        }
        net->b_o[j] += FP_MUL(LEARNING_RATE, output_deltas[j]);
    }
    
    /* 5. Обновление весов input->hidden */
    for (i = 0; i < INPUT_SIZE; i++) {
        for (j = 0; j < HIDDEN_SIZE; j++) {
            net->w_ih[i][j] += FP_MUL(LEARNING_RATE,
                             FP_MUL(hidden_deriv[j], inputs[i]));
        }
        net->b_h[j] += FP_MUL(LEARNING_RATE, hidden_deriv[j]);
    }
}

/* ============================================================
 *  DATASET: Простая задача классификации
 *  Пример: управление простым агентом
 *  Входы: [дист_до_цели, угол, скорость, энергия, ...]
 *  Выходы: [вперёд, назад, влево, вправо]
 * ============================================================ */
typedef struct {
    fp_t inputs[INPUT_SIZE];
    fp_t targets[OUTPUT_SIZE];
} Sample;

Sample dataset[TRAIN_SAMPLES] = {
    /* Образец 0: цель впереди -> идти вперёд */
    {{INT_TO_FP(100), INT_TO_FP(0), INT_TO_FP(10), INT_TO_FP(90),
      INT_TO_FP(1), INT_TO_FP(0), INT_TO_FP(0), INT_TO_FP(0)},
     {INT_TO_FP(1), INT_TO_FP(0), INT_TO_FP(0), INT_TO_FP(0)}},
     
    /* Образец 1: цель слева -> поворот влево */
    {{INT_TO_FP(70), INT_TO_FP(-45), INT_TO_FP(5), INT_TO_FP(85),
      INT_TO_FP(1), INT_TO_FP(0), INT_TO_FP(0), INT_TO_FP(0)},
     {INT_TO_FP(0), INT_TO_FP(0), INT_TO_FP(1), INT_TO_FP(0)}},
     
    /* ... больше образцов ... */
};

/* ============================================================
 *  SAVE/LOAD WEIGHTS (для дискеты 360KB/1.44MB)
 * ============================================================ */
int save_network(Network *net, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;
    
    /* Magic number */
    uint16_t magic = 0x534E /* 'SN' */;
    fwrite(&magic, 2, 1, f);
    
    /* Архитектура */
    uint8_t arch[3] = {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    fwrite(arch, 1, 3, f);
    
    /* Веса (raw binary) */
    fwrite(net->w_ih, sizeof(fp_t), INPUT_SIZE * HIDDEN_SIZE, f);
    fwrite(net->w_ho, sizeof(fp_t), HIDDEN_SIZE * OUTPUT_SIZE, f);
    fwrite(net->b_h, sizeof(fp_t), HIDDEN_SIZE, f);
    fwrite(net->b_o, sizeof(fp_t), OUTPUT_SIZE, f);
    
    fclose(f);
    return 0;
}

int load_network(Network *net, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return -1;
    
    uint16_t magic;
    fread(&magic, 2, 1, f);
    if (magic != 0x534E) { fclose(f); return -2; }  /* Invalid format */
    
    uint8_t arch[3];
    fread(arch, 1, 3, f);
    /* TODO: проверка соответствия архитектуры */
    
    fread(net->w_ih, sizeof(fp_t), INPUT_SIZE * HIDDEN_SIZE, f);
    fread(net->w_ho, sizeof(fp_t), HIDDEN_SIZE * OUTPUT_SIZE, f);
    fread(net->b_h, sizeof(fp_t), HIDDEN_SIZE, f);
    fread(net->b_o, sizeof(fp_t), OUTPUT_SIZE, f);
    
    fclose(f);
    return 0;
}

/* ============================================================
 *  MAIN: Training Loop
 * ============================================================ */
int main(void) {
    Network net;
    int epoch, sample;
    fp_t total_error;
    
    printf("SKYNET Training System v0.1\n");
    printf("Target: 8088 CPU, 512KB RAM\n\n");
    
    /* Инициализация весов (Xavier initialization, упрощённо) */
    srand(12345);  /* Фиксированный seed для воспроизводимости */
    {
        int i, j;
        fp_t scale = INT_TO_FP(1);  /* ~0.29 для 8 входов */
        for (i = 0; i < INPUT_SIZE; i++)
            for (j = 0; j < HIDDEN_SIZE; j++)
                net.w_ih[i][j] = (rand() % 256 - 128) * scale / 128;
        
        for (i = 0; i < HIDDEN_SIZE; i++)
            for (j = 0; j < OUTPUT_SIZE; j++)
                net.w_ho[i][j] = (rand() % 256 - 128) * scale / 128;
        
        for (i = 0; i < HIDDEN_SIZE; i++)
            net.b_h[i] = 0;
        for (i = 0; i < OUTPUT_SIZE; i++)
            net.b_o[i] = 0;
    }
    
    printf("Training started... (%d epochs, %d samples)\n", 
           EPOCHS, TRAIN_SAMPLES);
    printf("[This will take a WHILE on 8088...]\n\n");
    
    /* === ЦИКЛ ОБУЧЕНИЯ === */
    for (epoch = 0; epoch < EPOCHS; epoch++) {
        total_error = 0;
        
        /* Проходим по всем образцам */
        for (sample = 0; sample < TRAIN_SAMPLES; sample++) {
            train_step(&net, dataset[sample].inputs, dataset[sample].targets);
            
            /* Считаем ошибку (MSE, упрощённо) */
            forward_pass(&net, dataset[sample].inputs);
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                fp_t diff = dataset[sample].targets[o] - output[o];
                total_error += FP_MUL(diff, diff);
            }
        }
        
        /* Каждые 100 эпох выводим прогресс */
        if (epoch % 100 == 0) {
            printf("Epoch %4d | Error: %d.%02d\n", 
                   epoch, FP_TO_INT(total_error), 
                   ((total_error & 0xFF) * 100) >> 8);
            
            /* Сохраняем чекпоинт (на случай зависания) */
            if (epoch % 500 == 0 && epoch > 0) {
                save_network(&net, "SKYNET.CKP");
                printf("  -> Checkpoint saved.\n");
            }
        }
    }
    
    /* === ФИНАЛЬНОЕ СОХРАНЕНИЕ === */
    printf("\nTraining complete!\n");
    printf("Saving weights to SKYNET.WGT...\n");
    
    if (save_network(&net, "SKYNET.WGT") == 0) {
        printf("SUCCESS! File size: %d bytes\n",
               sizeof(net));  /* ~400-500 bytes! */
        printf("\nNow run SKYNET.COM for inference agent.\n");
    } else {
        printf("ERROR saving weights!\n");
        return 1;
    }
    
    return 0;
}