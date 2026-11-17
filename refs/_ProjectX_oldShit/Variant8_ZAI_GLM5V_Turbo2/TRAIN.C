/*
 * ============================================================
 *  TRAIN.C - Neural Network Trainer v1.2
 *  Compatible: TCC (MinGW), GCC, Clang, Turbo C, Watcom
 *  
 *  FIX: LUT tables generated at runtime (no large static init)
 * ============================================================
 */

#include "skynet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 *  LOOKUP TABLES (генерируются при старте!)
 *  static = глобальные, заполняются один раз в init_luts()
 * ============================================================ */
static uint8_t SIGMOID_LUT[256];
static uint8_t SIGMOID_DERIV_LUT[256];

/* Флаг инициализации (чтобы не генерировать дважды) */
static int luts_initialized = 0;

/* ============================================================
 *  ГЕНЕРАЦИЯ LOOKUP TABLES
 *  Вызывается один раз в начале main() или при первом use
 * ============================================================ */
void init_luts(void) {
    int i;
    double x, val; /* используем double, но без exp() */
    
    if (luts_initialized) return;
    
    for (i = 0; i < 256; i++) {
        /*
         * Аппроксимация sigmoid БЕЗ exp():
         * Используем полином или piecewise-linear
         * 
         * Быстрая аппроксимация:
         * sigmoid(x) ≈ 0.5 + 0.5 * x / (1 + |x|)  для |x| < 4
         * 
         * Или ещё проще - предрассчитанные значения через формулу
         */
        x = ((double)i / 64.0) - 2.0;  /* -2.0 .. +2.0 */
        
        /* Полиномиальная аппроксимация сигмоида (достаточно точная!) */
        double abs_x = (x < 0) ? -x : x;
        val = 0.5 * x / (1.0 + abs_x) + 0.5;
        
        /* Ограничиваем [0..1] */
        if (val < 0.0) val = 0.0;
        if (val > 1.0) val = 1.0;
        
        SIGMOID_LUT[i] = (uint8_t)(val * 255.0);
        
        /* Производная: y*(1-y) */
        SIGMOID_DERIV_LUT[i] = (uint8_t)(val * (1.0 - val) * 255.0);
    }
    
    luts_initialized = 1;
}

/* ============================================================
 *  АКТИВАЦИИ (используют LUT)
 * ============================================================ */

/**
 * sigmoid_lut(x) - Сигмоида через lookup table
 * Вход: x в формате Q8.8 (примерно -512..+512)
 * Выход: Q8.8 значение 0..FP_ONE (0..256)
 */
static int16_t sigmoid_lut(int16_t x) {
    uint8_t index;
    int16_t result;
    int32_t shifted;
    
    /* Гарантируем что LUT инициализирован */
    if (!luts_initialized) init_luts();
    
    /* 
     * Преобразуем Q8.8 в индекс 0..255:
     * 1. Сдвигаем вправо на FP_SHIFT (берём целую часть со знаком)
     * 2. Добавляем смещение 128 чтобы получить 0..255
     * 
     * Диапазон входа: примерно -128..+127 (после сдвига)
     * После смещения: 0..255
     */
    shifted = (int32_t)x >> FP_SHIFT;  /* Целая часть */
    shifted += 128;                    /* Смещение */
    
    /* Ограничиваем диапазон [0..255] */
    if (shifted < 0) shifted = 0;
    if (shifted > 255) shifted = 255;
    
    index = (uint8_t)shifted;
    
    /* Получаем из таблицы (0..255) и масштабируем в Q8.8 */
    result = (int16_t)SIGMOID_LUT[index] << FP_SHIFT;
    
    return result;
}

/**
 * sigmoid_deriv_from_output(output) - Производная через LUT
 * Используется в backpropagation
 * Вход: output в Q8.8 (значение сигмоиды 0..FP_ONE)
 * Выход: производная в Q8.8
 */
static int16_t sigmoid_deriv_from_output(int16_t output) {
    uint8_t index;
    
    if (!luts_initialized) init_luts();
    
    /* Берём целую часть выхода (0..1 в Q8.8 = 0..256) */
    index = (uint8_t)((uint16_t)(output >> FP_SHIFT) & 0xFF);
    
    return (int16_t)SIGMOID_DERIV_LUT[index] << FP_SHIFT;
}

/* ============================================================
 *  ГЛОБАЛЬНЫЕ ДАННЫЕ НЕЙРОСЕТИ
 * ============================================================ */
static NeuralNetwork net;

/* Активации (переиспользуемые буферы) */
static int16_t hidden_act[NN_HIDDEN];
static int16_t output_act[NN_OUTPUTS];

/* Для backpropagation */
static int16_t output_deltas[NN_OUTPUTS];
static int16_t hidden_deltas[NN_HIDDEN];

/* ============================================================
 *  FORWARD PASS (Инференс / Прямой проход)
 * ============================================================ */
void forward(NeuralNetwork *nn, int16_t inputs[NN_INPUTS]) {
    int i, j;
    int32_t acc;
    
    /* ===== СКРЫТЫЙ СЛОЙ ===== */
    for (i = 0; i < NN_HIDDEN; i++) {
        acc = 0;
        
        /* Взвешенная сумма: Σ(input[j] * w_ih[j][i]) */
        for (j = 0; j < NN_INPUTS; j++) {
            acc += (int32_t)inputs[j] * nn->w_ih[j][i];
        }
        
        /* Добавляем bias (преобразуем в формат аккумулятора) */
        acc += (int32_t)nn->b_hidden[i] << FP_SHIFT;
        
        /* Применяем активацию (сигмоида через LUT) */
        hidden_act[i] = sigmoid_lut((int16_t)(acc >> FP_SHIFT));
    }
    
    /* ===== ВЫХОДНОЙ СЛОЙ ===== */
    for (i = 0; i < NN_OUTPUTS; i++) {
        acc = 0;
        
        for (j = 0; j < NN_HIDDEN; j++) {
            acc += (int32_t)hidden_act[j] * nn->w_ho[j][i];
        }
        
        acc += (int32_t)nn->b_output[i] << FP_SHIFT;
        output_act[i] = sigmoid_lut((int16_t)(acc >> FP_SHIFT));
    }
}

/* ============================================================
 *  BACKPROPAGATION (Обучение - один шаг SGD)
 * ============================================================ */
void train_step(NeuralNetwork *nn,
                int16_t inputs[NN_INPUTS],
                int16_t targets[NN_OUTPUTS]) {
    int i, j;
    int16_t error;
    int32_t acc;
    
    /* Шаг 1: Прямой проход (вычисляет hidden_act[], output_act[]) */
    forward(nn, inputs);
    
    /* 
     * Шаг 2: Ошибки выходного слоя
     * delta = (target - output) * sigmoid'(output)
     */
    for (i = 0; i < NN_OUTPUTS; i++) {
        error = targets[i] - output_act[i];
        output_deltas[i] = fp_mul(error, sigmoid_deriv_from_output(output_act[i]));
    }
    
    /*
     * Шаг 3: Ошибки скрытого слоя
     * delta_hidden[j] = Σ(delta_output[k] * w_ho[j][k]) * sigmoid'(hidden[j])
     */
    for (j = 0; j < NN_HIDDEN; j++) {
        acc = 0;
        for (i = 0; i < NN_OUTPUTS; i++) {
            acc += (int32_t)output_deltas[i] * nn->w_ho[j][i];
        }
        hidden_deltas[j] = fp_mul((int16_t)(acc >> FP_SHIFT),
                            sigmoid_deriv_from_output(hidden_act[j]));
    }
    
    /*
     * Шаг 4: Обновление весов hidden→output
     * w_ho[j][i] += LR * delta_output[i] * hidden_act[j]
     * b_output[i] += LR * delta_output[i]
     */
    for (j = 0; j < NN_HIDDEN; j++) {
        for (i = 0; i < NN_OUTPUTS; i++) {
            nn->w_ho[j][i] += fp_mul(LEARNING_RATE,
                             fp_mul(output_deltas[i], hidden_act[j]));
        }
    }
    for (i = 0; i < NN_OUTPUTS; i++) {
        nn->b_output[i] += fp_mul(LEARNING_RATE, output_deltas[i]);
    }
    
    /*
     * Шаг 5: Обновление весов input→hidden
     * w_ih[i][j] += LR * delta_hidden[j] * inputs[i]
     * b_hidden[j] += LR * delta_hidden[j]
     */
    for (i = 0; i < NN_INPUTS; i++) {
        for (j = 0; j < NN_HIDDEN; j++) {
            nn->w_ih[i][j] += fp_mul(LEARNING_RATE,
                             fp_mul(hidden_deltas[j], inputs[i]));
        }
    }
    for (j = 0; j < NN_HIDDEN; j++) {
        nn->b_hidden[j] += fp_mul(LEARNING_RATE, hidden_deltas[j]);
    }
}

/* ============================================================
 *  ИНИЦИАЛИЗАЦИЯ ВЕСОВ (Xavier-like)
 * ============================================================ */
void init_weights(NeuralNetwork *nn) {
    int i, j;
    uint32_t seed;
    
    seed = 12345;  /* Фиксированный seed для воспроизводимости */
    
    /* Простая LCG генерация случайных чисел */
    #define NEXT_RAND() ((seed = seed * 1103515245UL + 12345UL) >> 16)
    
    /* Инициализация весов small random values [-128..+127] */
    for (i = 0; i < NN_INPUTS; i++) {
        for (j = 0; j < NN_HIDDEN; j++) {
            nn->w_ih[i][j] = (int16_t)(NEXT_RAND() % 256) - 128;
        }
    }
    
    for (i = 0; i < NN_HIDDEN; i++) {
        for (j = 0; j < NN_OUTPUTS; j++) {
            nn->w_ho[i][j] = (int16_t)(NEXT_RAND() % 256) - 128;
        }
        /* Байасы = 0 */
        nn->b_hidden[i] = 0;
    }
    
    for (i = 0; i < NN_OUTPUTS; i++) {
        nn->b_output[i] = 0;
    }
}

/* ============================================================
 *  ВЫЧИСЛЕНИЕ MSE ОШИБКИ
 * ============================================================ */
int16_t calculate_mse(NeuralNetwork *nn,
                      TrainingSample *dataset,
                      int num_samples) {
    int s, i;
    int32_t total_error;
    int16_t diff;
    
    total_error = 0;
    
    for (s = 0; s < num_samples; s++) {
        forward(nn, dataset[s].inputs);
        
        for (i = 0; i < NN_OUTPUTS; i++) {
            diff = dataset[s].targets[i] - output_act[i];
            total_error += (int32_t)diff * diff;
        }
    }
    
    /* Возвращаем среднее в Q8.8 */
    return (int16_t)(total_error / (num_samples * NN_OUTPUTS));
}

/* ============================================================
 *  СОХРАНЕНИЕ/ЗАГРУЗКА ВЕСОВ
 * ============================================================ */
int save_weights(NeuralNetwork *nn, const char *filename) {
    FILE *f;
    WeightFileHeader header;
    
    f = fopen(filename, "wb");
    if (!f) return -1;
    
    /* Заголовок */
    header.magic = SKYNET_MAGIC;
    header.version = SKYNET_VERSION;
    header.inputs = NN_INPUTS;
    header.hidden = NN_HIDDEN;
    header.outputs = NN_OUTPUTS;
    header.reserved = 0;
    
    fwrite(&header, sizeof(header), 1, f);
    
    /* Веса (raw binary) */
    fwrite(nn->w_ih, sizeof(int16_t), NN_INPUTS * NN_HIDDEN, f);
    fwrite(nn->w_ho, sizeof(int16_t), NN_HIDDEN * NN_OUTPUTS, f);
    fwrite(nn->b_hidden, sizeof(int16_t), NN_HIDDEN, f);
    fwrite(nn->b_output, sizeof(int16_t), NN_OUTPUTS, f);
    
    fclose(f);
    return 0;
}

int load_weights(NeuralNetwork *nn, const char *filename) {
    FILE *f;
    WeightFileHeader header;
    
    f = fopen(filename, "rb");
    if (!f) return -1;
    
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return -2;
    }
    
    if (header.magic != SKYNET_MAGIC) {
        fclose(f);
        return -3;  /* Неверный формат */
    }
    
    fread(nn->w_ih, sizeof(int16_t), NN_INPUTS * NN_HIDDEN, f);
    fread(nn->w_ho, sizeof(int16_t), NN_HIDDEN * NN_OUTPUTS, f);
    fread(nn->b_hidden, sizeof(int16_t), NN_HIDDEN, f);
    fread(nn->b_output, sizeof(int16_t), NN_OUTPUTS, f);
    
    fclose(f);
    return 0;
}

/* ============================================================
 *  ДАТАСЕТ ДЛЯ ОБУЧЕНИЯ
 *  Задача: Агент в 2D мире принимает решения
 * ============================================================ */
#define NUM_TRAINING_SAMPLES 20

static TrainingSample training_data[NUM_TRAINING_SAMPLES] = {
    /* 0: Цель вперёд → идти вперёд */
    {{INT_TO_FP(90), INT_TO_FP(0), INT_TO_FP(5), INT_TO_FP(100),
      INT_TO_FP(200), INT_TO_FP(0), INT_TO_FP(50), INT_TO_FP(0)},
     {FP_ONE, 0, 0, 0}},
     
    /* 1: Цель слева → поворот влево */
    {{INT_TO_FP(70), INT_TO_FP(-45), INT_TO_FP(3), INT_TO_FP(95),
      INT_TO_FP(150), INT_TO_FP(30), INT_TO_FP(50), INT_TO_FP(0)},
     {0, 0, FP_ONE, 0}},
     
    /* 2: Цель справа → поворот вправо */
    {{INT_TO_FP(70), INT_TO_FP(45), INT_TO_FP(3), INT_TO_FP(95),
      INT_TO_FP(150), INT_TO_FP(-30), INT_TO_FP(50), INT_TO_FP(0)},
     {0, 0, 0, FP_ONE}},
     
    /* 3: Препятствие впереди → назад */
    {{INT_TO_FP(50), INT_TO_FP(0), INT_TO_FP(2), INT_TO_FP(80),
      INT_TO_FP(10), INT_TO_FP(0), INT_TO_FP(50), INT_TO_FP(0)},
     {0, FP_ONE, 0, 0}},
     
    /* 4: Низкая энергия → отдых */
    {{INT_TO_FP(30), INT_TO_FP(10), INT_TO_FP(1), INT_TO_FP(15),
      INT_TO_FP(100), INT_TO_FP(0), INT_TO_FP(80), INT_TO_FP(0)},
     {0, FP_ONE, 0, 0}},
     
    /* 5: Почти прямо, немного вправо */
    {{INT_TO_FP(85), INT_TO_FP(10), INT_TO_FP(4), INT_TO_FP(90),
      INT_TO_FP(180), INT_TO_FP(15), INT_TO_FP(30), INT_TO_FP(1)},
     {FP_ONE, 0, 0, 0}},
     
    /* 6: Цель левее */
    {{INT_TO_FP(60), INT_TO_FP(-30), INT_TO_FP(6), INT_TO_FP(85),
      INT_TO_FP(120), INT_TO_FP(-20), INT_TO_FP(70), INT_TO_FP(2)},
     {0, 0, FP_ONE, 0}},
     
    /* 7: Цель правее */
    {{INT_TO_FP(40), INT_TO_FP(60), INT_TO_FP(2), INT_TO_FP(70),
      INT_TO_FP(80), INT_TO_FP(45), INT_TO_FP(90), INT_TO_FP(3)},
     {0, 0, 0, FP_ONE}},
     
    /* 8: Немного влево */
    {{INT_TO_FP(50), INT_TO_FP(-15), INT_TO_FP(3), INT_TO_FP(88),
      INT_TO_FP(90), INT_TO_FP(10), INT_TO_FP(40), INT_TO_FP(5)},
     {FP_ONE, 0, 0, 0}},
     
    /* 9: Немного вправо */
    {{INT_TO_FP(55), INT_TO_FP(20), INT_TO_FP(4), INT_TO_FP(92),
      INT_TO_FP(110), INT_TO_FP(-5), INT_TO_FP(60), INT_TO_FP(7)},
     {0, 0, 0, FP_ONE}},
     
    /* 10: Прямо близко */
    {{INT_TO_FP(95), INT_TO_FP(5), INT_TO_FP(7), INT_TO_FP(98),
      INT_TO_FP(220), INT_TO_FP(-5), INT_TO_FP(20), INT_TO_FP(0)},
     {FP_ONE, 0, 0, 0}},
     
    /* 11: Сильно влево */
    {{INT_TO_FP(65), INT_TO_FP(-60), INT_TO_FP(2), INT_TO_FP(80),
      INT_TO_FP(130), INT_TO_FP(45), INT_TO_FP(55), INT_TO_FP(1)},
     {0, 0, FP_ONE, 0}},
     
    /* 12: Сильно вправо */
    {{INT_TO_FP(65), INT_TO_FP(60), INT_TO_FP(2), INT_TO_FP(82),
      INT_TO_FP(130), INT_TO_FP(-45), INT_TO_FP(58), INT_TO_FP(2)},
     {0, 0, 0, FP_ONE}},
     
    /* 13: Критически низкая энергия */
    {{INT_TO_FP(20), INT_TO_FP(5), INT_TO_FP(0), INT_TO_FP(10),
      INT_TO_FP(5), INT_TO_FP(10), INT_TO_FP(95), INT_TO_FP(4)},
     {0, FP_ONE, 0, 0}},
     
    /* 14: Идеально прямо */
    {{INT_TO_FP(99), INT_TO_FP(-10), INT_TO_FP(8), INT_TO_FP(99),
      INT_TO_FP(250), INT_TO_FP(8), INT_TO_FP(10), INT_TO_FP(0)},
     {FP_ONE, 0, 0, 0}},
     
    /* 15: Крайне лево */
    {{INT_TO_FP(45), INT_TO_FP(-80), INT_TO_FP(1), INT_TO_FP(75),
      INT_TO_FP(95), INT_TO_FP(70), INT_TO_FP(65), INT_TO_FP(3)},
     {0, 0, FP_ONE, 0}},
     
    /* 16: Крайне право */
    {{INT_TO_FP(45), INT_TO_FP(80), INT_TO_FP(1), INT_TO_FP(73),
      INT_TO_FP(95), INT_TO_FP(-70), INT_TO_FP(68), INT_TO_FP(2)},
     {0, 0, 0, FP_ONE}},
     
    /* 17: Быстро вперёд */
    {{INT_TO_FP(75), INT_TO_FP(0), INT_TO_FP(9), INT_TO_FP(96),
      INT_TO_FP(190), INT_TO_FP(0), INT_TO_FP(35), INT_TO_FP(1)},
     {FP_ONE, 0, 0, 0}},
     
    /* 18: Средне-лево */
    {{INT_TO_FP(35), INT_TO_FP(-25), INT_TO_FP(4), INT_TO_FP(87),
      INT_TO_FP(70), INT_TO_FP(18), INT_TO_FP(75), INT_TO_FP(6)},
     {0, 0, FP_ONE, 0}},
     
    /* 19: Средне-право */
    {{INT_TO_FP(35), INT_TO_FP(25), INT_TO_FP(4), INT_TO_FP(86),
      INT_TO_FP(72), INT_TO_FP(-18), INT_TO_FP(73), INT_TO_FP(5)},
     {0, 0, 0, FP_ONE}}
};

/* ============================================================
 *  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ВЫВОДА
 * ============================================================ */

/**
 * Вывод Q8.8 числа как decimal
 * Пример: 0x0100 (256) → "1.00"
 *          0xFF00 (-256) → "-1.00"
 */
void print_fp(int16_t value) {
    int16_t whole;
    uint16_t frac;
    unsigned int ufrac;
    
    whole = value >> FP_SHIFT;       /* Целая часть */
    
    /* Дробная часть: берем младшие 8 бит, масштабируем 0..99 */
    ufrac = (uint16_t)(value & 0xFF);
    frac = (ufrac * 100) >> FP_SHIFT;
    
    /* Обработка отрицательного нуля */
    if (value < 0 && whole == 0) {
        printf("-0.%02u", frac);
    } else {
        printf("%d.%02u", (int)whole, frac);
    }
}

/**
 * Вывод строки прогресса обучения
 */
void print_progress(int epoch, int16_t error) {
    printf("Epoch %4d | Loss: ", epoch);
    print_fp(error);
    printf("\n");
}

/* ============================================================
 *  MAIN - Главный цикл обучения
 * ============================================================ */
int main(void) {
    int epoch, sample;
    int16_t current_error;
    int best_epoch = 0;
    int16_t best_error;
    
    /* === Инициализация LUT (ВАЖНО!) === */
    init_luts();
    
    best_error = INT_TO_FP(999);  /* Начальное "худшее" значение */
    
    /* === БАННЕР === */
    printf("============================================\n");
    printf("   SKYNET Neural Network Trainer v1.2\n");
    printf("   Pure C | Fixed-point Q8.8 | LUT-based\n");
    printf("============================================\n\n");
    
    printf("Architecture: %d -> %d -> %d\n", 
           NN_INPUTS, NN_HIDDEN, NN_OUTPUTS);
    printf("Parameters:  %d weights\n", 
           NN_INPUTS*NN_HIDDEN + NN_HIDDEN*NN_OUTPUTS + NN_HIDDEN + NN_OUTPUTS);
    printf("Training set: %d samples\n", NUM_TRAINING_SAMPLES);
    printf("Max epochs:   %d\n\n", EPOCHS);
    
    /* === Шаг 1: Инициализация весов === */
    printf("[1/3] Initializing neural network...\n");
    init_weights(&net);
    printf("       Weights initialized with random values.\n");
    
    /* === Шаг 2: Попытка загрузить существующие веса === */
    printf("[2/3] Checking for existing model...\n");
    if (load_weights(&net, "SKYNET.WGT") == 0) {
        printf("       Found SKYNET.WGT - continuing training.\n");
    } else {
        printf("       No existing model found.\n");
        printf("       Starting from scratch.\n");
    }
    
    /* === Шаг 3: ОБУЧЕНИЕ === */
    printf("\n[3/3] Training started...\n");
    printf("--------------------------------------------\n");
    printf(" Epoch  |  Loss (MSE)\n");
    printf("--------+-------------\n");
    
    for (epoch = 0; epoch < EPOCHS; epoch++) {
        /* Проходим по всем обучающим образцам */
        for (sample = 0; sample < NUM_TRAINING_SAMPLES; sample++) {
            train_step(&net,
                       training_data[sample].inputs,
                       training_data[sample].targets);
        }
        
        /* Каждые N эпох выводим прогресс и сохраняем чекпоинт */
        if (epoch % 200 == 0 || epoch == EPOCHS - 1) {
            current_error = calculate_mse(&net, training_data,
                                          NUM_TRAINING_SAMPLES);
            print_progress(epoch, current_error);
            
            /* Сохраняем лучшую модель */
            if (current_error < best_error) {
                best_error = current_error;
                best_epoch = epoch;
                save_weights(&net, "SKYNET.BEST");
            }
        }
        
        /* Чекпоинт каждые 500 эпох */
        if (epoch > 0 && epoch % 500 == 0) {
            save_weights(&net, "SKYNET.CKP");
            printf("       [Checkpoint saved]\n");
        }
    }
    
    printf("--------------------------------------------\n\n");
    
    /* === ФИНАЛЬНОЕ СОХРАНЕНИЕ === */
    printf("Saving final model to SKYNET.WGT... ");
    
    if (save_weights(&net, "SKYNET.WGT") == 0) {
        printf("OK\n");
        printf("  File size: ~%u bytes\n",
               (unsigned)(sizeof(NeuralNetwork) + sizeof(WeightFileHeader)));
        printf("  Best loss:  ");
        print_fp(best_error);
        printf(" (at epoch %d)\n", best_epoch);
    } else {
        printf("FAILED!\n");
        return 1;
    }
    
    /* === ТЕСТ: Быстрая проверка на первом образце === */
    printf("\n=== Quick Inference Test ===\n");
    printf("Input sample 0: [dist=90, angle=0, speed=5, energy=100, ...]\n");
    printf("Expected action: FORWARD\n\n");
    
    forward(&net, training_data[0].inputs);
    
    printf("Raw outputs: [");
    {
        int i;
        for (i = 0; i < NN_OUTPUTS; i++) {
            print_fp(output_act[i]);
            if (i < NN_OUTPUTS - 1) printf(", ");
        }
    }
    printf("]\n");
    
    /* Argmax - какое действие выбрано? */
    {
        int pred = 0;
        int i;
        const char *action_names[] = {"FORWARD", "BACKWARD", "LEFT", "RIGHT"};
        
        for (i = 1; i < NN_OUTPUTS; i++) {
            if (output_act[i] > output_act[pred]) {
                pred = i;
            }
        }
        
        printf("Predicted: %s (%d)\n", action_names[pred], pred);
    }
    
    printf("\n============================================\n");
    printf(" Training complete!\n");
    printf(" Run 'skynet.com' to deploy AI agent.\n");
    printf("============================================\n");
    
    return 0;
}