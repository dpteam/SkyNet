/*
 * ============================================================
 *  SKYNET.C - AI Agent v1.3 (Windows/TCC Compatible)
 *  
 *  Works with: TCC(MinGW), GCC, Clang, MSVC
 *  NOTE: For real TSR functionality, compile under DOS!
 *  This is a simulation/test version for Windows.
 * ============================================================
 */

#include "skynet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Для Windows-specific функций (если нужны) */
#ifdef _WIN32
    #include <windows.h>
#endif

/* ============================================================
 *  КОНСТАНТЫ АГЕНТА
 * ============================================================ */
#define THINK_INTERVAL   2000    /* ms между "мышлением" */
#define ACTION_COOLDOWN  3000    /* ms cooldown */

/* ============================================================
 *  LOOKUP TABLES (генерируем как в train.c)
 * ============================================================ */
static uint8_t SIGMOID_LUT[256];
static int luts_initialized = 0;

void init_luts(void) {
    int i;
    double x, val;
    
    if (luts_initialized) return;
    
    for (i = 0; i < 256; i++) {
        x = ((double)i / 64.0) - 2.0;
        /* Аппроксимация без exp() для совместимости */
        double abs_x = (x < 0) ? -x : x;
        val = 0.5 * x / (1.0 + abs_x) + 0.5;
        if (val < 0.0) val = 0.0;
        if (val > 1.0) val = 1.0;
        SIGMOID_LUT[i] = (uint8_t)(val * 255.0);
    }
    luts_initialized = 1;
}

/* Сигмоида через LUT */
static int16_t sigmoid_lut_fp(int16_t x) {
    int32_t shifted;
    uint8_t idx;
    
    if (!luts_initialized) init_luts();
    
    shifted = ((int32_t)x >> FP_SHIFT) + 128;
    if (shifted < 0) shifted = 0;
    if (shifted > 255) shifted = 255;
    idx = (uint8_t)shifted;
    
    return (int16_t)SIGMOID_LUT[idx] << FP_SHIFT;
}

/* ============================================================
 *  ГЛОБАЛЬНЫЕ ДАННЫЕ
 * ============================================================ */
static NeuralNetwork brain;

/* Сенсоры и активации */
static int16_t sensor_values[NN_INPUTS];
static int16_t hidden_layer[NN_HIDDEN];
static int16_t decision_outputs[NN_OUTPUTS];

/* Статистика */
static uint32_t total_decisions = 0;
static uint32_t action_counts[NN_OUTPUTS] = {0, 0, 0, 0};

/* Состояние */
static int agent_running = 0;

/* ============================================================
 *  НЕЙРОСЕТЬ: FORWARD PASS (инференс)
 * ============================================================ */
static void run_inference(void) {
    int i, j;
    int32_t acc;
    
    /* Скрытый слой */
    for (i = 0; i < NN_HIDDEN; i++) {
        acc = (int32_t)brain.b_hidden[i] << FP_SHIFT;
        for (j = 0; j < NN_INPUTS; j++) {
            acc += (int32_t)sensor_values[j] * brain.w_ih[j][i];
        }
        hidden_layer[i] = sigmoid_lut_fp((int16_t)(acc >> FP_SHIFT));
    }
    
    /* Выходной слой */
    for (i = 0; i < NN_OUTPUTS; i++) {
        acc = (int32_t)brain.b_output[i] << FP_SHIFT;
        for (j = 0; j < NN_HIDDEN; j++) {
            acc += (int32_t)hidden_layer[j] * brain.w_ho[j][i];
        }
        decision_outputs[i] = sigmoid_lut_fp((int16_t)(acc >> FP_SHIFT));
    }
}

/* ============================================================
 *  СЕНСОРЫ: Чтение состояния системы (Windows версия)
 * ============================================================ */
static void read_sensors(void) {
    time_t now;
    struct tm *tm_info;
    
    /* Сенсор 0: Время (секунды дня) */
    time(&now);
    tm_info = localtime(&now);
    sensor_values[0] = INT_TO_FP(tm_info->tm_hour * 60 + tm_info->tm_min);
    
    /* Сенсор 1: Секунды */
    sensor_values[1] = INT_TO_FP(tm_info->tm_sec);
    
    /* Сенсор 2: День месяца */
    sensor_values[2] = INT_TO_FP(tm_info->tm_mday);
    
    /* Сенсоры 3-7: Псевдослучайные/системные данные */
    sensor_values[3] = INT_TO_FP(clock() % 100);
    sensor_values[4] = INT_TO_FP(total_decisions % 256);
    sensor_values[5] = INT_TO_FP((now >> 4) & 0xFF);
    sensor_values[6] = INT_TO_FP(tm_info->tm_wday * 10);
    sensor_values[7] = INT_TO_FP(rand() % 128);
}

/* ============================================================
 *  АКТУАТОРЫ: Выполнение действия
 * ============================================================ */
static const char *ACTION_NAMES[] = {"FORWARD", "BACKWARD", "LEFT", "RIGHT"};

static void execute_action(int action_index) {
    total_decisions++;
    if (action_index >= 0 && action_index < NN_OUTPUTS) {
        action_counts[action_index]++;
    }
    
    switch (action_index) {
        case 0: /* FORWARD */
            printf("  [ACTION] >>> FORWARD! (%lu total)\n", 
                   (unsigned long)total_decisions);
            break;
            
        case 1: /* BACKWARD */
            printf("  [ACTION] <<< BACKWARD! (%lu total)\n",
                   (unsigned long)total_decisions);
            break;
            
        case 2: /* LEFT */
            printf("  [ACTION] <<< LEFT TURN! (%lu total)\n",
                   (unsigned long)total_decisions);
            break;
            
        case 3: /* RIGHT */
            printf("  [ACTION] >>> RIGHT TURN! (%lu total)\n",
                   (unsigned long)total_decisions);
            break;
            
        default:
            printf("  [ACTION] ??? UNKNOWN (%d)\n", action_index);
            break;
    }
}

/* ============================================================
 *  ПРИНЯТИЕ РЕШЕНИЯ
 * ============================================================ */
static void think_and_act(void) {
    int i, best_action;
    int16_t max_output;
    
    /* 1. Читаем сенсоры */
    read_sensors();
    
    /* 2. Нейроинференс */
    run_inference();
    
    /* 3. Argmax - выбираем действие с максимальным выходом */
    best_action = 0;
    max_output = decision_outputs[0];
    for (i = 1; i < NN_OUTPUTS; i++) {
        if (decision_outputs[i] > max_output) {
            max_output = decision_outputs[i];
            best_action = i;
        }
    }
    
    /* 4. Выполняем действие */
    execute_action(best_action);
}

/* ============================================================
 *  ВЫВОД СТАТУСА
 * ============================================================ */
static void show_status(void) {
    int i;
    
    printf("\n");
    printf("╔══════════════════════════════════════╗\n");
    printf("║     SKYNET AGENT STATUS              ║\n");
    printf("╠══════════════════════════════════════╣\n");
    printf("║ Total decisions: %18lu ║\n", (unsigned long)total_decisions);
    printf("╠══════════════════════════════════════╣\n");
    printf("║ Action distribution:                ║\n");
    printf("║   [0] FORWARD : %6lu              ║\n", (unsigned long)action_counts[0]);
    printf("║   [1] BACKWARD: %6lu              ║\n", (unsigned long)action_counts[1]);
    printf("║   [2] LEFT    : %6lu              ║\n", (unsigned long)action_counts[2]);
    printf("║   [3] RIGHT   : %6lu              ║\n", (unsigned long)action_counts[3]);
    printf("╠══════════════════════════════════════╣\n");
    printf("║ Neural outputs:                     ║\n");
    printf("║   [");
    for (i = 0; i < NN_OUTPUTS; i++) {
        int16_t val = decision_outputs[i];
        int whole = val >> FP_SHIFT;
        unsigned frac = ((unsigned)(val & 0xFF) * 100) >> FP_SHIFT;
        if (val < 0 && whole == 0) printf("-");
        printf("%d.%02d", whole, frac);
        if (i < NN_OUTPUTS - 1) printf("|");
    }
    printf("]\n");
    printf("╚══════════════════════════════════════╝\n");
}

/* ============================================================
 *  ЗАГРУЗКА ВЕСОВ
 * ============================================================ */
static int load_brain(const char *filename) {
    FILE *f;
    WeightFileHeader header;
    
    f = fopen(filename, "rb");
    if (!f) {
        printf("ERROR: Cannot open '%s'\n", filename);
        return -1;
    }
    
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        printf("ERROR: Invalid file format\n");
        return -2;
    }
    
    if (header.magic != SKYNET_MAGIC) {
        fclose(f);
        printf("ERROR: Not a SKYNET weight file\n");
        return -3;
    }
    
    printf("Loading network: %d-%d-%d\n",
           header.inputs, header.hidden, header.outputs);
    
    fread(brain.w_ih, sizeof(int16_t), NN_INPUTS * NN_HIDDEN, f);
    fread(brain.w_ho, sizeof(int16_t), NN_HIDDEN * NN_OUTPUTS, f);
    fread(brain.b_hidden, sizeof(int16_t), NN_HIDDEN, f);
    fread(brain.b_output, sizeof(int16_t), NN_OUTPUTS, f);
    
    fclose(f);
    return 0;
}

/* ============================================================
 *  MAIN - Точка входа агента (Windows версия)
 * ============================================================ */
int main(int argc, char *argv[]) {
    int i;
    int num_cycles = 10;  /* Сколько раз "подумать" по умолчанию */
    int show_status_flag = 0;
    char input[256];
    
    /* === БАННЕР === */
    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║     SKYNET AI AGENT v1.3 (Pure C)       ║\n");
    printf("║     Windows Simulation Mode             ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");
    
    /* === Парсинг аргументов === */
    if (argc > 1) {
        if (strcmp(argv[1], "/?") == 0 || strcmp(argv[1], "-h") == 0 ||
            strcmp(argv[1], "--help") == 0) {
            printf("Usage: skynet.exe [options]\n\n");
            printf("Options:\n");
            printf("  /N <num>   Number of thinking cycles (default: 10)\n");
            printf("  /S         Show status after each cycle\n");
            printf("  /? , -h    Show this help\n\n");
            printf("Example:\n");
            printf("  skynet.exe /N 50 /S\n");
            return 0;
        }
        
        for (i = 1; i < argc; i++) {
            if (strcmp(argv[i], "/N") == 0 || strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    num_cycles = atoi(argv[++i]);
                    if (num_cycles < 1) num_cycles = 1;
                    if (num_cycles > 10000) num_cycles = 10000;
                }
            }
            else if (strcmp(argv[i], "/S") == 0 || strcmp(argv[i], "-s") == 0) {
                show_status_flag = 1;
            }
        }
    }
    
    /* === Инициализация LUT === */
    init_luts();
    srand((unsigned)time(NULL));
    
    /* === Загрузка весов === */
    printf("[*] Loading neural weights from SKYNET.WGT...\n");
    
    if (load_brain("SKYNET.WGT") != 0) {
        printf("\n[*] No weight file found!\n");
        printf("    Run TRAIN.EXE first to train the network.\n");
        printf("    Continuing with random/uninitialized weights...\n\n");
        /* Инициализируем нулями чтобы не было мусора */
        memset(&brain, 0, sizeof(brain));
    } else {
        printf("[*] Weights loaded successfully!\n\n");
    }
    
    /* === ЗАПУСК АГЕНТА === */
    printf("========================================\n");
    printf(" Starting AI Agent Simulation\n");
    printf(" Cycles: %d | Verbose: %s\n", 
           num_cycles, show_status_flag ? "ON" : "OFF");
    printf("========================================\n\n");
    
    agent_running = 1;
    
    for (i = 0; i < num_cycles && agent_running; i++) {
        printf("--- Cycle %d/%d ---\n", i + 1, num_cycles);
        
        /* "Думаем" и действуем */
        think_and_act();
        
        /* Показываем статус если включено */
        if (show_status_flag) {
            show_status();
        }
        
        /* Небольшая задержка (для наглядности) */
        #ifdef _WIN32
            Sleep(500);  /* 500ms */
        #else
            /* Кроссплатформенная задержка */
            {
                clock_t end = clock() + (CLOCKS_PER_SEC / 2);
                while (clock() < end);
            }
        #endif
    }
    
    /* === ФИНАЛЬНЫЙ ОТЧЁТ === */
    printf("\n\n========================================\n");
    printf(" SIMULATION COMPLETE\n");
    printf("========================================\n");
    show_status();
    
    printf("\nAgent made %lu decisions.\n", (unsigned long)total_decisions);
    printf("\nFor real TSR functionality (DOS resident program),\n");
    printf("compile this file with Turbo C++ 2.01 under DOS.\n");
    printf("========================================\n");
    
    return 0;
}