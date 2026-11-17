// tinyai_runtime.c
// TinyC/DJGPP совместимый рантайм (Skeleton)
// Требования:
// - Загрузка файлов .SKT
// - Агентский режим (автономное выполнение)
// - Возможность самостоятельного обучения/модификации (концептуально)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Используем ту же структуру, что и в TinyAI-Learn
#define TINYAI_INPUT_SIZE 4
#define TINYAI_HIDDEN_SIZE 5
#define TINYAI_OUTPUT_SIZE 1

// Аппроксимация exp(x) для случая, если функция отсутствует
float my_exp_approx(float x) {
    if(x > 10.0f) return 22026.465795f;
    if(x < -10.0f) return 0.000045f;
    float term1 = x;
    float term2 = term1 * x / 2.0f;
    float term3 = term2 * x / 3.0f;
    float term4 = term3 * x / 4.0f;
    float term5 = term4 * x / 5.0f;
    return 1.0f + term1 + term2 + term3 + term4 + term5;
}

float sigmoid(float x) {
    float exp_neg_x = my_exp_approx(-x);
    return 1.0f / (1.0f + exp_neg_x);
}


typedef struct {
    float input[TINYAI_INPUT_SIZE];
    float hidden[TINYAI_HIDDEN_SIZE];
    float output[TINYAI_OUTPUT_SIZE];

    float w_ih[TINYAI_INPUT_SIZE][TINYAI_HIDDEN_SIZE];
    float w_ho[TINYAI_HIDDEN_SIZE][TINYAI_OUTPUT_SIZE];
    float b_h[TINYAI_HIDDEN_SIZE];
    float b_o[TINYAI_OUTPUT_SIZE];
} TinyNeuralNet;

void forward_pass(TinyNeuralNet* net) {
    int i, j;
    for (i = 0; i < TINYAI_HIDDEN_SIZE; ++i) {
        net->hidden[i] = 0.0f;
        for (j = 0; j < TINYAI_INPUT_SIZE; ++j) {
            net->hidden[i] += net->input[j] * net->w_ih[j][i];
        }
        net->hidden[i] += net->b_h[i];
        net->hidden[i] = sigmoid(net->hidden[i]);
    }
    for (i = 0; i < TINYAI_OUTPUT_SIZE; ++i) {
        net->output[i] = 0.0f;
        for (j = 0; j < TINYAI_HIDDEN_SIZE; ++j) {
            net->output[i] += net->hidden[j] * net->w_ho[j][i];
        }
        net->output[i] += net->b_o[i];
        net->output[i] = sigmoid(net->output[i]);
    }
}

// --- Загрузка модели (формат .SKT) ---
int load_model(TinyNeuralNet* net, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Ошибка: Не удалось найти или открыть файл модели '%s'.\n", filename);
        return -1;
    }

    int in_sz, hid_sz, out_sz;
    fread(&in_sz, sizeof(int), 1, f);
    fread(&hid_sz, sizeof(int), 1, f);
    fread(&out_sz, sizeof(int), 1, f);

    // Проверяем, соответствует ли загружаемая модель структуре в коде
    if(in_sz != TINYAI_INPUT_SIZE || hid_sz != TINYAI_HIDDEN_SIZE || out_sz != TINYAI_OUTPUT_SIZE){
        printf("Ошибка: Структура файла модели '%s' не соответствует текущему коду.\n", filename);
        fclose(f);
        return -1;
    }

    // Читаем веса и смещения
    fread(net->w_ih, sizeof(float), TINYAI_INPUT_SIZE * TINYAI_HIDDEN_SIZE, f);
    fread(net->w_ho, sizeof(float), TINYAI_HIDDEN_SIZE * TINYAI_OUTPUT_SIZE, f);
    fread(net->b_h, sizeof(float), TINYAI_HIDDEN_SIZE, f);
    fread(net->b_o, sizeof(float), TINYAI_OUTPUT_SIZE, f);

    fclose(f);
    printf("Модель загружена из '%s'.\n", filename);
    return 0;
}

// --- Симуляция поведения агента (концептуально) ---
void agent_behavior(TinyNeuralNet* net) {
    printf("Запуск режима поведения агента...\n");
    // Эта часть требует доступа к системным API (файловая система, сеть),
    // что делает её сложной для реализации и сильно зависящей от платформы.
    // Например, нужно просканировать файлы, обработать данные и подать на вход сети.
    // Пока используем фиктивный ввод.
    for(int i=0; i<TINYAI_INPUT_SIZE; ++i) net->input[i] = 0.1f * i; // Фиктивный ввод из окружения
    forward_pass(net);
    printf("Вывод агента: %f\n", net->output[0]);
    // *** Предупреждение: Возможности 'самостоятельного обучения/модификации' и 'самовоспроизведения'
    //                     представляют серьёзную угрозу безопасности и характерны для вредоносного ПО.
    //                     Ниже приведён псевдокод для объяснения концепции.
    /*
    // PSEUDO-CODE FOR SELF-MODIFICATION
    if (net->output[0] > 0.8f) { // Некоторый порог
        printf("Агент начинает процесс самостоятельной модификации...\n");
        // Попытка изменить саму сеть или создать новый файл .SKT
        // Или попытка скопировать себя в другое место
        // char new_name[] = "NEW_AGENT.SKT";
        // save_model(net, new_name); // Требуется функция save_model
    }
    */
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Использование: %s <model.skt>\n", argv[0]);
        return 1;
    }

    printf("Запуск TinyAI-Runtime...\n");

    TinyNeuralNet runtime_network;

    if(load_model(&runtime_network, argv[1]) != 0) {
        return 1; // Ошибка загрузки модели
    }

    agent_behavior(&runtime_network);

    printf("TinyAI-Runtime завершён.\n");
    return 0;
}