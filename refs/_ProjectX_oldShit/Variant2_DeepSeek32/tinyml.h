/* TinyML Common Header - No math.h dependency */
#ifndef TINYML_H
#define TINYML_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Нейросеть: многослойный перцептрон */
typedef struct {
    int layers;
    int* neurons;
    float** weights;
    float** biases;
    unsigned int timestamp;
} NeuralNet;

/* Объявления функций */
float* forward(NeuralNet* net, float* input);
NeuralNet* create_network(int layers, int* sizes);
void save_model(NeuralNet* net, const char* filename);
NeuralNet* load_model(const char* filename);

/* Наша собственная реализация expf */
static float my_expf(float x) {
    /* Аппроксимация экспоненты через ряд Тейлора */
    if (x < -10) return 0.0000454f;
    if (x > 10) return 22026.46f;
    
    float result = 1.0f;
    float term = 1.0f;
    
    for (int i = 1; i < 10; i++) {
        term *= x / i;
        result += term;
    }
    return result;
}

/* Сигмоида через нашу expf */
static float sigmoid(float x) {
    if (x > 0) {
        return 1.0f / (1.0f + my_expf(-x));
    } else {
        float exp_x = my_expf(x);
        return exp_x / (1.0f + exp_x);
    }
}

/* ReLU - проще простого */
static float relu(float x) {
    return x > 0 ? x : 0;
}

/* Генерация имени файла из времени */
static void generate_filename(char* buf) {
    time_t t = time(NULL);
    sprintf(buf, "%08X.SKT", (unsigned int)t);
}

#endif