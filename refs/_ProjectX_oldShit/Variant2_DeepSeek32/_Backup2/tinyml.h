/* TinyML Common Header - Simplified for TCC */
#ifndef TINYML_H
#define TINYML_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Нейросеть: многослойный перцептрон */
typedef struct {
    int layers;
    int* neurons;
    float** weights;
    float** biases;
} NeuralNet;

/* Базовые функции активации */
static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float relu(float x) {
    return x > 0 ? x : 0;
}

/* Генерация имени файла из времени */
static void generate_filename(char* buf) {
    time_t t = time(NULL);
    sprintf(buf, "%08X.SKT", (unsigned int)t);
}

#endif