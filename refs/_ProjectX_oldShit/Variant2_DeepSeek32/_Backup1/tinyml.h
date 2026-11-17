/* TinyML Common Header */
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
    float* activations;
} NeuralNet;

/* Формат файла SKT */
#pragma pack(push, 1)
typedef struct {
    char magic[4];          /* "SKT" */
    unsigned int timestamp; /* UNIX время */
    unsigned short layers;  /* Количество слоев */
    unsigned short* sizes;  /* Размеры слоев */
    float* data;            /* Веса и смещения */
} SKTFile;
#pragma pack(pop)

/* Базовые функции активации */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float relu(float x) {
    return x > 0 ? x : 0;
}

/* Генерация имени файла из времени */
void generate_filename(char* buf) {
    time_t t = time(NULL);
    sprintf(buf, "%08X.SKT", (unsigned int)t);
}

#endif