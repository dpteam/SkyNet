/* TinyML Common Functions */
#include "tinyml.h"

/* Прямое распространение - общая функция */
float* forward(NeuralNet* net, float* input) {
    float* current = (float*)malloc(net->neurons[0] * sizeof(float));
    memcpy(current, input, net->neurons[0] * sizeof(float));
    
    for(int l = 0; l < net->layers-1; l++) {
        float* next = (float*)malloc(net->neurons[l+1] * sizeof(float));
        
        for(int i = 0; i < net->neurons[l+1]; i++) {
            float sum = net->biases[l][i];
            
            for(int j = 0; j < net->neurons[l]; j++) {
                sum += current[j] * net->weights[l][j * net->neurons[l+1] + i];
            }
            
            next[i] = sigmoid(sum);
        }
        
        free(current);
        current = next;
    }
    
    return current;
}

/* Инициализация сети - общая функция */
NeuralNet* create_network(int layers, int* sizes) {
    NeuralNet* net = (NeuralNet*)malloc(sizeof(NeuralNet));
    net->layers = layers;
    net->neurons = (int*)malloc(layers * sizeof(int));
    net->weights = (float**)malloc((layers-1) * sizeof(float*));
    net->biases = (float**)malloc((layers-1) * sizeof(float*));
    net->timestamp = (unsigned int)time(NULL);
    
    memcpy(net->neurons, sizes, layers * sizeof(int));
    
    for(int i = 0; i < layers-1; i++) {
        int size = sizes[i] * sizes[i+1];
        net->weights[i] = (float*)malloc(size * sizeof(float));
        net->biases[i] = (float*)malloc(sizes[i+1] * sizeof(float));
        
        /* Инициализация случайными весами */
        for(int j = 0; j < size; j++)
            net->weights[i][j] = (rand() / (float)RAND_MAX) * 2 - 1;
        for(int j = 0; j < sizes[i+1]; j++)
            net->biases[i][j] = (rand() / (float)RAND_MAX) * 2 - 1;
    }
    
    return net;
}

/* Сохранение модели - общая функция */
void save_model(NeuralNet* net, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if(!f) return;
    
    /* Заголовок */
    fwrite("SKT", 1, 3, f);
    fputc(0, f);
    
    /* Обновляем timestamp перед сохранением */
    net->timestamp = (unsigned int)time(NULL);
    fwrite(&net->timestamp, sizeof(net->timestamp), 1, f);
    
    fwrite(&net->layers, sizeof(net->layers), 1, f);
    fwrite(net->neurons, sizeof(int), net->layers, f);
    
    /* Сохранение весов и смещений */
    for(int i = 0; i < net->layers-1; i++) {
        int size = net->neurons[i] * net->neurons[i+1];
        fwrite(net->weights[i], sizeof(float), size, f);
        fwrite(net->biases[i], sizeof(float), net->neurons[i+1], f);
    }
    
    fclose(f);
}

/* Загрузка модели - общая функция */
NeuralNet* load_model(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if(!f) return NULL;
    
    /* Проверка магического числа */
    char magic[4];
    fread(magic, 1, 4, f);
    if(memcmp(magic, "SKT", 3) != 0) {
        fclose(f);
        return NULL;
    }
    
    NeuralNet* net = (NeuralNet*)malloc(sizeof(NeuralNet));
    
    /* Теперь читаем timestamp в правильное место */
    fread(&net->timestamp, sizeof(unsigned int), 1, f);
    
    /* Читаем layers как int */
    fread(&net->layers, sizeof(int), 1, f);
    
    net->neurons = (int*)malloc(net->layers * sizeof(int));
    fread(net->neurons, sizeof(int), net->layers, f);
    
    /* Загрузка весов */
    net->weights = (float**)malloc((net->layers-1) * sizeof(float*));
    net->biases = (float**)malloc((net->layers-1) * sizeof(float*));
    
    for(int i = 0; i < net->layers-1; i++) {
        int size = net->neurons[i] * net->neurons[i+1];
        net->weights[i] = (float*)malloc(size * sizeof(float));
        net->biases[i] = (float*)malloc(net->neurons[i+1] * sizeof(float));
        
        fread(net->weights[i], sizeof(float), size, f);
        fread(net->biases[i], sizeof(float), net->neurons[i+1], f);
    }
    
    fclose(f);
    return net;
}