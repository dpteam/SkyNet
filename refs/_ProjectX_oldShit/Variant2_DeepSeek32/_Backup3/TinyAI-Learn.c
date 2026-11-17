/* TinyAI-Learn - Компактный тренер нейросетей */
#include "tinyml.h"

#define LEARNING_RATE 0.01f
#define EPOCHS 1000

/* Инициализация сети */
NeuralNet* create_network(int layers, int* sizes) {
    NeuralNet* net = (NeuralNet*)malloc(sizeof(NeuralNet));
    net->layers = layers;
    net->neurons = (int*)malloc(layers * sizeof(int));
    net->weights = (float**)malloc((layers-1) * sizeof(float*));
    net->biases = (float**)malloc((layers-1) * sizeof(float*));
    
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

/* Прямое распространение */
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

/* Обучение на одном примере */
void train_step(NeuralNet* net, float* input, float* target) {
    float* output = forward(net, input);
    
    for(int l = net->layers-2; l >= 0; l--) {
        /* Упрощенный backpropagation */
        for(int i = 0; i < net->neurons[l+1]; i++) {
            float error = output[i] - target[i];
            
            for(int j = 0; j < net->neurons[l]; j++) {
                net->weights[l][j * net->neurons[l+1] + i] -= 
                    LEARNING_RATE * error * input[j];
            }
            net->biases[l][i] -= LEARNING_RATE * error;
        }
    }
    
    free(output);
}

/* Сохранение модели */
void save_model(NeuralNet* net, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if(!f) return;
    
    /* Заголовок */
    fwrite("SKT", 1, 3, f);
    fputc(0, f);
    
    unsigned int timestamp = (unsigned int)time(NULL);
    fwrite(&timestamp, sizeof(timestamp), 1, f);
    
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

/* Быстрый рандом (без зависимостей) */
static unsigned int fast_rand = 1;
static float fast_rand_float() {
    fast_rand = fast_rand * 1103515245 + 12345;
    return (fast_rand & 0x7FFF) / (float)0x7FFF;
}

int main(int argc, char** argv) {
    if(argc < 2) {
        printf("Usage: TINYAI-L <input_file>\n");
        return 1;
    }
    
    /* Инициализация рандома */
    fast_rand = (unsigned int)time(NULL);
    
    /* Создание простой сети */
    int sizes[] = {256, 64, 32, 64, 256};
    NeuralNet* net = create_network(5, sizes);
    
    /* Обучение на файле */
    FILE* f = fopen(argv[1], "rb");
    if(f) {
        float input[256] = {0};
        float target[256] = {0};
        
        fread(input, 1, 256, f);
        memcpy(target, input, 256 * sizeof(float));
        
        for(int e = 0; e < EPOCHS; e++) {
            train_step(net, input, target);
        }
        
        fclose(f);
    }
    
    /* Сохранение модели */
    char filename[13];
    generate_filename(filename);
    save_model(net, filename);
    
    printf("Model saved to %s\n", filename);
    
    return 0;
}