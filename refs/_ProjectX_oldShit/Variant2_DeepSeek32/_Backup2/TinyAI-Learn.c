/* TinyAI-Learn - Компактный тренер нейросетей */
#include "tinyml.h"

#define MAX_PATH 260
#define LEARNING_RATE 0.01f
#define EPOCHS 1000

/* Простая структура для поиска файлов в DOS */
typedef struct {
    char name[13];
    unsigned int size;
    unsigned int time;
} DOSFileInfo;

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
    fputc(0, f); /* Завершение магического числа */
    
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

/* Простой поиск файлов для DOS */
int find_first_file(const char* pattern, DOSFileInfo* info) {
    FILE* f = fopen(pattern, "rb");
    if(!f) return 0;
    
    strcpy(info->name, pattern);
    fseek(f, 0, SEEK_END);
    info->size = ftell(f);
    fclose(f);
    info->time = (unsigned int)time(NULL);
    return 1;
}

/* Обработка одного файла */
void process_file(const char* filename, NeuralNet* net) {
    FILE* f = fopen(filename, "rb");
    if(!f) return;
    
    float input[256] = {0};
    float target[256] = {0};
    
    fread(input, 1, 256, f);
    memcpy(target, input, 256 * sizeof(float));
    
    for(int e = 0; e < EPOCHS; e++) {
        train_step(net, input, target);
    }
    
    fclose(f);
}

int main(int argc, char** argv) {
    if(argc < 2) {
        printf("Usage: TINYAI-L <directory or file>\n");
        return 1;
    }
    
    srand(time(NULL));
    
    /* Создание простой сети */
    int sizes[] = {256, 64, 32, 64, 256}; /* Автоэнкодер */
    NeuralNet* net = create_network(5, sizes);
    
    /* Простая обработка: если это файл, обучаем на нем */
    DOSFileInfo info;
    if(find_first_file(argv[1], &info)) {
        process_file(argv[1], net);
    } else {
        /* Пытаемся обработать как файл с именем argv[1] */
        process_file(argv[1], net);
    }
    
    /* Сохранение модели */
    char filename[13];
    generate_filename(filename);
    save_model(net, filename);
    
    printf("Model saved to %s\n", filename);
    
    return 0;
}