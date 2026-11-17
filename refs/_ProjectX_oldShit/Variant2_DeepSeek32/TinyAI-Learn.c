/* TinyAI-Learn - Компактный тренер нейросетей */
#include "tinyml.h"

#define LEARNING_RATE 0.01f
#define EPOCHS 1000

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
    srand(time(NULL));
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