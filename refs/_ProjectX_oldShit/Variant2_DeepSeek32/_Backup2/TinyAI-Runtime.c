/* TinyAI-Runtime - Автономный агент с самообучением */
#include "tinyml.h"

#define AGENT_MODE 1
#define MAX_MEMORY 65536

/* Глобальная память агента */
typedef struct {
    NeuralNet* net;
    char knowledge[MAX_MEMORY];
    int knowledge_size;
    char current_model[13];
} Agent;

/* Загрузка модели */
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
    
    fread(&net->timestamp, sizeof(unsigned int), 1, f);
    fread(&net->layers, sizeof(unsigned short), 1, f);
    
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

/* Автономное принятие решений */
void agent_decision(Agent* agent, float* input) {
    float* output = forward(agent->net, input);
    
    /* Простое правило: если выход > 0.5, действие */
    for(int i = 0; i < agent->net->neurons[agent->net->layers-1]; i++) {
        if(output[i] > 0.5f) {
            /* Сохранение в знания */
            if(agent->knowledge_size < MAX_MEMORY - 100) {
                memcpy(&agent->knowledge[agent->knowledge_size], 
                       input, 100);
                agent->knowledge_size += 100;
            }
        }
    }
    
    free(output);
}

/* Функция обучения для рантайма */
void train_step_runtime(NeuralNet* net, float* input, float* target) {
    float* output = forward(net, input);
    
    for(int l = net->layers-2; l >= 0; l--) {
        for(int i = 0; i < net->neurons[l+1]; i++) {
            float error = output[i] - target[i];
            
            for(int j = 0; j < net->neurons[l]; j++) {
                net->weights[l][j * net->neurons[l+1] + i] -= 
                    0.01f * error * input[j];
            }
            net->biases[l][i] -= 0.01f * error;
        }
    }
    
    free(output);
}

/* Самообучение на накопленных знаниях */
void self_learn(Agent* agent) {
    if(agent->knowledge_size < 100) return;
    
    for(int i = 0; i < agent->knowledge_size - 100; i += 100) {
        float input[256] = {0};
        float target[256] = {0};
        
        memcpy(input, &agent->knowledge[i], 100);
        memcpy(target, input, 100 * sizeof(float));
        
        train_step_runtime(agent->net, input, target);
    }
}

/* Поиск моделей в файловой системе (упрощенный) */
void find_and_learn(Agent* agent) {
    /* Ищем файлы .SKT в текущей директории */
    FILE* f = fopen("*.SKT", "rb");
    if(f) {
        fclose(f);
        /* Упрощенная логика: если нашли другой файл, загружаем его */
        if(strcmp("00000000.SKT", agent->current_model) != 0) {
            NeuralNet* other = load_model("00000000.SKT");
            if(other) {
                /* Упрощенное слияние: усреднение весов */
                for(int l = 0; l < agent->net->layers-1; l++) {
                    int size = agent->net->neurons[l] * 
                              agent->net->neurons[l+1];
                    for(int i = 0; i < size; i++) {
                        agent->net->weights[l][i] = 
                            (agent->net->weights[l][i] + 
                             other->weights[l][i]) / 2;
                    }
                }
                
                /* Сохранение объединенной модели */
                char newname[13];
                generate_filename(newname);
                save_model(agent->net, newname);
                strcpy(agent->current_model, newname);
            }
        }
    }
}

/* Основной цикл агента */
void agent_loop() {
    Agent agent;
    agent.net = NULL;
    agent.knowledge_size = 0;
    
    /* Пытаемся загрузить модель по умолчанию */
    agent.net = load_model("00000000.SKT");
    if(agent.net) {
        strcpy(agent.current_model, "00000000.SKT");
    } else {
        /* Создание новой модели если нет существующей */
        int sizes[] = {256, 128, 64, 128, 256};
        agent.net = create_network(5, sizes);
        generate_filename(agent.current_model);
        save_model(agent.net, agent.current_model);
    }
    
    /* Основной цикл */
    while(1) {
        /* 1. Сбор данных из окружения (заглушка) */
        float sensor_data[256] = {0};
        
        /* 2. Принятие решений */
        agent_decision(&agent, sensor_data);
        
        /* 3. Самообучение */
        self_learn(&agent);
        
        /* 4. Поиск новых знаний */
        find_and_learn(&agent);
        
        /* 5. Сохранение прогресса каждые 10 циклов */
        static int cycle = 0;
        if(cycle % 10 == 0) {
            save_model(agent.net, agent.current_model);
        }
        cycle++;
        
        /* Пауза для экономии CPU */
        /* Простая задержка для DOS */
        for(int i = 0; i < 1000000; i++) {
            /* busy wait */
        }
    }
}

int main(int argc, char** argv) {
    printf("TinyAI Runtime Agent v1.0\n");
    
    #ifdef AGENT_MODE
    agent_loop();
    #else
    /* Интерактивный режим */
    if(argc > 1) {
        NeuralNet* net = load_model(argv[1]);
        if(net) {
            /* Обработка входных данных */
            float input[256] = {0};
            /* ... */
        }
    }
    #endif
    
    return 0;
}