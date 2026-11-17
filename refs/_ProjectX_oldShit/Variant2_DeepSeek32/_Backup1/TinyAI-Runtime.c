/* TinyAI-Runtime - Автономный агент с самообучением */
#include "tinyml.h"

#define AGENT_MODE 1
#define MAX_MEMORY 65536

/* Глобальная память агента */
typedef struct {
    NeuralNet* net;
    char knowledge[MAX_MEMORY];
    int knowledge_size;
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
    
    NeuralNet* net = malloc(sizeof(NeuralNet));
    
    fread(&net->timestamp, sizeof(unsigned int), 1, f);
    fread(&net->layers, sizeof(unsigned short), 1, f);
    
    net->neurons = malloc(net->layers * sizeof(unsigned short));
    fread(net->neurons, sizeof(unsigned short), net->layers, f);
    
    /* Загрузка весов */
    net->weights = malloc((net->layers-1) * sizeof(float*));
    net->biases = malloc((net->layers-1) * sizeof(float*));
    
    for(int i = 0; i < net->layers-1; i++) {
        int size = net->neurons[i] * net->neurons[i+1];
        net->weights[i] = malloc(size * sizeof(float));
        net->biases[i] = malloc(net->neurons[i+1] * sizeof(float));
        
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

/* Самообучение на накопленных знаниях */
void self_learn(Agent* agent) {
    if(agent->knowledge_size < 100) return;
    
    for(int i = 0; i < agent->knowledge_size - 100; i += 100) {
        float input[256] = {0};
        float target[256] = {0};
        
        memcpy(input, &agent->knowledge[i], 100);
        memcpy(target, input, 100 * sizeof(float));
        
        /* Упрощенное дообучение */
        train_step(agent->net, input, target);
    }
}

/* Поиск моделей в файловой системе */
void find_and_learn(Agent* agent) {
    struct find_t file;
    
    if(_dos_findfirst("*.SKT", _A_NORMAL, &file) == 0) {
        do {
            if(strcmp(file.name, agent->current_model) != 0) {
                /* Загрузка и объединение знаний */
                NeuralNet* other = load_model(file.name);
                if(other) {
                    /* Упрощенное слияние моделей */
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
        } while(_dos_findnext(&file) == 0);
    }
}

/* Основной цикл агента */
void agent_loop() {
    Agent agent = {0};
    agent.net = NULL;
    
    /* Поиск последней модели */
    struct find_t file;
    unsigned long latest_time = 0;
    char latest_file[13] = {0};
    
    if(_dos_findfirst("*.SKT", _A_NORMAL, &file) == 0) {
        do {
            if(file.wr_time > latest_time) {
                latest_time = file.wr_time;
                strcpy(latest_file, file.name);
            }
        } while(_dos_findnext(&file) == 0);
    }
    
    if(latest_file[0]) {
        agent.net = load_model(latest_file);
        strcpy(agent.current_model, latest_file);
    }
    
    if(!agent.net) {
        /* Создание новой модели если нет существующей */
        int sizes[] = {256, 128, 64, 128, 256};
        agent.net = create_network(5, sizes);
        generate_filename(agent.current_model);
        save_model(agent.net, agent.current_model);
    }
    
    /* Основной цикл */
    while(1) {
        /* 1. Сбор данных из окружения */
        float sensor_data[256] = {0};
        /* Заглушка для сенсорных данных */
        
        /* 2. Принятие решений */
        agent_decision(&agent, sensor_data);
        
        /* 3. Самообучение */
        self_learn(&agent);
        
        /* 4. Поиск новых знаний */
        find_and_learn(&agent);
        
        /* 5. Сохранение прогресса */
        save_model(agent.net, agent.current_model);
        
        /* Пауза для экономии CPU */
        delay(1000);
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