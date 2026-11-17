// tinyai_learn.c
// TinyC/DJGPP совместимый минималистичный тренер нейросети (Skeleton)
// Требования:
// - Целевая архитектура: базовый x86 (8088 и выше)
// - Зависимости: только стандартная библиотека C
// - Выходной формат: YYYYMMDD.SKT (8.3 формат на основе времени)

#include <stdio.h>      // Ввод-вывод в файл, стандартный ввод-вывод
#include <stdlib.h>     // malloc, free, общие утилиты
#include <string.h>     // Работа со строками
#include <time.h>       // Получение текущего времени для имени файла

// --- Определение структуры сети ---
#define TINYAI_INPUT_SIZE 4   // Пример размера входа
#define TINYAI_HIDDEN_SIZE 5  // Пример размера скрытого слоя
#define TINYAI_OUTPUT_SIZE 1  // Пример размера выхода

// Аппроксимация exp(x) для случаев, когда функция отсутствует (например, в tcc без libm)
float my_exp_approx(float x) {
    if(x > 10.0f) return 22026.465795f; // ~e^10
    if(x < -10.0f) return 0.000045f;    // ~e^-10
    float term1 = x;
    float term2 = term1 * x / 2.0f;
    float term3 = term2 * x / 3.0f;
    float term4 = term3 * x / 4.0f;
    float term5 = term4 * x / 5.0f;
    return 1.0f + term1 + term2 + term3 + term4 + term5;
}

typedef struct {
    float input[TINYAI_INPUT_SIZE];
    float hidden[TINYAI_HIDDEN_SIZE];
    float output[TINYAI_OUTPUT_SIZE];

    float w_ih[TINYAI_INPUT_SIZE][TINYAI_HIDDEN_SIZE]; // Веса вход -> скрытый
    float w_ho[TINYAI_HIDDEN_SIZE][TINYAI_OUTPUT_SIZE]; // Веса скрытый -> выход
    float b_h[TINYAI_HIDDEN_SIZE]; // Смещение скрытого слоя
    float b_o[TINYAI_OUTPUT_SIZE]; // Смещение выходного слоя
} TinyNeuralNet;

// --- Функция активации (например, сигмоида) ---
float sigmoid(float x) {
    float exp_neg_x = my_exp_approx(-x);
    return 1.0f / (1.0f + exp_neg_x);
}

// --- Инициализация весов ---
void init_net(TinyNeuralNet* net) {
    srand((unsigned int)time(NULL)); // Простая инициализация (низкая энтропия)
    int i, j;
    for(i = 0; i < TINYAI_INPUT_SIZE; ++i){
        for(j = 0; j < TINYAI_HIDDEN_SIZE; ++j){
            net->w_ih[i][j] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // От -1 до 1
        }
    }
    for(i = 0; i < TINYAI_HIDDEN_SIZE; ++i){
        for(j = 0; j < TINYAI_OUTPUT_SIZE; ++j){
             net->w_ho[i][j] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        }
        net->b_h[i] = 0.0f; // Начинаем со смещений 0
    }
    for(j = 0; j < TINYAI_OUTPUT_SIZE; ++j){
        net->b_o[j] = 0.0f;
    }
}

// --- Прямой проход (Forward Propagation) ---
void forward_pass(TinyNeuralNet* net) {
    int i, j;
    // Расчёт скрытого слоя
    for (i = 0; i < TINYAI_HIDDEN_SIZE; ++i) {
        net->hidden[i] = 0.0f;
        for (j = 0; j < TINYAI_INPUT_SIZE; ++j) {
            net->hidden[i] += net->input[j] * net->w_ih[j][i];
        }
        net->hidden[i] += net->b_h[i];
        net->hidden[i] = sigmoid(net->hidden[i]);
    }

    // Расчёт выходного слоя
    for (i = 0; i < TINYAI_OUTPUT_SIZE; ++i) {
        net->output[i] = 0.0f;
        for (j = 0; j < TINYAI_HIDDEN_SIZE; ++j) {
            net->output[i] += net->hidden[j] * net->w_ho[j][i];
        }
        net->output[i] += net->b_o[i];
        net->output[i] = sigmoid(net->output[i]);
    }
}

// --- Генерация имени файла (формат YYYYMMDD.HHH) ---
void generate_filename(char* filename_buffer, size_t buffer_size) {
    time_t now = time(0);
    struct tm *tm_now = localtime(&now);

    int year = tm_now->tm_year + 1900;
    int month = tm_now->tm_mon + 1; // tm_mon начинается с 0
    int day = tm_now->tm_mday;
    
    // Простая комбинация секунд и минут для создания 8-символьного HEX
    unsigned int hash_val = (unsigned int)(tm_now->tm_sec * 100 + tm_now->tm_min); 
    snprintf(filename_buffer, buffer_size, "%04d%02d%02d.%03X", year, month, day, hash_val % 0x1000);
}

// --- Сохранение модели (формат .SKT) ---
int save_model(TinyNeuralNet* net, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("╬ЁЁюЁ: ЌҐ ў¤ ыюё№ ёю§¤ Є№ м ©ыр єю¤Ґыш '%s'.\n", filename);
        return -1;
    }

    // Запись метаданных сети
    int in_size = TINYAI_INPUT_SIZE;
    int hid_size = TINYAI_HIDDEN_SIZE;
    int out_size = TINYAI_OUTPUT_SIZE;
    fwrite(&in_size, sizeof(int), 1, f);
    fwrite(&hid_size, sizeof(int), 1, f);
    fwrite(&out_size, sizeof(int), 1, f);

    // Запись весов и смещений
    fwrite(net->w_ih, sizeof(float), TINYAI_INPUT_SIZE * TINYAI_HIDDEN_SIZE, f);
    fwrite(net->w_ho, sizeof(float), TINYAI_HIDDEN_SIZE * TINYAI_OUTPUT_SIZE, f);
    fwrite(net->b_h, sizeof(float), TINYAI_HIDDEN_SIZE, f);
    fwrite(net->b_o, sizeof(float), TINYAI_OUTPUT_SIZE, f);

    fclose(f);
    printf("ћю¤Ґы№ ёю§¤р­р ў '%s'.\n", filename);
    return 0;
}

// --- Пустой цикл обучения (без реальной логики) ---
// Текущая функция просто использует фиктивные данные.
// Реальная логика должна извлекать данные из указанной папки.
void dummy_train_from_path(const char* data_path, TinyNeuralNet* net) {
    printf("─юыцҐ­ ЎлЄ№ ї®ҐҐҐ ­Ґ©Ё®­­л© єҐЎҐЁ, їҐ°ҐЎш°р№ ш ўлЎш°р№ ¤р­­лҐ ш§ ёўҐЄр '%s'...\n", data_path);
    printf("ќЈҐ­Ґ°ш°є¬ ¬­юЈҐёЄю єҐ¤шҐ Ё ўлїшёыҐ­шп, їюёЄюы№Єє ъюЄєё шёїюы№§єҐ¬ ўёҐЈю єҐЇ®ъҐЁшўрЄ№­лҐ ёшёЄҐ¬л єҐҐЄҐЁш­Јр.\n");
    printf("─®Є єҐҐЄҐЁш­Ј ­Ґ Ё¬ЇыҐ¬Ґ­Єш°ющ, ё¤ҐырҐ¬ мшЄЄшўл© ўл§юў ё ёђҐ¤ҐЄҐ ­ҐЄ®Є®°ле ёєрЄЁіҐёЄшї ўе®¤ют.\n");
    // Пример: подать несколько входов и получить результат
    for(int i=0; i<TINYAI_INPUT_SIZE; ++i) net->input[i] = 0.5f; // Фиктивный ввод
    forward_pass(net);
    printf("┬лўю¤: %f\n", net->output[0]);
    // Здесь должна быть реализована логика чтения файлов из data_path и их использования для обучения
}

int main(int argc, char *argv[]) {
    printf("«рЇєёЄ TinyAI-Learn...\n");

    // Инициализация сети
    TinyNeuralNet network;
    init_net(&network);

    if (argc < 2) {
        printf("╚ёЇюы№§шўрҐ¬ мшЄЄшўлҐ ¤р­­лҐ ¤ып ўлїшёыҐ­шп.\n");
        // Вызов старой фиктивной функции, если нет аргумента
        // dummy_train(&network); 
    } else {
        // Вызов новой функции, которая должна обрабатывать путь
        dummy_train_from_path(argv[1], &network);
    }


    // Генерация имени файла
    char skt_filename[13]; // 8.3 формат + завершающий нуль
    generate_filename(skt_filename, sizeof(skt_filename));

    // Сохранение модели
    if(save_model(&network, skt_filename) != 0) {
        return 1; // Ошибка
    }

    printf("TinyAI-Learn §рўҐ°°Ґ­.\n");
    return 0;
}