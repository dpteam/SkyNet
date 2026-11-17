// tinyai_learn.c
// TinyC/DJGPP 호환 최소형 뉴럴 네트워크 학습기 (Skeleton)
// 요구 사항:
// - 타겟 아키텍처: 기본 x86 (8088 이상)
// - 의존성: 없음 (표준 C 라이브러리만 사용)
// - 출력 형식: YYYYMMDD.SKT (8.3 포맷의 시간 기반 해시)

#include <stdio.h>      // 파일 입출력, 표준 입출력
#include <stdlib.h>     // malloc, free, 일반 유틸리티
#include <string.h>     // 문자열 조작
#include <time.h>       // 파일 이름 생성을 위한 현재 시간
#include <math.h>       // 수학 연산 (tcc/djgpp에서 사용 가능 여부 확인 필요)

// --- 네트워크 구조 정의 ---
#define TINYAI_INPUT_SIZE 4   // 예시 입력 크기
#define TINYAI_HIDDEN_SIZE 5  // 예시 은닉층 크기
#define TINYAI_OUTPUT_SIZE 1  // 예시 출력 크기

typedef struct {
    float input[TINYAI_INPUT_SIZE];
    float hidden[TINYAI_HIDDEN_SIZE];
    float output[TINYAI_OUTPUT_SIZE];

    float w_ih[TINYAI_INPUT_SIZE][TINYAI_HIDDEN_SIZE]; // 입력 -> 은닉 가중치
    float w_ho[TINYAI_HIDDEN_SIZE][TINYAI_OUTPUT_SIZE]; // 은닉 -> 출력 가중치
    float b_h[TINYAI_HIDDEN_SIZE]; // 은닉층 편향
    float b_o[TINYAI_OUTPUT_SIZE]; // 출력층 편향
} TinyNeuralNet;

// --- 활성화 함수 (예: 시그모이드) ---
float sigmoid(float x) {
    if (x > 10.0f) return 1.0f; // 오버플로우 방지
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + exp(-x));
}

// --- 가중치 초기화 ---
void init_net(TinyNeuralNet* net) {
    srand(time(NULL)); // 매우 간단한 초기화 (정확도 낮음)
    int i, j;
    for(i = 0; i < TINYAI_INPUT_SIZE; ++i){
        for(j = 0; j < TINYAI_HIDDEN_SIZE; ++j){
            net->w_ih[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // -1 ~ 1 사이
        }
    }
    for(i = 0; i < TINYAI_HIDDEN_SIZE; ++i){
        for(j = 0; j < TINYAI_OUTPUT_SIZE; ++j){
            net->w_ho[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        net->b_h[i] = 0.0f; // 편향은 0으로 시작
    }
    for(j = 0; j < TINYAI_OUTPUT_SIZE; ++j){
        net->b_o[j] = 0.0f;
    }
}

// --- 순전파 (Forward Propagation) ---
void forward_pass(TinyNeuralNet* net) {
    int i, j;
    // 은닉층 계산
    for (i = 0; i < TINYAI_HIDDEN_SIZE; ++i) {
        net->hidden[i] = 0.0f;
        for (j = 0; j < TINYAI_INPUT_SIZE; ++j) {
            net->hidden[i] += net->input[j] * net->w_ih[j][i];
        }
        net->hidden[i] += net->b_h[i];
        net->hidden[i] = sigmoid(net->hidden[i]);
    }

    // 출력층 계산
    for (i = 0; i < TINYAI_OUTPUT_SIZE; ++i) {
        net->output[i] = 0.0f;
        for (j = 0; j < TINYAI_HIDDEN_SIZE; ++j) {
            net->output[i] += net->hidden[j] * net->w_ho[j][i];
        }
        net->output[i] += net->b_o[i];
        net->output[i] = sigmoid(net->output[i]);
    }
}

// --- 파일 이름 생성 (YYYYMMDD.HHH 형식) ---
void generate_filename(char* filename_buffer, size_t buffer_size) {
    time_t now = time(0);
    struct tm *tm_now = localtime(&now);

    int year = tm_now->tm_year + 1900;
    int month = tm_now->tm_mon + 1; // tm_mon은 0부터 시작
    int day = tm_now->tm_mday;
    
    // 간단한 해시 함수 대신 시간 초와 분을 조합하여 8자리 HEX 생성
    unsigned int hash_val = (unsigned int)(tm_now->tm_sec * 100 + tm_now->tm_min); 
    snprintf(filename_buffer, buffer_size, "%04d%02d%02d.%03X", year, month, day, hash_val % 0x1000);
    // 주의: 이 해싱은 충돌 가능성이 높으며 실제 프로덕션 용도로는 부적절합니다.
}

// --- 모델 저장 (.SKT 포맷) ---
int save_model(TinyNeuralNet* net, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("오류: 모델 파일 '%s'를 생성할 수 없습니다.\n", filename);
        return -1;
    }

    // 네트워크 메타데이터 쓰기
    fwrite(&(TINYAI_INPUT_SIZE), sizeof(int), 1, f);
    fwrite(&(TINYAI_HIDDEN_SIZE), sizeof(int), 1, f);
    fwrite(&(TINYAI_OUTPUT_SIZE), sizeof(int), 1, f);

    // 가중치 및 편향 쓰기
    fwrite(net->w_ih, sizeof(float), TINYAI_INPUT_SIZE * TINYAI_HIDDEN_SIZE, f);
    fwrite(net->w_ho, sizeof(float), TINYAI_HIDDEN_SIZE * TINYAI_OUTPUT_SIZE, f);
    fwrite(net->b_h, sizeof(float), TINYAI_HIDDEN_SIZE, f);
    fwrite(net->b_o, sizeof(float), TINYAI_OUTPUT_SIZE, f);

    fclose(f);
    printf("모델이 '%s'에 저장되었습니다.\n", filename);
    return 0;
}

// --- 더미 학습 루프 (실제 학습 로직 없음) ---
void dummy_train(TinyNeuralNet* net) {
    // 이 함수는 실제 역전파 학습 알고리즘을 포함하지 않습니다.
    // 실제 구현은 훨씬 복잡하며, 현재 요구 사항(초소형, 8088 호환 등)과 상충될 수 있습니다.
    printf("더미 학습 단계 실행 중...\n");
    // 예시: 몇 가지 입력을 넣고 결과 확인
    for(int i=0; i<TINYAI_INPUT_SIZE; ++i) net->input[i] = 0.5f; // 더미 입력
    forward_pass(net);
    printf("출력: %f\n", net->output[0]);
    // 실제 학습 로직이 여기에 들어가야 함
}

int main(int argc, char *argv[]) {
    printf("TinyAI-Learn 시작 중...\n");

    // 네트워크 초기화
    TinyNeuralNet network;
    init_net(&network);

    // 더미 데이터로 학습 시뮬레이션
    dummy_train(&network);

    // 파일 이름 생성
    char skt_filename[13]; // 8.3 포맷 + 널 종료자
    generate_filename(skt_filename, sizeof(skt_filename));

    // 모델 저장
    if(save_model(&network, skt_filename) != 0) {
        return 1; // 오류 발생
    }

    printf("TinyAI-Learn 완료.\n");
    return 0;
}