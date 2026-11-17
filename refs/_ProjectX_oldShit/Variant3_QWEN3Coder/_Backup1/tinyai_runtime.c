// tinyai_runtime.c
// TinyC/DJGPP 호환 런타임 (Skeleton)
// 요구 사항:
// - .SKT 파일 로드
// - 에이전트 모드 (자체 실행)
// - 자체 학습/수정 가능성 (개념적)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// TinyAI-Learn 과 동일한 구조체 사용
#define TINYAI_INPUT_SIZE 4
#define TINYAI_HIDDEN_SIZE 5
#define TINYAI_OUTPUT_SIZE 1

typedef struct {
    float input[TINYAI_INPUT_SIZE];
    float hidden[TINYAI_HIDDEN_SIZE];
    float output[TINYAI_OUTPUT_SIZE];

    float w_ih[TINYAI_INPUT_SIZE][TINYAI_HIDDEN_SIZE];
    float w_ho[TINYAI_HIDDEN_SIZE][TINYAI_OUTPUT_SIZE];
    float b_h[TINYAI_HIDDEN_SIZE];
    float b_o[TINYAI_OUTPUT_SIZE];
} TinyNeuralNet;

float sigmoid(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + exp(-x));
}

void forward_pass(TinyNeuralNet* net) {
    int i, j;
    for (i = 0; i < TINYAI_HIDDEN_SIZE; ++i) {
        net->hidden[i] = 0.0f;
        for (j = 0; j < TINYAI_INPUT_SIZE; ++j) {
            net->hidden[i] += net->input[j] * net->w_ih[j][i];
        }
        net->hidden[i] += net->b_h[i];
        net->hidden[i] = sigmoid(net->hidden[i]);
    }
    for (i = 0; i < TINYAI_OUTPUT_SIZE; ++i) {
        net->output[i] = 0.0f;
        for (j = 0; j < TINYAI_HIDDEN_SIZE; ++j) {
            net->output[i] += net->hidden[j] * net->w_ho[j][i];
        }
        net->output[i] += net->b_o[i];
        net->output[i] = sigmoid(net->output[i]);
    }
}

// --- 모델 로드 (.SKT 포맷) ---
int load_model(TinyNeuralNet* net, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("오류: 모델 파일 '%s'를 찾거나 열 수 없습니다.\n", filename);
        return -1;
    }

    int in_sz, hid_sz, out_sz;
    fread(&in_sz, sizeof(int), 1, f);
    fread(&hid_sz, sizeof(int), 1, f);
    fread(&out_sz, sizeof(int), 1, f);

    if(in_sz != TINYAI_INPUT_SIZE || hid_sz != TINYAI_HIDDEN_SIZE || out_sz != TINYAI_OUTPUT_SIZE){
        printf("오류: 모델 파일 '%s'의 구조가 현재 코드와 다릅니다.\n", filename);
        fclose(f);
        return -1;
    }

    fread(net->w_ih, sizeof(float), TINYAI_INPUT_SIZE * TINYAI_HIDDEN_SIZE, f);
    fread(net->w_ho, sizeof(float), TINYAI_HIDDEN_SIZE * TINYAI_OUTPUT_SIZE, f);
    fread(net->b_h, sizeof(float), TINYAI_HIDDEN_SIZE, f);
    fread(net->b_o, sizeof(float), TINYAI_OUTPUT_SIZE, f);

    fclose(f);
    printf("'%s'에서 모델을 불러왔습니다.\n", filename);
    return 0;
}

// --- 에이전트 행동 시뮬레이션 (개념적) ---
void agent_behavior(TinyNeuralNet* net) {
    printf("에이전트 행동 모드 시작...\n");
    // 이 부분은 실제 시스템 API(파일 시스템, 네트워크)에 접근해야 하며,
    // 매우 구현하기 복잡하고 플랫폼에 따라 달라집니다.
    // 예를 들어, 파일을 스캔하고 데이터를 처리하여 네트워크에 입력하는 로직이 필요합니다.
    // 현재는 더미 입력으로 대체합니다.
    for(int i=0; i<TINYAI_INPUT_SIZE; ++i) net->input[i] = 0.1f * i; // 더미 환경 입력
    forward_pass(net);
    printf("에이전트 출력: %f\n", net->output[0]);
    // *** 경고: '자체 학습/수정' 및 '자체 복제' 기능은 보안상 위험하며,
    //           실제 구현 시 매우 신중하게 접근해야 합니다.
    //           이는 악성코드나 바이러스의 전형적인 특징입니다.
    //           아래 코드는 그 개념만 설명하기 위한 가상의 의사코드입니다.
    /*
    // PSEUDO-CODE FOR SELF-MODIFICATION
    if (net->output[0] > 0.8f) { // 특정 임계값
        printf("에이전트가 자체 수정을 시작합니다...\n");
        // 네트워크 자체를 변경하거나 새로운 .SKT 파일을 생성하려는 시도
        // 또는 자기 자신을 복제하여 다른 위치에 저장
        // char new_name[] = "NEW_AGENT.SKT";
        // save_model(net, new_name); // save_model 함수 필요
    }
    */
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("사용법: %s <model.skt>\n", argv[0]);
        return 1;
    }

    printf("TinyAI-Runtime 시작 중...\n");

    TinyNeuralNet runtime_network;

    if(load_model(&runtime_network, argv[1]) != 0) {
        return 1; // 모델 로드 실패
    }

    agent_behavior(&runtime_network);

    printf("TinyAI-Runtime 종료.\n");
    return 0;
}