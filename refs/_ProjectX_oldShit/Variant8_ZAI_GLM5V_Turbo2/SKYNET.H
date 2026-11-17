/*
 * ============================================================
 *  SKYNET.H - Universal Header v1.2
 *  Works: TCC(MinGW), GCC, Clang, Turbo C++, Watcom, MSC
 * ============================================================
 */

#ifndef SKYNET_H
#define SKYNET_H

/* ===== Типы данных - безопасное определение ===== */
#if defined(__TURBOC__) || defined(__BORLANDC__)
    /* Turbo C++ / Borland C для DOS - нет stdint.h */
    typedef signed char     int8_t;
    typedef unsigned char   uint8_t;
    typedef signed short    int16_t;
    typedef unsigned short  uint16_t;
    typedef long            int32_t;
    typedef unsigned long   uint32_t;

#elif defined(_MSC_VER) && (_MSC_VER < 1600)
    /* Старый Visual Studio */
    typedef __int8          int8_t;
    typedef unsigned __int8 uint8_t;
    typedef __int16         int16_t;
    typedef unsigned __int16 uint16_t;
    typedef __int32         int32_t;
    typedef unsigned __int32 uint32_t;

#else
    /* Все остальные: GCC, Clang, modern TCC, Watcom и т.д. */
    #include <stdint.h>
#endif

/* ===== Константы сети ===== */
#define NN_INPUTS       8
#define NN_HIDDEN       16
#define NN_OUTPUTS      4

#define LEARNING_RATE   64          /* Q8.8: 0.25 */
#define EPOCHS          2000

/* ===== Fixed-Point Arithmetic (Q8.8 format) ===== */
#define FP_SHIFT        8
#define FP_ONE          (256)       /* 1.0 in Q8.8 */
#define FP_HALF         (128)       /* 0.5 */
#define FP_QUARTER      (64)        /* 0.25 */

#define INT_TO_FP(x)    ((int16_t)((x) << FP_SHIFT))
#define FP_TO_INT(x)    ((x) >> FP_SHIFT)

/* Умножение двух Q8.8 → результат Q8.8 */
static int16_t fp_mul(int16_t a, int16_t b) {
    return (int16_t)(((int32_t)a * b) >> FP_SHIFT);
}

/* Деление */
static int16_t fp_div(int16_t a, int16_t b) {
    if (b == 0) return 0;
    return (int16_t)(((int32_t)a << FP_SHIFT) / b);
}

/* ===== Структуры данных ===== */

/* Нейросеть */
typedef struct {
    int16_t w_ih[NN_INPUTS][NN_HIDDEN];     /* Веса: input→hidden */
    int16_t w_ho[NN_HIDDEN][NN_OUTPUTS];    /* Веса: hidden→output */
    int16_t b_hidden[NN_HIDDEN];             /* Байасы скрытого слоя */
    int16_t b_output[NN_OUTPUTS];            /* Байасы выходного слоя */
} NeuralNetwork;

/* Один обучающий образец */
typedef struct {
    int16_t inputs[NN_INPUTS];              /* Входы (Q8.8) */
    int16_t targets[NN_OUTPUTS];            /* Цели (Q8.8) */
} TrainingSample;

/* Заголовок файла весов */
#define SKYNET_MAGIC     0x534E  /* 'SN' */
#define SKYNET_VERSION   0x0001

typedef struct {
    uint16_t magic;
    uint16_t version;
    uint8_t  inputs;
    uint8_t  hidden;
    uint8_t  outputs;
    uint16_t reserved;
} WeightFileHeader;

#endif /* SKYNET_H */