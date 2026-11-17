/*
    seed_agi_toy.c

    Minimal active-inference "AGI seed" toy.

    This is NOT AGI.
    It is a tiny demonstration of:
      - prediction model
      - surprise-driven learning
      - action selection by preferred future states
      - curiosity / exploration bonus
      - forgetting / memory decay

    Intentionally 16-bit friendly:
      - no malloc
      - no floating point
      - no large arrays
      - only integer arithmetic
*/

#include <stdio.h>

#define SENSES      16
#define ACTIONS     3
#define WORLD_SIZE  32
#define FOOD_COUNT  3
#define RUN_STEPS   20000U
#define DECAY_EVERY 512U

typedef unsigned int u16;
typedef unsigned char u8;

/*
    Transition memory:
    P(next_sense | current_sense, action)

    Very small:
    16 senses * 3 actions * 16 next senses * 2 bytes = 1536 bytes
*/
static u16 trans[SENSES][ACTIONS][SENSES];
static u16 total[SENSES][ACTIONS];

/*
    Preferences over senses.

    Sense encodes signed distance to nearest food:
      index 0  => far left  (-7)
      index 7  => food here (0)
      index 14 => far right (+7)
      index 15 => unused / no food

    Agent prefers index 7.
*/
static u8 pref[SENSES] = {
    5, 10, 20, 35, 55, 85, 150, 255,
    150, 85, 55, 35, 20, 10, 5, 0
};

static int food[FOOD_COUNT] = {4, 15, 28};
static int pos = WORLD_SIZE / 2;

static u16 eaten = 0;
static u16 age = 0;
static u16 rng = 12345u;

/* Simple deterministic 16-bit xorshift RNG */
static u16 rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 5;
    return rng;
}

static int duplicate_food(int idx, int p)
{
    int i;
    for (i = 0; i < FOOD_COUNT; ++i) {
        if (i != idx && food[i] == p) {
            return 1;
        }
    }
    return 0;
}

static void respawn_food(int idx)
{
    int attempts = 0;
    int p = 0;

    do {
        p = (int)(rnd() % WORLD_SIZE);
        ++attempts;
    } while ((p == pos || duplicate_food(idx, p)) && attempts < 64);

    food[idx] = p;
}

/*
    Sense = clipped signed distance to nearest food.

    Returns 0..14 normally:
      -7 => 0
       0 => 7
      +7 => 14
*/
static int sense(void)
{
    int i;
    int best_abs = 999;
    int best_off = 0;

    for (i = 0; i < FOOD_COUNT; ++i) {
        int off = food[i] - pos;
        int abs_off = off < 0 ? -off : off;

        if (abs_off < best_abs) {
            best_abs = abs_off;
            best_off = off;
        }
    }

    if (best_off < -7) best_off = -7;
    if (best_off > 7)  best_off = 7;

    return best_off + 7;
}

/*
    Forgetting / compression.

    Halves all counts periodically.
    This prevents saturation and allows adaptation.
*/
static void decay_model(void)
{
    int s, a, ns;

    for (s = 0; s < SENSES; ++s) {
        for (a = 0; a < ACTIONS; ++a) {
            for (ns = 0; ns < SENSES; ++ns) {
                trans[s][a][ns] >>= 1;
            }
            total[s][a] >>= 1;
        }
    }
}

static void learn(int s, int a, int ns)
{
    /* Avoid 16-bit overflow by decaying first if needed */
    if (trans[s][a][ns] == 65535u || total[s][a] == 65535u) {
        decay_model();
    }

    ++trans[s][a][ns];
    ++total[s][a];

    if (++age >= DECAY_EVERY) {
        age = 0;
        decay_model();
    }
}

/*
    Choose action by approximate active-inference score:

    1. Expected preference of next state.
    2. Uncertainty bonus / curiosity.
*/
static int choose_action(int s)
{
    int a, ns;
    int best_a;
    unsigned long best_score, score;
    unsigned int tot;

    best_a = 1;
    best_score = 0;

    for (a = 0; a < ACTIONS; ++a) {
        tot = total[s][a];

        if (tot == 0) {
            /* Strong curiosity for unknown state-action pairs */
            score = 100000UL + (unsigned long)(rnd() & 0xFFu);
        } else {
            score = 0;

            for (ns = 0; ns < SENSES; ++ns) {
                score += (unsigned long)trans[s][a][ns]
                       * (unsigned long)pref[ns];
            }

            score /= tot;

            /* Smaller uncertainty bonus when model is confident */
            score += 5000UL / ((unsigned long)tot + 1UL);

            /* Tiny random tie-break */
            score += (unsigned long)(rnd() & 0x0Fu);
        }

        if (score > best_score) {
            best_score = score;
            best_a = a;
        }
    }

    return best_a;
}

static void do_action(int a)
{
    if (a == 0) {              /* move left */
        if (pos > 0) {
            --pos;
        }
    } else if (a == 2) {       /* move right */
        if (pos < WORLD_SIZE - 1) {
            ++pos;
        }
    }
    /* a == 1: stay */
}

static void maybe_eat(void)
{
    int i;

    for (i = 0; i < FOOD_COUNT; ++i) {
        if (food[i] == pos) {
            ++eaten;
            respawn_food(i);
        }
    }
}

int main(void)
{
    unsigned int step;
    int s;

    s = sense();

    printf("Tiny active-inference AGI seed toy\n");
    printf("world_size=%d food_count=%d steps=%u\n",
           WORLD_SIZE, FOOD_COUNT, (unsigned)RUN_STEPS);

    for (step = 0; step < RUN_STEPS; ++step) {
        int a;
        int ns;

        a = choose_action(s);

        do_action(a);
        maybe_eat();

        ns = sense();
        learn(s, a, ns);

        if ((step & 0x00FFu) == 0) {
            printf("step=%5u pos=%2d sense=%2d act=%c eaten=%u\n",
                   step,
                   pos,
                   s,
                   (a == 0 ? '<' : (a == 2 ? '>' : '.')),
                   eaten);
        }

        s = ns;
    }

    printf("done eaten=%u\n", eaten);
    return 0;
}