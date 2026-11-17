/* sk_selftest.c - internal checks for the M0 core.
 *
 * These are not decoration.  Every invariant the recovery path relies on
 * is asserted here, so a port to a new compiler or CPU fails loudly in
 * the selftest instead of quietly in year 12 of the mission.
 *
 * SKYNET M0 / subsystem: core/selftest
 */
#include <stdio.h>
#include <string.h>
#include "sk_selftest.h"
#include "sk_math.h"
#include "sk_arena.h"
#include "sk_clock.h"
#include "sk_vault.h"

static int sk_fail;
static int sk_pass;

static void sk_check(const char *what, int ok)
{
    if (ok) {
        sk_pass++;
    } else {
        sk_fail++;
        printf("  FAIL  %s\n", what);
    }
}

static void sk_near(const char *what, sk_fix got, sk_i32 want, sk_i32 tol)
{
    sk_i32 d = (sk_i32)got - want;

    if (d < 0)
        d = -d;
    if (d > tol) {
        sk_fail++;
        printf("  FAIL  %s: got %ld want %ld +-%ld\n",
               what, (long)got, (long)want, (long)tol);
    } else {
        sk_pass++;
    }
}

static void sk_test_math(void)
{
    sk_rng a, b;
    sk_u32 i;
    int same;

    /* pi = 3.14159265 * 65536 = 205887.4 */
    sk_near("pi", sk_consts_get()->pi, 205887L, 8L);
    /* e = 2.71828183 * 65536 = 178145.4 */
    sk_near("e", sk_consts_get()->e, 178145L, 8L);
    /* sqrt2 = 1.41421356 * 65536 = 92681.9 */
    sk_near("sqrt2", sk_consts_get()->sqrt2, 92682L, 8L);
    /* ln2 = 0.69314718 * 65536 = 45426.1 */
    sk_near("ln2", sk_consts_get()->ln2, 45426L, 8L);
    /* golden = 1.61803399 * 65536 = 106039.0 */
    sk_near("golden", sk_consts_get()->golden, 106039L, 8L);

    /* 1.5 * 2.5 = 3.75 -> 245760 */
    sk_check("fix_mul 1.5*2.5",
             sk_fix_mul(98304L, 163840L) == 245760L);
    /* -1.5 * 2.5 = -3.75 */
    sk_check("fix_mul sign",
             sk_fix_mul(-98304L, 163840L) == -245760L);
    /* 1/3 = 0.33333 -> 21845 */
    sk_check("fix_div 1/3", sk_fix_div(SK_FIX_ONE, 196608L) == 21845L);
    /* 10/4 = 2.5 */
    sk_check("fix_div 10/4", sk_fix_div(655360L, 262144L) == 163840L);
    /* division by zero saturates, never traps */
    sk_check("fix_div /0 saturates", sk_fix_div(SK_FIX_ONE, 0) == SK_FIX_MAX);
    /* sqrt(4) = 2 exactly */
    sk_near("sqrt(4)", sk_fix_sqrt(262144L), 131072L, 2L);
    /* exp(0) = 1 */
    sk_near("exp(0)", sk_fix_exp(0), SK_FIX_ONE, 2L);
    /* exp(1) = 2.71828 */
    sk_near("exp(1)", sk_fix_exp(SK_FIX_ONE), 178145L, 64L);
    /* tanh(0) = 0, tanh(1) = 0.76159 */
    sk_near("tanh(0)", sk_fix_tanh(0), 0L, 2L);
    sk_near("tanh(1)", sk_fix_tanh(SK_FIX_ONE), 49920L, 200L);
    /* tanh saturates instead of exploding */
    sk_check("tanh(50) = 1", sk_fix_tanh(50L * SK_FIX_ONE) == SK_FIX_ONE);

    /* CRC32C check value from the spec */
    sk_check("crc32c(\"123456789\")",
             sk_crc32c("123456789", 9) == 0xE3069283UL);

    /* PRNG: same seed, same stream; different seed, different stream */
    sk_rng_seed(&a, 1, 2, 3, 4);
    sk_rng_seed(&b, 1, 2, 3, 4);
    same = 1;
    for (i = 0; i < 64; i++) {
        if (sk_rng_next(&a) != sk_rng_next(&b))
            same = 0;
    }
    sk_check("rng deterministic", same);

    sk_rng_seed(&a, 1, 2, 3, 4);
    sk_rng_seed(&b, 1, 2, 3, 5);
    same = 1;
    for (i = 0; i < 8; i++) {
        if (sk_rng_next(&a) != sk_rng_next(&b))
            same = 0;
    }
    sk_check("rng seed matters", !same);

    /* the all-zero state must not be a fixed point */
    sk_rng_seed(&a, 0, 0, 0, 0);
    sk_check("rng zero-seed breaks", sk_rng_next(&a) != 0);
}

static const sk_u32 sk_prime_ref[20] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71
};

static const sk_u32 sk_super_ref[10] = {
    3, 5, 11, 17, 31, 41, 59, 67, 83, 109
};

static void sk_test_clock(void)
{
    sk_clock c;
    sk_u32 i;
    int ok;
    sk_u32 v500, idx500;

    sk_clock_init(&c);
    ok = 1;
    for (i = 0; i < 20; i++) {
        if (!sk_clock_next(&c) || c.value != sk_prime_ref[i])
            ok = 0;
    }
    sk_check("first 20 primes", ok);
    sk_check("index after 20", c.index == 20);

    /* super-primes: the p_n for which n is prime */
    sk_clock_init(&c);
    i = 0;
    ok = 1;
    while (i < 10) {
        if (!sk_clock_next(&c))
            break;
        if (c.is_super) {
            if (c.value != sk_super_ref[i])
                ok = 0;
            i++;
        }
    }
    sk_check("first 10 super-primes", ok && i == 10);
    sk_check("super count matches", c.supers == 10);

    /* seek must continue exactly where the run left off */
    sk_clock_init(&c);
    v500 = 0;
    idx500 = 0;
    for (i = 0; i < 1000; i++) {
        sk_clock_next(&c);
        if (i == 499) {
            v500 = c.value;
            idx500 = c.index;
        }
    }
    {
        sk_clock d;
        sk_clock_seek(&d, v500, idx500);
        for (i = 500; i < 1000; i++)
            sk_clock_next(&d);
        sk_check("clock seek resumes exactly", d.value == c.value);
        sk_check("clock seek index", d.index == c.index);
    }

    /* is_prime agrees with the sieve over a range */
    ok = 1;
    for (i = 0; i < 200; i++) {
        sk_u32 n, d, prime;
        n = i;
        prime = 1;
        if (n < 2) {
            prime = 0;
        } else {
            for (d = 2; d * d <= n; d++) {
                if (n % d == 0) {
                    prime = 0;
                    break;
                }
            }
        }
        if (sk_clock_is_prime(n) != (int)prime)
            ok = 0;
    }
    sk_check("is_prime agrees with trial division", ok);
}

static void sk_test_arena(void)
{
    sk_arena a;
    void *p;
    sk_u32 mark;
    int ok;

    sk_check("arena init", sk_arena_init(&a, 4096, SK_MEM_OS) == 1);
    mark = sk_arena_mark(&a);
    p = sk_arena_alloc(&a, 100);
    sk_check("arena alloc", p != SK_NULL);
    sk_check("arena advances", sk_arena_mark(&a) >= mark + 100);
    sk_arena_release(&a, mark);
    sk_check("arena release", sk_arena_mark(&a) == mark);

    /* exhausting the arena returns NULL and never aborts */
    ok = 1;
    while (sk_arena_alloc(&a, 256) != SK_NULL)
        ;
    sk_check("arena OOM returns NULL", a.oom > 0);
    sk_arena_reset(&a);
    sk_check("arena reset", sk_arena_mark(&a) == 0);
    (void)ok;
    sk_arena_done(&a);

    /* degradation (invariant I6): an impossible request must produce a
       smaller arena, not a failure */
    {
        sk_arena big;
        sk_u32 got;
        got = sk_arena_init_best(&big, 0xFFFFFFF0UL, 32UL * 1024UL, SK_MEM_OS);
        sk_check("arena degrades instead of failing", got > 0);
        sk_check("arena degradation is within bounds",
                 got > 0 && got <= 0xFFFFFFF0UL);
        if (got > 0) {
            sk_check("degraded arena is usable",
                     sk_arena_alloc(&big, 16) != SK_NULL);
            sk_arena_done(&big);
        }
    }
}

int sk_selftest_run(void)
{
    sk_fail = 0;
    sk_pass = 0;

    printf("SkyNet M0 selftest\n");
    printf("------------------\n");

    sk_math_init();
    sk_test_math();
    sk_test_clock();
    sk_test_arena();

    printf("vault...\n");
    sk_check("vault crash/recovery", sk_vault_selftest() == 1);

    printf("------------------\n");
    printf("%d passed, %d failed\n", sk_pass, sk_fail);
    return sk_fail;
}
