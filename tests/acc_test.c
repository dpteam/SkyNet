/* acc_test.c - unit test for the 64-bit accumulator helpers.
 *
 * Written because the forward pass saturated every attention output and the
 * cause was not obvious from the model code: it had to be proved whether
 * the accumulator itself is wrong or the values fed to it.
 */
#include <stdio.h>
#include "sk_types.h"
#include "sk_math.h"

static int fails;

static void chk(const char *what, sk_fix got, sk_fix want)
{
    if (got != want) {
        printf("FAIL %-28s got %ld want %ld\n", what, (long)got, (long)want);
        fails++;
    } else {
        printf("ok   %-28s %ld\n", what, (long)got);
    }
}

int main(void)
{
    sk_u32 hi, lo;

    sk_math_init();

    /* 1.0 * 1.0 = 1.0 */
    hi = 0; lo = SK_FIX_ONE;
    chk("1.0 * 1.0", sk_acc_scale(hi, lo, SK_FIX_ONE), SK_FIX_ONE);

    /* 100.0 * 1.0 = 100.0 */
    hi = 0; lo = 100L * SK_FIX_ONE;
    chk("100 * 1.0", sk_acc_scale(hi, lo, SK_FIX_ONE), 100L * SK_FIX_ONE);

    /* 610.0 * 0.004 = 2.44: the case that matters, a big sum times a small
       scale, which is exactly what a matvec does. */
    hi = 0; lo = 610L * SK_FIX_ONE;
    chk("610 * 0.004", sk_acc_scale(hi, lo, 262L),
        sk_fix_mul(610L * SK_FIX_ONE, 262L));

    /* A sum that does not fit Q16.16 at all: 14870.0 */
    hi = 0; lo = 14870L * SK_FIX_ONE;
    chk("14870 * 0.004", sk_acc_scale(hi, lo, 262L),
        sk_fix_mul(14870L * SK_FIX_ONE, 262L));

    /* The same with a 64-bit accumulator: 14870 * 96 = 1427520, which is
       past Q16.16 by itself. */
    hi = 0; lo = 0;
    {
        int i;
        for (i = 0; i < 96; i++)
            sk_acc_add(&hi, &lo, 14870L * SK_FIX_ONE);
        chk("96 * 14870 * 0.004", sk_acc_scale(hi, lo, 262L),
            sk_fix_mul(1427520L * SK_FIX_ONE, 262L));
    }

    /* Negative scale */
    hi = 0; lo = 610L * SK_FIX_ONE;
    chk("610 * -0.004", sk_acc_scale(hi, lo, -262L),
        -sk_fix_mul(610L * SK_FIX_ONE, 262L));

    /* Overflow must saturate, not wrap */
    hi = 0x7FFFFFFFUL; lo = 0xFFFFFFFFUL;
    chk("huge * 1.0 saturates", sk_acc_scale(hi, lo, SK_FIX_ONE), SK_FIX_MAX);

    printf(fails ? "\n%d FAILED\n" : "\nall passed\n", fails);
    return fails ? 1 : 0;
}
