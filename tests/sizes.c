/* sizes.c - what the platform actually gives us.
 *
 * The fixed-point code assumes sk_i32 is exactly 32 bits. On Linux x86-64
 * `long` is 64 bits, so that assumption is false there and the arithmetic
 * silently behaves differently from Windows and DOS. This prints the truth
 * instead of trusting the typedef. */
#include <stdio.h>
#include "sk_types.h"
#include "sk_tensor.h"
#include "sk_forward.h"

int main(void)
{
    printf("char=%d short=%d int=%d long=%d\n",
           (int)sizeof(char), (int)sizeof(short),
           (int)sizeof(int), (int)sizeof(long));
    printf("sk_u32=%d sk_i32=%d sk_fix=%d\n",
           (int)sizeof(sk_u32), (int)sizeof(sk_i32), (int)sizeof(sk_fix));
    printf("SK_FIX_MAX=%ld SK_FIX_MIN=%ld\n",
           (long)SK_FIX_MAX, (long)SK_FIX_MIN);

    /* Does the type actually behave as 32 bits? */
    {
        sk_i32 v = (sk_i32)0x7FFFFFFFUL;
        printf("(sk_i32)0x7FFFFFFF = %ld, +1 = %ld\n",
               (long)v, (long)(v + 1));
    }
    /* The model struct must stay small: on DOS the whole data segment is
       64 KB and the stack is 8 KB, so anything sized by SK_T_MAX* has to
       live in the pool, not in the struct. */
    printf("sk_model=%d sk_table=%d sk_pool=%d sk_ctx=%d\n",
           (int)sizeof(sk_model), (int)sizeof(sk_table), (int)sizeof(sk_pool), (int)sizeof(sk_ctx));
    return 0;
}
