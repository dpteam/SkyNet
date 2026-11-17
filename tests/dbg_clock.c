#include <stdio.h>
#include "sk_clock.h"

int main(void)
{
    sk_clock c;
    int i;

    sk_clock_init(&c);
    for (i = 0; i < 30; i++) {
        if (!sk_clock_next(&c)) {
            printf("exhausted at %d\n", i);
            break;
        }
        printf("%lu%s\n", (unsigned long)c.value, c.is_super ? " *" : "");
    }
    return 0;
}
