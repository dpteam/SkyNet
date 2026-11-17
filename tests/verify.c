/* verify.c - check that a primes file is the exact prefix of the prime
 * sequence, with no gaps, no duplicates and no non-primes.
 *
 * This is the judge for the crash test.  It uses the same Prime Clock the
 * program under test uses, so the two disagree only if one of them is
 * wrong - and the selftest already pins the clock against known tables.
 */
#include <stdio.h>
#include <stdlib.h>
#include "sk_clock.h"

int main(int argc, char **argv)
{
    sk_clock c;
    FILE *f;
    unsigned long line, n, bad, dups, gaps, want;
    const char *path;

    path = (argc > 1) ? argv[1] : "primes.txt";
    f = fopen(path, "r");
    if (f == NULL) {
        printf("verify: cannot open %s\n", path);
        return 2;
    }

    sk_clock_init(&c);
    line = 0;
    bad = 0;
    dups = 0;
    gaps = 0;
    want = 0;

    while (fscanf(f, "%lu", &n) == 1) {
        line++;
        if (!sk_clock_next(&c)) {
            printf("verify: clock exhausted at line %lu\n", line);
            bad++;
            break;
        }
        want = (unsigned long)c.value;
        if (n != want) {
            if (line > 1 && n < want)
                dups++;
            else
                gaps++;
            bad++;
            if (bad <= 5)
                printf("verify: line %lu has %lu, expected %lu\n", line, n, want);
        }
    }
    fclose(f);

    printf("lines      : %lu\n", line);
    printf("last value : %lu\n", want);
    printf("mismatches : %lu (dups %lu, gaps %lu)\n", bad, dups, gaps);

    if (bad == 0 && line > 0) {
        printf("PASS: exact prefix of the prime sequence\n");
        return 0;
    }
    printf("FAIL\n");
    return 1;
}
