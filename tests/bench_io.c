/* bench_io.c - how expensive is a durable write on this machine?
   The answer decides the checkpoint policy, so it is measured, not assumed. */
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "sk_io.h"

#define N 20

int main(void)
{
    FILE *f;
    clock_t t0, t1;
    int i;
    char buf[64];

    memset(buf, 'x', sizeof(buf));

    t0 = clock();
    for (i = 0; i < N; i++) {
        f = fopen("build/bench.bin", "wb");
        if (!f) return 1;
        fwrite(buf, 1, sizeof(buf), f);
        sk_io_sync(f);
        fclose(f);
    }
    t1 = clock();
    printf("write+sync+close : %.1f ms each\n",
           1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC / N);

    t0 = clock();
    for (i = 0; i < N; i++) {
        f = fopen("build/bench.bin", "wb");
        if (!f) return 1;
        fwrite(buf, 1, sizeof(buf), f);
        fclose(f);
    }
    t1 = clock();
    printf("write+close only : %.1f ms each\n",
           1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC / N);

    t0 = clock();
    for (i = 0; i < N; i++) {
        f = fopen("build/bench2.bin", "wb");
        if (!f) return 1;
        fclose(f);
    }
    t1 = clock();
    printf("open+close only  : %.1f ms each\n",
           1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC / N);

    remove("build/bench.bin");
    remove("build/bench2.bin");
    return 0;
}
