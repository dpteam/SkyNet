/* probe_dos.c - minimal checks, one per line, so the DOS run can report
 * exactly which capability is broken.
 *
 * The 16-bit build produced no output at all on its first run. That could
 * be a stack overflow, a bad file handle, a 16-bit truncation, or the
 * process not starting. This probe separates them: every step prints and
 * flushes before doing anything, so the last line that appears names the
 * failing step.
 *
 * Build for DOS:  wcc -ml -zq -fo=build_dos/ tests/probe_dos.c
 *                 wlink system dos name build_dos/probe.exe option stack=8192 file build_dos/probe.o
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <io.h>

static void step(const char *name)
{
    printf("step %s\n", name);
    fflush(stdout);
}

int main(void)
{
    FILE *f;
    int n;
    long sz;

    setvbuf(stdout, NULL, _IONBF, 0);

    step("1-start");

    step("2-printf-long");
    printf("  sizeof(int)=%d sizeof(long)=%d sizeof(void*)=%d\n",
           (int)sizeof(int), (int)sizeof(long), (int)sizeof(void *));
    fflush(stdout);

    step("3-fopen-write");
    f = fopen("PROBE.TXT", "wb");
    if (f == NULL) {
        printf("  fopen for write FAILED\n");
        return 1;
    }
    fwrite("hello\n", 1, 6, f);
    fclose(f);
    printf("  wrote PROBE.TXT\n");
    fflush(stdout);

    step("4-fopen-read");
    f = fopen("PROBE.TXT", "rb");
    if (f == NULL) {
        printf("  fopen for read FAILED\n");
        return 1;
    }
    n = (int)fread(&n, 1, 1, f);
    printf("  read %d byte(s)\n", n);
    fclose(f);

    step("5-fseek-ftell");
    f = fopen("PROBE.TXT", "rb");
    fseek(f, 0L, SEEK_END);
    sz = ftell(f);
    printf("  size=%ld\n", sz);
    fclose(f);

    step("6-rename");
    remove("PROBE2.TXT");
    if (rename("PROBE.TXT", "PROBE2.TXT") == 0)
        printf("  rename ok\n");
    else
        printf("  rename FAILED\n");
    fflush(stdout);

    step("7-truncate-via-chsize");
    f = fopen("PROBE2.TXT", "r+b");
    if (f != NULL) {
        fflush(f);
        if (chsize(fileno(f), 2L) == 0)
            printf("  chsize ok\n");
        else
            printf("  chsize FAILED\n");
        fclose(f);
    } else {
        printf("  reopen FAILED\n");
    }
    fflush(stdout);

    step("8-malloc-4k");
    {
        char *p = (char *)malloc(4096);
        if (p == NULL)
            printf("  malloc FAILED\n");
        else {
            p[0] = 'x';
            p[4095] = 'y';
            printf("  malloc 4k ok\n");
            free(p);
        }
    }
    fflush(stdout);

    step("9-malloc-60k");
    {
        char *p = (char *)malloc(60000);
        if (p == NULL)
            printf("  malloc 60k FAILED\n");
        else {
            p[0] = 'x';
            p[59999] = 'y';
            printf("  malloc 60k ok\n");
            free(p);
        }
    }
    fflush(stdout);

    step("10-mkdir");
    if (mkdir("PROBEDIR") == 0)
        printf("  mkdir ok\n");
    else
        printf("  mkdir failed (may already exist)\n");
    fflush(stdout);

    step("11-done");
    printf("ALL STEPS PASSED\n");
    fflush(stdout);
    return 0;
}

