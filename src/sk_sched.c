/* sk_sched.c - cooperative step scheduler.  See sk_sched.h.
 *
 * SKYNET M0 / subsystem: core/sched
 */
#include <string.h>
#include "sk_sched.h"

void sk_sched_init(sk_sched *s, sk_arena *arena, sk_u32 mem_budget)
{
    if (s == SK_NULL)
        return;
    memset(s, 0, sizeof(*s));
    s->arena = arena;
    s->mem_budget = mem_budget;
}

int sk_sched_add(sk_sched *s, const char *name, sk_task_fn fn, void *ctx,
                 sk_u32 period, int priority)
{
    sk_task *t;

    if (s == SK_NULL || fn == SK_NULL || s->count >= SK_SCHED_MAX)
        return -1;
    t = &s->tasks[s->count];
    t->name = name;
    t->fn = fn;
    t->ctx = ctx;
    t->period = (period == 0) ? 1UL : period;
    t->next = 0;
    t->runs = 0;
    t->priority = priority;
    t->state = SK_TASK_RUN;
    s->count++;
    return 0;
}

static int sk_sched_find(sk_sched *s, const char *name)
{
    int i;

    if (s == SK_NULL || name == SK_NULL)
        return -1;
    for (i = 0; i < s->count; i++) {
        if (s->tasks[i].name != SK_NULL && strcmp(s->tasks[i].name, name) == 0)
            return i;
    }
    return -1;
}

void sk_sched_pause(sk_sched *s, const char *name)
{
    int i;

    i = sk_sched_find(s, name);
    if (i >= 0 && s->tasks[i].state == SK_TASK_RUN)
        s->tasks[i].state = SK_TASK_PAUSED;
}

void sk_sched_resume(sk_sched *s, const char *name)
{
    int i;

    i = sk_sched_find(s, name);
    if (i >= 0 && s->tasks[i].state == SK_TASK_PAUSED)
        s->tasks[i].state = SK_TASK_RUN;
}

int sk_sched_state(sk_sched *s, const char *name)
{
    int i;

    i = sk_sched_find(s, name);
    return (i < 0) ? SK_TASK_DEAD : s->tasks[i].state;
}

sk_u32 sk_sched_runs(const sk_sched *s, const char *name)
{
    int i;

    if (s == SK_NULL || name == SK_NULL)
        return 0;
    for (i = 0; i < s->count; i++) {
        if (s->tasks[i].name != SK_NULL && strcmp(s->tasks[i].name, name) == 0)
            return s->tasks[i].runs;
    }
    return 0;
}

void sk_sched_tick(sk_sched *s)
{
    int i;

    if (s == SK_NULL)
        return;
    s->tick++;

    for (i = 0; i < s->count; i++) {
        sk_task *t = &s->tasks[i];

        if (t->state != SK_TASK_RUN)
            continue;
        if (t->next > s->tick)
            continue;
        t->fn(t->ctx, s->tick);
        t->runs++;
        t->next = s->tick + t->period;
    }

    /* Memory governor: if the arena is over budget, park the least
       important running step.  It resumes when a snapshot finds room.
       Degrading is survivable; aborting is not. */
    if (s->arena != SK_NULL && s->mem_budget != 0
        && s->arena->used > s->mem_budget) {
        int victim = -1;
        int worst = -1;
        for (i = 0; i < s->count; i++) {
            if (s->tasks[i].state != SK_TASK_RUN)
                continue;
            if (s->tasks[i].priority > worst) {
                worst = s->tasks[i].priority;
                victim = i;
            }
        }
        if (victim >= 0) {
            s->tasks[victim].state = SK_TASK_PAUSED;
            s->starved++;
        }
    } else if (s->arena != SK_NULL && s->mem_budget != 0) {
        /* room again: bring paused steps back, highest priority first */
        for (i = 0; i < s->count; i++) {
            if (s->tasks[i].state == SK_TASK_PAUSED)
                s->tasks[i].state = SK_TASK_RUN;
        }
    }
}
