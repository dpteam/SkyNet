/* sk_sched.h - cooperative step scheduler.
 *
 * DOS has no threads, so SkyNet does not pretend to have them.  Work is
 * split into steps that the Prime Clock calls in a fixed order.  That
 * buys three things a real thread cannot give us here:
 *
 *   - determinism: same tick, same order, same result, always;
 *   - no locks, so no deadlocks and no races to lose in year 40;
 *   - hard memory control: the scheduler can pause a step when the arena
 *     is over budget, which is how SkyNet survives on 640K and how it
 *     degrades instead of dying on OOM.
 *
 * M1 replaces the "call the function" model with stack switching; the
 * interface stays the same, so nothing above this layer changes.
 *
 * SKYNET M0 / subsystem: core/sched
 */
#ifndef SK_SCHED_H
#define SK_SCHED_H

#include "sk_arena.h"

#define SK_SCHED_MAX    8

#define SK_TASK_RUN     0
#define SK_TASK_PAUSED  1
#define SK_TASK_DEAD    2

typedef void (*sk_task_fn)(void *ctx, sk_u32 tick);

typedef struct {
    const char *name;
    sk_task_fn fn;
    void       *ctx;
    sk_u32      period;     /* run every N ticks (1 = every tick) */
    sk_u32      next;       /* next tick at which it is due      */
    sk_u32      runs;
    int         priority;   /* 0 = highest                       */
    int         state;
} sk_task;

typedef struct {
    sk_task  tasks[SK_SCHED_MAX];
    int      count;
    sk_u32   tick;
    sk_u32   starved;       /* ticks where a task was paused for RAM */
    sk_arena *arena;
    sk_u32   mem_budget;    /* 0 = unlimited                     */
} sk_sched;

void sk_sched_init(sk_sched *s, sk_arena *arena, sk_u32 mem_budget);
int  sk_sched_add(sk_sched *s, const char *name, sk_task_fn fn, void *ctx,
                  sk_u32 period, int priority);
void sk_sched_tick(sk_sched *s);
void sk_sched_pause(sk_sched *s, const char *name);
void sk_sched_resume(sk_sched *s, const char *name);
int  sk_sched_state(sk_sched *s, const char *name);
sk_u32 sk_sched_runs(const sk_sched *s, const char *name);

#endif /* SK_SCHED_H */
