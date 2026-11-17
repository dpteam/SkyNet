/* sk_selftest.h - internal checks. */
#ifndef SK_SELFTEST_H
#define SK_SELFTEST_H

#include "sk_types.h"

/* Runs every check.  Returns the number of failures (0 = all good). */
int sk_selftest_run(void);

#endif /* SK_SELFTEST_H */
