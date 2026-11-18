# Portable makefile.  Works with GNU make everywhere, and the flags are
# strict enough that a compiler which is not really C89 will complain.
#
#   make            build ./skynet and build/verify
#   make test       selftest + crash test
#   make 32         build a 32-bit binary (needs gcc-multilib)
#   make clean

CC      ?= cc
CFLAGS  ?= -std=c89 -pedantic -Wall -Wextra -O2
LDFLAGS ?=

SRC = src/main.c src/sk_math.c src/sk_arena.c src/sk_clock.c \
      src/sk_vault.c src/sk_sched.c src/sk_io.c src/sk_mem.c src/sk_tensor.c src/sk_forward.c src/sk_grad.c src/sk_train.c src/sk_shadow.c src/sk_selftest.c

OBJ = $(SRC:.c=.o)

all: skynet build/verify

skynet: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -Isrc -c $< -o $@

build/verify: tests/verify.c src/sk_clock.o
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tests/verify.c src/sk_clock.o

32:
	@mkdir -p build
	$(CC) $(CFLAGS) -m32 -o build/skynet32 $(SRC)
	$(CC) $(CFLAGS) -m32 -Isrc -o build/verify32 tests/verify.c src/sk_clock.c

test: all
	./skynet selftest
	sh tests/crash_test.sh 12

clean:
	rm -f $(OBJ) skynet
	rm -rf build

.PHONY: all test 32 clean
