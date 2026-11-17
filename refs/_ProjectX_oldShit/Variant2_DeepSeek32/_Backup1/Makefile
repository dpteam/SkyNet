# Makefile для TinyAI
CC=tcc  # Tiny C Compiler
CFLAGS=-Wall -O1

all: tinyai-learn tinyai-runtime

tinyai-learn: TinyAI-Learn.c
	$(CC) $(CFLAGS) -o tinyai-learn TinyAI-Learn.c -lm

tinyai-runtime: TinyAI-Runtime.c
	$(CC) $(CFLAGS) -o tinyai-runtime TinyAI-Runtime.c -lm

# Для DJGPP (DOS)
djgpp:
	gcc -O1 -o tinyai-learn.exe TinyAI-Learn.c -lm
	gcc -O1 -o tinyai-runtime.exe TinyAI-Runtime.c -lm

clean:
	rm -f tinyai-learn tinyai-runtime *.exe *.o *.SKT