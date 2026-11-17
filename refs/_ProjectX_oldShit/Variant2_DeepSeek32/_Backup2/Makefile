# Makefile для TinyAI (Tiny C Compiler)
CC=tcc
CFLAGS=-Wall -O1

all: tinyai-learn.exe tinyai-runtime.exe

tinyai-learn.exe: TinyAI-Learn.c tinyml.h
	$(CC) $(CFLAGS) -o tinyai-learn.exe TinyAI-Learn.c -lm

tinyai-runtime.exe: TinyAI-Runtime.c tinyml.h
	$(CC) $(CFLAGS) -o tinyai-runtime.exe TinyAI-Runtime.c -lm

clean:
	del tinyai-learn.exe tinyai-runtime.exe *.o *.SKT 2>nul

# Для DJGPP (если нужно)
djgpp:
	gcc -O1 -o tinyai-learn-djgpp.exe TinyAI-Learn.c -lm
	gcc -O1 -o tinyai-runtime-djgpp.exe TinyAI-Runtime.c -lm