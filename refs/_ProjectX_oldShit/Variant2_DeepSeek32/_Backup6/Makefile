# Makefile для TinyAI (без математической библиотеки)
CC=tcc
CFLAGS=-Wall

all: tinyai-learn.exe tinyai-runtime.exe

tinyai-learn.exe: TinyAI-Learn.c tinyml.c tinyml.h
	$(CC) $(CFLAGS) -o tinyai-learn.exe tinyml.c TinyAI-Learn.c

tinyai-runtime.exe: TinyAI-Runtime.c tinyml.c tinyml.h
	$(CC) $(CFLAGS) -o tinyai-runtime.exe tinyml.c TinyAI-Runtime.c

clean:
	del tinyai-learn.exe tinyai-runtime.exe *.o *.SKT 2>nul