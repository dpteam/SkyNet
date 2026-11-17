# ============================================================
#  MAKEFILE for SKYNET - Universal Version
#  Compatible: TCC(MinGW), GCC, MSVC
# ============================================================

CC = tcc

# Флаги БЕЗ -lm! (TCC сам подключает math)
CFLAGS = -O2 -Wall -I.
LDFLAGS = 

.PHONY: all clean test help

all: train.exe skynet.exe

train.exe: train.c skynet.h
	@echo "[Compiling TRAIN.EXE...]"
	$(CC) $(CFLAGS) -o$@ $<
	@echo "[OK]"

skynet.exe: skynet.c skynet.h
	@echo "[Compiling SKYNET.EXE...]"
	$(CC) $(CFLAGS) -o$@ $<
	@echo "[OK]"

test: train.exe
	.\train.exe

clean:
	-del train.exe 2>nul
	-del skynet.exe 2>nul
	-del *.wgt 2>nul
	-del *.ckp 2>nul
	-del *.best 2>nul

help:
	@echo "SKYNET Build System"
	@echo "Targets: all, test, clean, help"