# SKYNET Makefile for 8088 development
# Requires: TASM/MASM or NASM, Turbo C++ 2.0

# Компиляторы
CC = tcc
AS = tasm
LD = tlink

# Флаги для Tiny Model (COM файл)
CFLAGS = -mt -O2 -Z -G      # -mt = tiny model
AFLAGS = /m2 /t             # /t = COM формат

all: train.exe skynet.com

# === TRAIN.EXE (обучалка на C) ===
train.exe: skynet-train.c
	$(CC) $(CFLAGS) -etrain.exe skynet-train.c

# === SKYNET.COM (агент на ASM) ===
skynet.obj: skynet.asm
	$(AS) $(AFLAGS) skynet.asm

skynet.com: skynet.obj
	$(LD) /t skynet.obj,,NUL,,    # /t = COM формат

# === Запуск ===
run: skynet.com
	copy skynet.com a:            # На дискету!
	a:\skynet.com

# === Очистка ===
clean:
	del *.obj
	del *.exe
	del *.com
	del *.wgt
	del *.ckp

# === Тестирование в DOSBox ===
dosbox:
	dosbox -c "mount c ." -c "c:" -c "train.exe"