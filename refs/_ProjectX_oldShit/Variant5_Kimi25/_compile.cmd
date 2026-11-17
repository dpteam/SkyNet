@echo off
:: TinyC
tcc learn.c -o learn.exe
tcc runtime.c -o runtime.exe

:: DJGPP (DOS)
:: gcc learn.c -o learn.exe -Os -s -std=c89
:: gcc runtime.c -o runtime.exe -Os -s -std=c89

:: MinGW (Windows XP+)
:: gcc learn.c -o learn.exe -Os -s -m32 -std=c89
:: gcc runtime.c -o runtime.exe -Os -s -m32 -std=c89
pause