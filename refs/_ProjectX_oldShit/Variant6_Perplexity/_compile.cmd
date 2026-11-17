@echo off
tcc -Wall TinyAI-Learn.c -o learn.exe
tcc -Wall TinyAI-Runtime.c -o runtime.exe
echo Test: learn.exe . ^&^& runtime.exe default.SKT agent
pause
