@echo off
REM === SKYNET DEPLOYMENT SCRIPT ===

echo ==========================================
echo   SKYNET AI SYSTEM - Pure C Edition
echo   For Intel 8088 (IBM PC/XT)
echo ==========================================

REM ---- Шаг 1: Скомпилировать ----
echo [1/4] Compiling...
call make all
if errorlevel 1 goto :error_compile
echo OK

REM ---- Шаг 2: Обучить нейросеть ----
echo.
echo [2/4] Training neural network...
echo (This takes 5-30 minutes on real 8088)
echo (Or run in DOSBox for speed)
train.exe
if errorlevel 1 goto :error_train
echo Weights saved to SKYNET.WGT

REM ---- Шаг 3: Проверить результат ----
echo.
echo [3/4] Testing inference...
dir skynet.wgt
echo File size should be ~300-500 bytes

REM ---- Шаг 4: Установить агента ----
echo.
echo [4/4] Installing AI agent...
skynet.com
if errorlevel 1 goto :error_install

echo.
echo ==========================================
echo   SUCCESS!
echo ==========================================
echo.
echo The agent is now resident in memory.
echo It will "think" every ~2 seconds.
echo Press ALT-S to see what it's deciding.
echo.
goto :end

:error_compile
echo ERROR: Compilation failed!
goto :end

:error_train
echo ERROR: Training failed!
goto :end

:error_install
echo ERROR: Agent installation failed!
goto :end

:end
pause