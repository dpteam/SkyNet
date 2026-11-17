@echo off
REM === SKYNET DEPLOYMENT SCRIPT ===
REM Для реального IBM PC/XT или эмулятора

echo ========================================
echo   SKYNET AI SYSTEM - Deployment
echo ========================================

REM Шаг 1: Обучение (на современном ПК или быстрой XT)
echo [1/3] Training neural network...
train.exe
if errorlevel 1 goto :train_fail
echo OK - Weights saved to SKYNET.WGT

REM Шаг 2: Копирование на дискету (для реального 8088)
echo [2/3] Copying to floppy...
if exist A:\ copy skynet.com A:\
if exist A:\ copy skynet.wgt A:\
if not exist A:\ echo WARNING: No floppy drive
echo OK - Files ready

REM Шаг 3: Установка агента
echo [3/3] Installing agent...
skynet.com
goto :end

:train_fail
echo ERROR: Training failed!
:end
echo ========================================
echo   Press any key to check status...
echo   (Agent is now resident in memory)
echo   Press ALT-S to see AI decisions
pause > nul