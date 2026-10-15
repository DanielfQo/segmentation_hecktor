@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === siam_full: paso 2, siames completo con Cross/Concat Mamba (unas 16 h: 14,5 de entrenamiento y 1 de evaluacion) ===
echo Mismo protocolo que full_base_fold0. Con --resume continua si se corto; con --no-test no mira el test.
echo Python corre con Ctrl+C desactivado (start /b): un Ctrl+C en esta ventana ya no lo detiene.
start "" /b /wait python -m bifusion3d.train --config configs/siam_full.yaml --resume --no-test
if errorlevel 1 (echo ERROR: entrenamiento & exit /b 11)
start "" /b /wait python -m bifusion3d.evaluate --run runs/siam_full --split val --checkpoint best
exit /b %errorlevel%
