@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === siam_light: paso 1, siames en niveles 0-1 con compuerta (unas 11-12 h) ===
echo Mismo protocolo que full_base_fold0. Con --resume continua si se corto; con --no-test no mira el test.
python -m bifusion3d.train --config configs/siam_light.yaml --resume --no-test
if errorlevel 1 (echo ERROR: entrenamiento & exit /b 11)
python -m bifusion3d.evaluate --run runs/siam_light --split val --checkpoint best
exit /b %errorlevel%
