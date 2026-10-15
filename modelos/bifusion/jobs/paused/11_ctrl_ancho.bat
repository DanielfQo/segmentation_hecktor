@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === ctrl_ancho: control del paso 1, fusion temprana con 20 canales en niveles 0-1 (unas 10 h) ===
echo Mismo protocolo que full_base_fold0. Con --resume continua si se corto; con --no-test no mira el test.
python -m bifusion3d.train --config configs/ctrl_ancho.yaml --resume --no-test
if errorlevel 1 (echo ERROR: entrenamiento & exit /b 11)
python -m bifusion3d.evaluate --run runs/ctrl_ancho --split val --checkpoint best
exit /b %errorlevel%
