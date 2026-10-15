@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === final de la prueba corta (el trabajo 04 se corto evaluando el test) ===
python -m bifusion3d.evaluate --run runs/smoke_siam_full --split test --checkpoint final
if errorlevel 1 (echo ERROR: evaluacion de test & exit /b 13)
python -m bifusion3d.tool check_run --run runs/smoke_siam_full --epochs 8
exit /b %errorlevel%
