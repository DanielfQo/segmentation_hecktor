@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === ctx2mm_base: mas contexto (2 x 2 x 3 mm, 144 x 192 x 192 mm) con el base (unas 18 h) ===
echo Mismo experimento que el trabajo 29 de modelos\mamba, pero desde esta cola y sin mirar el test.
python -m bifusion3d.train --config configs/ctx2mm_base.yaml --resume --no-test
if errorlevel 1 (echo ERROR: entrenamiento & exit /b 11)
python -m bifusion3d.evaluate --run runs/ctx2mm_base --split val --checkpoint best
exit /b %errorlevel%
