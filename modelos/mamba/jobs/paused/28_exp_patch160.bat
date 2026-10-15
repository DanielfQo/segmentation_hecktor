@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === mas contexto: patch 48x160x160 (base) ===
python -m mambalite3d.training.train --config configs/exp_patch160.yaml --resume
if errorlevel 1 (echo ERROR: entrenamiento & exit /b 11)
python -m mambalite3d.tools.evaluate --run runs/exp_patch160 --split val --checkpoint best
if errorlevel 1 exit /b 12
python -m mambalite3d.tools.evaluate --run runs/exp_patch160 --split test --checkpoint best
exit /b %errorlevel%
