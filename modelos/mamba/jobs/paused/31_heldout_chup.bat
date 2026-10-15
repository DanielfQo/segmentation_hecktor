@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === generalizacion: CHUP fuera del entrenamiento (base) ===
python -m mambalite3d.training.train --config configs/full_heldout_chup.yaml --resume
if errorlevel 1 (echo ERROR: entrenamiento & exit /b 11)
python -m mambalite3d.tools.evaluate --run runs/full_base_heldout_chup --split val --checkpoint best
if errorlevel 1 exit /b 12
python -m mambalite3d.tools.evaluate --run runs/full_base_heldout_chup --split test --checkpoint best
exit /b %errorlevel%
