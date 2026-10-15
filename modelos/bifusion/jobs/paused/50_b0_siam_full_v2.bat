@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === B0: siam_full con ROI 300 mm, ancla relativa y etiquetas por ganglio (unas 15 h) ===
echo Con --resume continua si se corto; con --no-test no mira el test.
python -m bifusion3d.train --config configs/b0_siam_full_v2.yaml --resume --no-test --thermal 83:72
if errorlevel 1 (echo ERROR: entrenamiento & exit /b 11)
python -m bifusion3d.evaluate --run runs/b0_siam_full_v2 --split val --checkpoint best
exit /b %errorlevel%
