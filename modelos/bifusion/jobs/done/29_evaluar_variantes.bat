@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === Evaluacion sin entrenar en validacion: ROI de 270 y 300 mm, vuelta cercana y lineal (GPU, unas 3 h) ===
echo Primero una prueba corta con 2 casos; despues siam_full, siam_light y base. Si se corta, relanzarlo sigue.
python -m bifusion3d.tools.evaluar_variantes --runs runs/siam_full:final --split val --max-cases 2 --out runs/variantes_humo
if errorlevel 1 (echo ERROR: fallo la prueba corta & exit /b 2)
if exist "runs\variantes_humo" rmdir /s /q "runs\variantes_humo"
python -m bifusion3d.tools.evaluar_variantes --runs runs/siam_full:final runs/siam_light:final ../mamba/runs/full_base_fold0:final --split val --out runs/variantes
exit /b %errorlevel%
