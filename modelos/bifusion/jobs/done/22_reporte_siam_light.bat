@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === siam_light frente al base en validacion (lee el run de mamba sin escribir en el) ===
python -m bifusion3d.tool report --models base=../mamba/runs/full_base_fold0 siam_light=runs/siam_light --out runs/resultados_siam_light
if errorlevel 1 exit /b 1
python -m bifusion3d.tool compare_runs --a ../mamba/runs/full_base_fold0:best --b runs/siam_light:best --split val
exit /b %errorlevel%
