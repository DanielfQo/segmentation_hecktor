@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === comparacion pareada con la regla fija del plan: checkpoint final de cada run (val, 135 casos) ===
python -m bifusion3d.tool compare_runs --a ../mamba/runs/full_base_fold0:final --b runs/siam_full:final --split val
python -m bifusion3d.tool compare_runs --a runs/siam_light:final --b runs/siam_full:final --split val
exit /b 0
