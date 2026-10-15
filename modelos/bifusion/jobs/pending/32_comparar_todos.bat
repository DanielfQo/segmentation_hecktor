@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === Comparacion pareada de siam_full final con todos los modelos (val, 135 casos) ===
for %%R in (../mamba/runs/full_base_fold0 ../mamba/runs/full_unet ../mamba/runs/exp_wide runs/siam_light) do python -m bifusion3d.tool compare_runs --a %%R:final --b runs/siam_full:final --split val
python -m bifusion3d.tool compare_runs --a ../mamba/runs/full_base_fold0:final --b ../mamba/runs/full_unet:final --split val
exit /b 0
