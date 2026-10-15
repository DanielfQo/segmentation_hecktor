@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === reevaluacion de base y wide con las metricas 2025 y 2022 (no se completo el 5 oct) ===
set FALLOS=0
for %%R in (full_base_fold0 exp_wide) do for %%C in (best final) do for %%S in (val test) do (
  python -m mambalite3d.tools.evaluate --run runs/%%R --split %%S --checkpoint %%C
  if errorlevel 1 set FALLOS=1
)
python -m mambalite3d.tools.report --models base=runs/full_base_fold0 wide=runs/exp_wide unet=runs/full_unet
echo === TTA con volteos en validacion (base best, wide final) ===
python -m mambalite3d.tools.evaluate --run runs/full_base_fold0 --split val --checkpoint best --mirror
if errorlevel 1 set FALLOS=1
python -m mambalite3d.tools.evaluate --run runs/exp_wide --split val --checkpoint final --mirror
if errorlevel 1 set FALLOS=1
exit /b %FALLOS%
exit /b %errorlevel%
