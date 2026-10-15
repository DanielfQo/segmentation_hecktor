@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === pruebas unitarias ===
python -m pytest -q tests
if errorlevel 1 goto fallo
echo === evaluacion rapida (2 casos) con las metricas 2022 y 2025 ===
python -m mambalite3d.tools.evaluate --run runs/full_base_fold0 --split val --checkpoint best --max-cases 2
if errorlevel 1 goto fallo
python -c "import json; d = json.load(open('runs/full_base_fold0/metrics_val_best_primeros2.json')); print('hecktor2025', d['hecktor2025']); print('hecktor2022', d['hecktor2022'])"
if errorlevel 1 goto fallo
del /q runs\full_base_fold0\metrics_val_best_primeros2.json runs\full_base_fold0\per_case_val_best_primeros2.csv runs\full_base_fold0\qc_val_best_primeros2_*.png
echo === reporte ===
python -m mambalite3d.tools.report --models base=runs/full_base_fold0 wide=runs/exp_wide unet=runs/full_unet
if errorlevel 1 goto fallo
exit /b 0
:fallo
echo ERROR: algo fallo; se pausan los trabajos pendientes hasta revisar
move /y jobs\pending\*.bat jobs\paused\ >nul 2>&1
exit /b 1
exit /b %errorlevel%
