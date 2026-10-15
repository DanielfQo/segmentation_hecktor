@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === Pruebas: verificacion del preprocesamiento, cache rodante, deteccion por grupos y guardia termica (CPU, unos minutos) ===
nvidia-smi --query-gpu=name,temperature.gpu --format=csv
python -m pytest -q tests/test_verify_preproc.py tests/test_cache_coverage.py tests/test_eval_groups.py tests/test_thermal.py tests/test_audit.py
exit /b %errorlevel%
