@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === Verificacion del preprocesamiento real (load_case) en los 680 casos. Solo CPU, unos 30-40 min ===
echo ROI por la formula en cada caso, imagen, etiqueta, vuelta a la rejilla original y determinismo.
echo Si se corta, relanzarlo continua donde quedo (runs\verificacion\casos.jsonl). No escribe en modelos\mamba.
python -m pytest -q tests/test_verify_preproc.py
if errorlevel 1 (echo ERROR: fallaron las pruebas de la verificacion & exit /b 2)
python -m bifusion3d.tools.verify_preproc --config configs/siam_full.yaml --out runs/verificacion --workers 2
set "RC=%errorlevel%"
echo === Cobertura de la cache rodante en los entrenamientos terminados (lee los runs, no escribe en ellos) ===
python -m bifusion3d.tools.cache_coverage --runs runs/siam_full runs/siam_light ../mamba/runs/full_base_fold0 ../mamba/runs/full_unet ../mamba/runs/exp_wide --out runs/verificacion/cache_entrenamientos.json
exit /b %RC%
