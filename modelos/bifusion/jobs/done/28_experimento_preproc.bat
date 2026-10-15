@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === Pruebas: remuestreo de etiquetas, vuelta lineal, escala del PET y herramientas de los experimentos ===
python -m pytest -q tests/test_labels.py tests/test_restore.py tests/test_pet_scale.py tests/test_experimento_preproc.py tests/test_evaluar_variantes.py tests/test_thermal.py
if errorlevel 1 (echo ERROR: fallaron las pruebas & exit /b 2)
echo === Experimento sin entrenar en los 680 casos: escala del PET, ROI de 270/300/330 mm y etiquetas (solo CPU, 1-2 h) ===
echo Si se corta, relanzarlo sigue donde quedo (runs\experimento_preproc\casos.jsonl).
python -m bifusion3d.tools.experimento_preproc --config configs/siam_full.yaml --out runs/experimento_preproc --workers 2
exit /b %errorlevel%
