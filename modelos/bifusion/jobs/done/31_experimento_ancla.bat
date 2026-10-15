@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === Pruebas: preprocesado v2 (ancla relativa, etiquetas por ganglio y suaves) y entrenamiento con etiquetas suaves ===
python -m pytest -q tests
if errorlevel 1 (echo ERROR: fallaron las pruebas & exit /b 2)
echo === Ancla absoluta frente a relativa en los 680 casos, ROI de 300 mm (solo CPU, ~30 min) ===
echo Si se corta, relanzarlo sigue donde quedo (runs\experimento_ancla\casos.jsonl).
python -m bifusion3d.tools.experimento_ancla --config configs/siam_full.yaml --down 300 --out runs/experimento_ancla --workers 2
exit /b %errorlevel%
