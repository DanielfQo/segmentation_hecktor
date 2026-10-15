@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === Fase 0: auditoria del preprocesamiento en los 680 casos (solo CPU, de 30 a 60 min) ===
echo Si se corta, relanzarlo continua donde quedo (runs\auditoria\casos.jsonl). No escribe en modelos\mamba.
python -m pytest -q tests/test_audit.py
if errorlevel 1 (echo ERROR: fallaron las pruebas de la auditoria & exit /b 2)
python -m bifusion3d.tools.audit_data --config configs/siam_full.yaml --out runs/auditoria
exit /b %errorlevel%
