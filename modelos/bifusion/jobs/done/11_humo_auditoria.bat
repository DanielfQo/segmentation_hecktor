@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === Fase 0, prueba corta: pruebas de la auditoria y 9 casos reales (solo CPU, unos 2 min) ===
echo Va antes que siam_full para revisar la herramienta mientras entrena. No escribe en modelos\mamba.
python -m pytest -q tests/test_audit.py
if errorlevel 1 (echo ERROR: fallaron las pruebas de la auditoria & exit /b 2)
python -m bifusion3d.tools.audit_data --config configs/siam_full.yaml --out runs/auditoria_humo --workers 2 --cases CHUM-001 CHUP-000 CHUS-003 HGJ-007 HMR-001 MDA-001 USZ-001 MDA-075 MDA-247
exit /b %errorlevel%
