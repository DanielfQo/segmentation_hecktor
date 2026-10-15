@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === Prueba de la guardia termica corregida (con torch) ===
python -m pytest -q tests/test_thermal.py
echo === Viabilidad de AnisoCrossNeXt-3D: VRAM y segundos por iteracion en la RTX 3050 (unos 5 min de GPU, datos sinteticos) ===
python -m bifusion3d.tools.viabilidad_aniso --out runs/viabilidad_aniso
exit /b %errorlevel%
