@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === tests de bifusion (pasos 1 a 4 y herramientas) ===
python -m pytest -q -s tests
set RC=%errorlevel%
echo === parametros y GMACs (patch 32x128x128) ===
python -m bifusion3d.tool efficiency --variants siam_cga base_bifusion siam_full_bifusion --config configs/siam_cga.yaml --out runs/eficiencia_pasos3y4.json
echo === VRAM de un paso de entrenamiento (patch 32x128x128, batch 2) ===
python -m bifusion3d.tool smoke_vram --variant siam_cga
python -m bifusion3d.tool smoke_vram --variant base_bifusion
python -m bifusion3d.tool smoke_vram --variant siam_full_bifusion
exit /b %RC%
