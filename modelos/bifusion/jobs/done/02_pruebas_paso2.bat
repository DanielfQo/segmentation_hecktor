@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === tests de bifusion (pasos 1 y 2) ===
python -m pytest -q -s tests
set RC=%errorlevel%
echo === parametros y GMACs (patch 32x128x128) ===
python -m bifusion3d.tool efficiency --variants siam_full siam_gate --config configs/siam_full.yaml --out runs/eficiencia_paso2.json
python -m bifusion3d.tool efficiency --variants base --config configs/ctrl_ancho.yaml --out runs/eficiencia_ctrl_ancho.json
echo === VRAM de un paso de entrenamiento (patch 32x128x128, batch 2) ===
python -m bifusion3d.tool smoke_vram --variant siam_full
python -m bifusion3d.tool smoke_vram --variant siam_gate
python -m bifusion3d.tool smoke_vram --variant base --features 20 20 32 48 64 96
exit /b %RC%
