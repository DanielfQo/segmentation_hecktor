@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set KMP_DUPLICATE_LIB_OK=TRUE
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
if errorlevel 1 (echo ERROR: no se pudo activar mambalite3d & exit /b 1)
echo === U-Net: reanudar si no termino, evaluar si falta ===
if exist runs\full_unet\done.json goto unet_eval
python -m mambalite3d.training.train --config configs/full.yaml --variant unet --name full_unet --resume
if errorlevel 1 (echo ERROR: entrenamiento U-Net & exit /b 11)
:unet_eval
if exist runs\full_unet\metrics_test_best.json goto unet_fin
python -m mambalite3d.tools.evaluate --run runs/full_unet --split val --checkpoint best
if errorlevel 1 exit /b 12
python -m mambalite3d.tools.evaluate --run runs/full_unet --split test --checkpoint best
if errorlevel 1 exit /b 13
:unet_fin
python -m mambalite3d.tools.report --models base=runs/full_base_fold0 wide=runs/exp_wide unet=runs/full_unet
exit /b %errorlevel%
