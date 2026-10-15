@echo off
rem Crea (o actualiza) el entorno conda "mambalite3d" con PyTorch + CUDA y las dependencias.
rem Uso: doble clic o desde una terminal en esta carpeta:  setup_env.bat
setlocal
cd /d "%~dp0"
set "CONDA_ROOT=%USERPROFILE%\miniconda3"
set "ENV_NAME=mambalite3d"
call "%CONDA_ROOT%\Scripts\activate.bat"
if errorlevel 1 (echo ERROR: no se encontro conda en %CONDA_ROOT% & exit /b 1)
if exist "%CONDA_ROOT%\envs\%ENV_NAME%\python.exe" (
  echo El entorno %ENV_NAME% ya existe
) else (
  echo Creando entorno %ENV_NAME% con Python 3.11 desde conda-forge
  call conda create -y -n %ENV_NAME% -c conda-forge --override-channels python=3.11 pip
  if errorlevel 1 (echo ERROR: conda create fallo & exit /b 2)
)
call conda activate %ENV_NAME%
if errorlevel 1 (echo ERROR: no se pudo activar %ENV_NAME% & exit /b 3)
python -m pip install --upgrade pip
rem PyTorch con CUDA: prueba cu130, luego cu128 y cu126 hasta que la GPU sea visible
for %%C in (cu130 cu128 cu126) do (
  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
  if errorlevel 1 (
    echo Probando PyTorch con %%C
    python -m pip uninstall -y torch >nul 2>&1
    python -m pip install torch --index-url https://download.pytorch.org/whl/%%C
  )
)
python -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu_ok', ok); sys.exit(0 if ok else 4)"
if errorlevel 1 (echo ERROR: PyTorch no ve la GPU & exit /b 4)
python -m pip install -r requirements.txt
if errorlevel 1 (echo ERROR: fallo la instalacion de dependencias & exit /b 5)
python -m mambalite3d.tools.check_env
exit /b %errorlevel%
