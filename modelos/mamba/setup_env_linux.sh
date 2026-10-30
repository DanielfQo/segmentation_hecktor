#!/usr/bin/env bash
# Crea (o actualiza) el entorno conda para MambaLite3D en Ubuntu: PyTorch + CUDA, dependencias y,
# si se puede compilar, los kernels de mamba_ssm (el codigo los detecta y los usa solo).
# Uso:  ./setup_env_linux.sh            (entorno mambalite3d_cuda, el que espera cola_experimentos.sh)
#       MAMBALITE_ENV=otro ./setup_env_linux.sh
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${MAMBALITE_ENV:-mambalite3d_cuda}"

if ! command -v conda >/dev/null 2>&1; then
  for raiz in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" /opt/conda; do
    [[ -f "$raiz/etc/profile.d/conda.sh" ]] && { source "$raiz/etc/profile.d/conda.sh"; break; }
  done
fi
command -v conda >/dev/null 2>&1 || { echo "ERROR: no se encontro conda; instala Miniconda primero" >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

nvidia-smi -L || { echo "ERROR: no hay driver NVIDIA (instala nvidia-driver-XXX y reinicia)" >&2; exit 1; }

if [[ -x "$(conda info --base)/envs/$ENV_NAME/bin/python" ]]; then
  echo "El entorno $ENV_NAME ya existe"
else
  echo "Creando entorno $ENV_NAME con Python 3.11 desde conda-forge"
  conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.11 pip
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# PyTorch con CUDA: prueba cu130, luego cu128 y cu126 hasta que la GPU sea visible
for C in cu130 cu128 cu126; do
  if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then break; fi
  echo "Probando PyTorch con $C"
  python -m pip uninstall -y torch >/dev/null 2>&1 || true
  python -m pip install torch --index-url "https://download.pytorch.org/whl/$C"
done
python -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu_ok', ok); sys.exit(0 if ok else 4)"

python -m pip install -r requirements.txt

# Kernels CUDA de Mamba (opcional): aceleran el barrido. Si no compilan, el codigo usa el
# selective scan en PyTorch puro (mismas matematicas, mas lento) y los checkpoints no son compatibles
# entre un entorno con mamba_ssm y uno sin el.
echo "Instalando mamba_ssm (opcional; puede tardar y necesita nvcc)"
python -m pip install ninja packaging
if python -m pip install --no-build-isolation causal-conv1d mamba-ssm; then
  python -c "import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__)"
else
  echo "AVISO: mamba_ssm no se instalo; se usara el barrido en PyTorch puro (mas lento, mismos resultados)"
fi

python -m mambalite3d.tools.check_env
