# Cabecera comun de los trabajos .sh: activa el entorno conda mambalite3d.
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh" || { echo "ERROR: no se encontro conda"; exit 1; }
conda activate mambalite3d || { echo "ERROR: no se pudo activar mambalite3d"; exit 1; }
