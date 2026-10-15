#!/usr/bin/env python3
"""Script principal para comparar las arquitecturas de MambaLite3D y baselines en HECKTOR.

Ejecución desde la carpeta 'mamba':
    python comparar_arquitecturas.py
    python comparar_arquitecturas.py --variants base wide unet
    python comparar_arquitecturas.py --variants all
    python comparar_arquitecturas.py --no-plot
    python comparar_arquitecturas.py --calc-ops

Genera:
- Resumen tabular por consola.
- Gráfica comparativa visual en 'runs/comparacion_arquitecturas.png'.
- Reporte detallado en 'runs/comparacion_arquitecturas.md'.
- Datos estructurados en 'runs/comparacion_arquitecturas.json'.
"""
import os
import subprocess
import sys
from pathlib import Path

# Asegurar que el directorio raíz de mamba esté en el path para importar mambalite3d
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from mambalite3d.tools.compare_architectures import main
except ImportError as err:
    # Si falta alguna dependencia (ej. scipy, torch, etc.) y existe el entorno conda mambalite3d, relanzar con él
    user_prof = os.environ.get("USERPROFILE", "")
    candidates = [
        Path(user_prof) / "miniconda3" / "envs" / "mambalite3d" / "python.exe",
        Path(user_prof) / "anaconda3" / "envs" / "mambalite3d" / "python.exe",
    ]
    conda_py = next((c for c in candidates if c.exists()), None)
    if conda_py and Path(sys.executable).resolve() != conda_py.resolve():
        cmd = [str(conda_py), str(Path(__file__).resolve())] + sys.argv[1:]
        sys.exit(subprocess.call(cmd))
    else:
        print(f"[ERROR] No se pudo importar dependencias de mambalite3d: {err}")
        print("Asegúrate de activar el entorno conda 'mambalite3d':")
        print("    conda activate mambalite3d")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
