"""Fusion bimodal PET/CT para MambaLite3D, sin modificar modelos/mamba.

Este paquete reutiliza `mambalite3d` (modelos/mamba) en modo de solo lectura: lectura de datos, ROI, muestreo,
augmentation, perdida, entrenamiento, inferencia, metricas y herramientas son exactamente los mismos; solo cambia la
red. Importarlo:

- anade modelos/mamba a sys.path para poder importar `mambalite3d`;
- desactiva la escritura de archivos .pyc, para no crear ni actualizar nada dentro de modelos/mamba.

La ruta a modelos/mamba se puede cambiar con la variable de entorno BIFUSION_MAMBA_DIR.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # nada de __pycache__ nuevos dentro de modelos/mamba

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_HERE)  # modelos/bifusion
MAMBA_DIR = os.path.normpath(os.environ.get("BIFUSION_MAMBA_DIR") or os.path.join(PROJECT_DIR, "..", "mamba"))

if not os.path.isdir(os.path.join(MAMBA_DIR, "mambalite3d")):
    raise ImportError(
        f"no se encontro el paquete mambalite3d en {MAMBA_DIR}; "
        "define BIFUSION_MAMBA_DIR con la ruta de la carpeta modelos/mamba"
    )
if MAMBA_DIR not in sys.path:
    sys.path.insert(0, MAMBA_DIR)

__version__ = "0.1.0"


def inside_mamba(path) -> bool:
    """True si `path` (absoluta o relativa al directorio actual) esta dentro de modelos/mamba."""
    p = os.path.normcase(os.path.realpath(os.path.abspath(str(path))))
    root = os.path.normcase(os.path.realpath(MAMBA_DIR))
    return p == root or p.startswith(root + os.sep)
