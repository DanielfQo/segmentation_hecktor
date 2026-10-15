"""Evaluacion (y ensambles) con la herramienta de mambalite3d, aceptando las variantes de fusion.

    python -m bifusion3d.evaluate --run runs/siam_light --split val --checkpoint best
    python -m bifusion3d.evaluate --run runs/siam_light --split val --checkpoint final --mirror
    python -m bifusion3d.evaluate --run runs/siam_light --split val --checkpoint final --thermal 83:72   (pausas por calor)

Ensamble / votacion (paso 6): los miembros pueden venir de modelos/mamba (solo se leen sus checkpoints); el resultado
se guarda en runs/ens_<nombre> de esta carpeta. Todos deben compartir split y preprocesamiento.

    python -m bifusion3d.evaluate --ensemble ../mamba/runs/full_base_fold0:best runs/siam_light:final --name base_siam --split val
"""
from __future__ import annotations

import sys

from . import patch, thermal
from .guard import refuse_inside_mamba


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    refuse_inside_mamba(argv, flags=("--run",))
    argv, thermal_cfg = thermal.config_from_argv(argv, None)
    patch.install()
    if thermal_cfg:
        thermal.install(thermal_cfg)
    from mambalite3d.tools import evaluate as base_evaluate

    return base_evaluate.main(argv)


if __name__ == "__main__":
    sys.exit(main())
