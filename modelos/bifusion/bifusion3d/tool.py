"""Lanzador de las herramientas de mambalite3d con las variantes de fusion disponibles.

    python -m bifusion3d.tool <herramienta> [argumentos de esa herramienta]

Herramientas:
  train, evaluate, cross_validate          versiones de bifusion3d (ver sus modulos)
  efficiency                               parametros, GMACs, VRAM y tiempos (acepta --variants y --models)
  smoke_vram                               VRAM pico y s/iteracion de un paso de entrenamiento (--variant)
  analyze, tune_postprocess                componentes y post-procesado de un run de esta carpeta
  report, compare_runs                     tablas y comparacion pareada (escriben en runs/ de esta carpeta)
  check_env, check_run                     comprobaciones del entorno y de un run

Con --thermal 83:72 cualquier herramienta que use la GPU pausa si se calienta (ver bifusion3d/thermal.py).

Ejemplos:
  python -m bifusion3d.tool efficiency --variants base siam_light --config configs/siam_light.yaml
  python -m bifusion3d.tool smoke_vram --variant siam_light
  python -m bifusion3d.tool compare_runs --a ../mamba/runs/full_base_fold0:best --b runs/siam_light:final --split val
"""
from __future__ import annotations

import importlib
import sys

from . import patch, thermal
from .guard import refuse_inside_mamba

TOOLS = {
    "train": "bifusion3d.train",
    "evaluate": "bifusion3d.evaluate",
    "cross_validate": "bifusion3d.cross_validate",
    "efficiency": "mambalite3d.tools.efficiency",
    "smoke_vram": "mambalite3d.tools.smoke_vram",
    "analyze": "mambalite3d.tools.analyze",
    "tune_postprocess": "mambalite3d.tools.tune_postprocess",
    "report": "mambalite3d.tools.report",
    "compare_runs": "mambalite3d.tools.compare_runs",
    "check_env": "mambalite3d.tools.check_env",
    "check_run": "mambalite3d.tools.check_run",
}
WRITES_INTO_RUN = {"analyze", "tune_postprocess"}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in TOOLS:
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    name, rest = argv[0], argv[1:]
    refuse_inside_mamba(rest, flags=("--run",) if name in WRITES_INTO_RUN else ())
    rest, thermal_cfg = thermal.config_from_argv(rest, None)
    patch.install()
    if thermal_cfg and name not in ("check_env",):
        thermal.install(thermal_cfg)
    module = importlib.import_module(TOOLS[name])
    if name == "check_env":  # su main() no recibe argumentos
        return module.main()
    return module.main(rest)


if __name__ == "__main__":
    sys.exit(main())
