"""Protecciones para no escribir nunca dentro de modelos/mamba."""
from __future__ import annotations

import os
from typing import Sequence

from . import MAMBA_DIR, inside_mamba


def refuse_inside_mamba(argv: Sequence[str], flags: Sequence[str] = ("--run",), check_cwd: bool = True) -> None:
    """Sale con un mensaje si el directorio actual o la carpeta de un flag (p. ej. --run) esta dentro de modelos/mamba.

    Las herramientas escriben sus resultados relativos al directorio actual (runs/...) o dentro de la carpeta del run
    que reciben con --run, asi que ambas cosas deben quedar fuera de modelos/mamba. Leer de modelos/mamba (por ejemplo
    los miembros de un ensamble o los checkpoints de efficiency) si esta permitido.
    """
    if check_cwd and inside_mamba(os.getcwd()):
        raise SystemExit(
            f"ejecuta los comandos de bifusion desde modelos/bifusion, no desde {os.getcwd()} "
            "(los resultados se escriben relativos al directorio actual)"
        )
    argv = list(argv)
    for i, arg in enumerate(argv):
        for flag in flags:
            value = None
            if arg == flag and i + 1 < len(argv):
                value = argv[i + 1]
            elif arg.startswith(flag + "="):
                value = arg.split("=", 1)[1]
            if value is not None and inside_mamba(value):
                raise SystemExit(
                    f"{flag} {value} esta dentro de {MAMBA_DIR}: esta herramienta escribe en la carpeta del run. "
                    "Evalua los runs de mamba desde modelos/mamba con mambalite3d.tools"
                )
