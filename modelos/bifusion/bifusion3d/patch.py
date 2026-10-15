"""Hace que el codigo de mambalite3d construya tambien las variantes de fusion.

Las herramientas de mambalite3d llaman a `build_model(variant, ...)`, que importan al cargarse. `install()` sustituye
esa referencia, solo en memoria y solo en este proceso, por `bifusion3d.network.build_model`, que construye las
variantes de fusion y delega todas las demas (base, wide, unet...) en MambaLite3D exactamente como antes. Ningun
archivo de modelos/mamba se modifica.

Tambien instala el preprocesado v2 (bifusion3d.data.preproc): con un config sin roi.ancla ni data.etiquetas, carga
los casos exactamente igual que mambalite3d.
"""
from __future__ import annotations

import importlib

from . import network

TARGETS = (
    "mambalite3d.models.network",
    "mambalite3d.training.train",
    "mambalite3d.tools.evaluate",
    "mambalite3d.tools.analyze",
    "mambalite3d.tools.efficiency",
    "mambalite3d.tools.smoke_vram",
)


def install():
    from .data import preproc

    preproc.install()  # opciones de preprocesado v2 (apagadas si el config no las pide: igual que mambalite3d)
    for name in TARGETS:
        module = importlib.import_module(name)
        module.build_model = network.build_model
    return network.build_model
