"""Deteccion de un PET que no esta en SUV y factor para corregirlo, sin sacar el caso del entrenamiento.

En SUV, el tejido blando del cuello (musculo, glandulas sin tumor) capta poco y de forma estable: su mediana ronda
0,7-0,9. Se mide la mediana del PET en los voxeles del cuerpo cuyo CT esta entre 0 y 100 HU, sobre la misma
rejilla gruesa de 4 mm que usa mio.compute_roi, con un cuerpo definido solo por el CT (asi la medida no depende de
la escala del PET). Si sale fuera de `rango`, el PET no esta en SUV y se multiplica por `referencia / mediana`.

Como segunda medida, independiente, se guarda el percentil 99 del PET en los 60 mm mas craneales del cuerpo (el
cerebro, cuando esta en el campo): en SUV ronda 8-12.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
import SimpleITK as sitk

from mambalite3d.data import io as mio

SOFT_HU = (0.0, 100.0)
RANGO = (0.3, 3.0)
REFERENCIA = 0.76  # mediana del tejido blando medida en los casos de CHUM; el experimento la recalcula con todos
CRANEAL_MM = 60.0


def _coarse(ct: sitk.Image, pet: sitk.Image):
    lo, hi = mio._physical_bounds(ct)
    size = np.maximum(np.ceil((hi - lo) / mio.COARSE_MM).astype(int) + 1, 1)
    ref = mio._grid(lo, (mio.COARSE_MM,) * 3, size)
    ct_c = sitk.GetArrayFromImage(mio._resample(ct, ref, sitk.sitkLinear, -1024, sitk.sitkFloat32))
    pet_c = sitk.GetArrayFromImage(mio._resample(pet, ref, sitk.sitkLinear, 0, sitk.sitkFloat32))
    return ct_c, pet_c


def pet_stats(ct: sitk.Image, pet: sitk.Image, roi_cfg: dict) -> Dict[str, float]:
    """Mediana del PET en el tejido blando y percentil 99 en la zona craneal, sobre la rejilla de 4 mm."""
    ct_c, pet_c = _coarse(ct, pet)
    body = ct_c > roi_cfg["body_hu"]
    if roi_cfg.get("main_body", True):
        body = mio.main_body(body, roi_cfg.get("open_iters", 2))
    soft = body & (ct_c >= SOFT_HU[0]) & (ct_c <= SOFT_HU[1])
    out = {"suv_tejido": float(np.median(pet_c[soft])) if soft.any() else float("nan"),
           "voxeles_tejido": int(soft.sum()), "suv_craneal_p99": float("nan")}
    zs = np.nonzero(body.any(axis=(1, 2)))[0]
    if len(zs):
        top = int(zs.max())  # LPS: +z es craneal
        n = int(round(CRANEAL_MM / mio.COARSE_MM))
        band = body[max(0, top - n + 1): top + 1]
        vals = pet_c[max(0, top - n + 1): top + 1][band]
        if vals.size:
            out["suv_craneal_p99"] = float(np.percentile(vals, 99))
    return out


def scale_factor(suv_tejido: float, referencia: float = REFERENCIA, rango: Sequence[float] = RANGO) -> float:
    """1 si la mediana del tejido blando es plausible en SUV; si no, referencia / mediana."""
    if not math.isfinite(suv_tejido) or suv_tejido <= 0:
        return 1.0
    if rango[0] <= suv_tejido <= rango[1]:
        return 1.0
    return float(referencia / suv_tejido)


def scaled(pet: sitk.Image, factor: float) -> sitk.Image:
    """El PET multiplicado por `factor`, con la misma geometria."""
    if factor == 1.0:
        return pet
    out = sitk.GetImageFromArray((sitk.GetArrayFromImage(pet).astype(np.float32) * np.float32(factor)))
    out.CopyInformation(pet)
    return out
