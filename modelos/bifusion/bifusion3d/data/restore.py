"""Vuelta de una prediccion del ROI a la rejilla original de la etiqueta.

mambalite3d.data.io.restore_to_original copia el voxel mas cercano de la clase ya elegida (vuelta "nearest"): con
cortes de 3 mm, los bordes de una estructura pequena quedan en escalon. `restore_probs_linear` interpola las
probabilidades de cada clase y elige la clase despues, en la rejilla original. En un empate gana el primer plano
(probabilidad >= 0,5 frente al fondo), la misma regla que un umbral de 0,5.

Solo se remuestrea la parte de la rejilla original que cubre la caja pedida (el ROI, o una caja menor), asi que no
hace falta memoria para todo el CT.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from mambalite3d.data import io as mio

IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def roi_box(origin_xyz: Sequence[float], spacing_xyz: Sequence[float], shape_zyx: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Caja fisica (lo, hi) en mm (x, y, z) que ocupan los voxeles de una rejilla de ROI (bordes, no centros)."""
    sp = np.asarray(spacing_xyz, dtype=float)
    lo = np.asarray(origin_xyz, dtype=float) - sp / 2
    hi = lo + np.asarray(shape_zyx[::-1], dtype=float) * sp
    return lo, hi


def index_box(ref: sitk.Image, lo_xyz, hi_xyz, pad: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Indices (x, y, z) [lo, hi) de `ref` cuyos voxeles tocan la caja fisica [lo_xyz, hi_xyz]."""
    corners = [(x, y, z) for x in (lo_xyz[0], hi_xyz[0]) for y in (lo_xyz[1], hi_xyz[1]) for z in (lo_xyz[2], hi_xyz[2])]
    idx = np.array([ref.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in c)) for c in corners])
    lo = np.floor(idx.min(axis=0) + 0.5).astype(int) - pad  # el voxel i ocupa [i - 0,5, i + 0,5)
    hi = np.ceil(idx.max(axis=0) + 0.5).astype(int) + pad
    size = np.array(ref.GetSize())
    return np.clip(lo, 0, size), np.clip(hi, 0, size)


def sub_reference(ref: sitk.Image, lo_xyz, hi_xyz) -> sitk.Image:
    """Rejilla con el espaciado y la orientacion de `ref`, limitada a los indices [lo, hi)."""
    sub = sitk.Image([int(h - l) for l, h in zip(lo_xyz, hi_xyz)], sitk.sitkUInt8)
    sub.SetSpacing(ref.GetSpacing())
    sub.SetDirection(ref.GetDirection())
    sub.SetOrigin(ref.TransformIndexToPhysicalPoint(tuple(int(v) for v in lo_xyz)))
    return sub


def _roi_img(arr_zyx: np.ndarray, origin_xyz, spacing_xyz) -> sitk.Image:
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    img.SetOrigin(tuple(float(o) for o in origin_xyz))
    img.SetDirection(IDENTITY)
    return img


def restore_probs_linear(probs: np.ndarray, origin_xyz, spacing_xyz, ref: sitk.Image,
                         box: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """probs (K, Z, Y, X) sobre el ROI -> etiqueta uint8 (Z, Y, X) en la rejilla de `ref`.

    Interpola linealmente las clases 1..K-1; el fondo es 1 menos su suma. Fuera de `box` (por defecto, todo el ROI)
    la etiqueta es 0.
    """
    out = np.zeros(ref.GetSize()[::-1], dtype=np.uint8)
    if box is None:
        box = roi_box(origin_xyz, spacing_xyz, probs.shape[1:])
    lo, hi = index_box(ref, *box)
    if np.any(hi <= lo):
        return out
    sub = sub_reference(ref, lo, hi)
    fg = np.stack([
        sitk.GetArrayFromImage(sitk.Resample(_roi_img(probs[k].astype(np.float32), origin_xyz, spacing_xyz), sub,
                                             sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32))
        for k in range(1, probs.shape[0])
    ])
    background = 1.0 - fg.sum(axis=0)
    lab = np.where(fg.max(axis=0) >= background, fg.argmax(axis=0) + 1, 0).astype(np.uint8)
    out[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = lab
    return out


def restore_label_nearest(label_zyx: np.ndarray, origin_xyz, spacing_xyz, ref: sitk.Image,
                          box: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """Etiqueta uint8 del ROI -> rejilla de `ref` copiando el voxel mas cercano, como mio.restore_to_original,
    pero solo dentro de `box` (por defecto, todo el ROI)."""
    out = np.zeros(ref.GetSize()[::-1], dtype=np.uint8)
    if box is None:
        box = roi_box(origin_xyz, spacing_xyz, label_zyx.shape)
    lo, hi = index_box(ref, *box)
    if np.any(hi <= lo):
        return out
    sub = sub_reference(ref, lo, hi)
    res = sitk.Resample(_roi_img(label_zyx.astype(np.uint8), origin_xyz, spacing_xyz), sub, sitk.Transform(),
                        sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)
    out[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = sitk.GetArrayFromImage(res)
    return out


def one_hot(label_zyx: np.ndarray, num_classes: int = 3) -> np.ndarray:
    return np.stack([(label_zyx == k).astype(np.float32) for k in range(num_classes)])


def restore_case(case, probs: np.ndarray, mode: str = "nearest"):
    """(prediccion, etiqueta de referencia, volumen del voxel en mm3) en la rejilla original de un CaseData.

    mode "nearest" es exactamente mio.restore_to_original sobre la clase elegida en el ROI; "linear" interpola las
    probabilidades.
    """
    if mode == "nearest":
        return mio.restore_to_original(case, probs.argmax(axis=0).astype(np.uint8))
    if mode != "linear":
        raise ValueError(f"vuelta desconocida {mode!r}: usa nearest o linear")
    if case.files is None or case.files.label is None:
        raise ValueError(f"{case.case_id}: no hay etiqueta original con la que evaluar")
    ref = mio.read_lps(case.files.label)
    pred = restore_probs_linear(probs, case.roi_origin_xyz, case.spacing_xyz, ref)
    gt = sitk.GetArrayFromImage(ref).astype(np.uint8)
    return pred, gt, float(np.prod(ref.GetSpacing()))
