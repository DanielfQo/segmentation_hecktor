"""Remuestreo de la etiqueta a la rejilla de trabajo: el actual, ganglio por ganglio y suave.

- nearest: lo que hace mambalite3d, el voxel mas cercano de la etiqueta entera. Con cortes de 3 mm, dos ganglios a
  menos de un corte se funden, un ganglio que cruza en diagonal puede quedar partido y uno muy pequeno puede
  desaparecer.
- nodos (`resample_nodes`): cada componente conexo de la etiqueta original (26-vecindad, por clase) se remuestrea
  con su propio identificador. Despues, en la rejilla nueva:
    1. un componente que desaparecio recupera el voxel de su centro (rescate);
    2. uno que quedo partido recupera los voxeles de fondo que el componente original atraviesa, sin tocar a otro
       componente de su clase (reconexion);
    3. donde dos componentes distintos de la misma clase quedan pegados, el de mas volumen cede su capa de contacto.
  Asi la etiqueta conserva el numero de ganglios del original. Componentes de clases distintas no se separan.
- suave (`soft_fractions`): fraccion de cada voxel que ocupa cada clase en la etiqueta original, medida con el voxel
  mas cercano en una rejilla mas fina (por defecto 3 x 2 x 2 subvoxeles en z, y, x), solo en la caja de la etiqueta.

Las funciones reciben la etiqueta original como imagen de SimpleITK (en LPS, como mio.read_lps) y la rejilla de
trabajo `ref` (direccion identidad, como la crea mio._grid) y devuelven arrays numpy (z, y, x).
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

STRUCT_26 = np.ones((3, 3, 3), dtype=bool)
BASE = 100000  # identificador de componente = clase * BASE + k
CLASSES = (1, 2)
FACTORS_ZYX = (3, 2, 2)


def _nearest(img: sitk.Image, ref: sitk.Image, pixel_type) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.Resample(img, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, pixel_type))


def resample_nearest(lab_img: sitk.Image, ref: sitk.Image) -> np.ndarray:
    """La etiqueta en la rejilla de trabajo, exactamente como mio.load_case."""
    return _nearest(lab_img, ref, sitk.sitkUInt8)


def count_components(lab: np.ndarray, cls: int) -> int:
    return int(ndimage.label(lab == cls, structure=STRUCT_26)[1])


def node_ids(arr: np.ndarray) -> np.ndarray:
    """Un identificador por componente conexo de cada clase: clase * BASE + k, con k creciente con el volumen
    (el componente mayor de cada clase tiene el identificador mas alto)."""
    ids = np.zeros(arr.shape, dtype=np.int32)
    for c in CLASSES:
        lab, n = ndimage.label(arr == c, structure=STRUCT_26)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        rank = np.empty(n, dtype=np.int32)
        rank[np.argsort(sizes, kind="stable")] = np.arange(1, n + 1, dtype=np.int32)
        fg = lab > 0
        ids[fg] = c * BASE + rank[lab[fg] - 1]
    return ids


def crop_to_label(lab_img: sitk.Image, pad: int = 1) -> sitk.Image:
    """La etiqueta recortada a su caja (mas `pad` voxeles): fuera de ella todo es fondo, asi que remuestrear el
    recorte da lo mismo que remuestrear la imagen entera, y es mucho mas rapido."""
    arr = sitk.GetArrayFromImage(lab_img)
    nz = np.argwhere(arr > 0)
    if len(nz) == 0:
        return lab_img
    lo = np.maximum(nz.min(axis=0) - pad, 0)[::-1]
    hi = np.minimum(nz.max(axis=0) + pad + 1, arr.shape)[::-1]
    return sitk.RegionOfInterest(lab_img, [int(v) for v in hi - lo], [int(v) for v in lo])


def _bbox(mask: np.ndarray, pad: int) -> Tuple[slice, ...]:
    nz = np.argwhere(mask)
    lo = np.maximum(nz.min(axis=0) - pad, 0)
    hi = np.minimum(nz.max(axis=0) + pad + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def fractions(img: sitk.Image, ref: sitk.Image, lo_xyz, hi_xyz, values: Sequence[int],
              factors_zyx: Sequence[int] = FACTORS_ZYX) -> np.ndarray:
    """(len(values), nz, ny, nx): fraccion de cada voxel de ref[lo:hi] (indices x, y, z) cuyo valor en `img` es cada
    uno de `values`, medida con el voxel mas cercano en una rejilla `factors_zyx` veces mas fina."""
    lo, hi = np.asarray(lo_xyz, dtype=int), np.asarray(hi_xyz, dtype=int)
    f_xyz = np.array(factors_zyx[::-1], dtype=int)
    spacing = np.array(ref.GetSpacing(), dtype=float)
    fine_sp = spacing / f_xyz
    edge = np.array(ref.TransformIndexToPhysicalPoint(tuple(int(v) for v in lo))) - spacing / 2
    fine = sitk.Image([int(v) for v in (hi - lo) * f_xyz], sitk.sitkUInt8)
    fine.SetSpacing(tuple(float(s) for s in fine_sp))
    fine.SetOrigin(tuple(float(v) for v in edge + fine_sp / 2))
    fine.SetDirection(ref.GetDirection())
    pixel = sitk.sitkInt32 if max(values) > 255 else sitk.sitkUInt8
    s = _nearest(img, fine, pixel)
    nzb, nyb, nxb = (hi - lo)[::-1]
    fz, fy, fx = factors_zyx
    return np.stack([(s == v).reshape(nzb, fz, nyb, fy, nxb, fx).mean(axis=(1, 3, 5)) for v in values]).astype(np.float32)


def _rescue(new: np.ndarray, orig_ids: np.ndarray, orig_img: sitk.Image, ref: sitk.Image) -> int:
    """Componentes que la rejilla nueva hizo desaparecer: su voxel mas cercano al centro vuelve a ser suyo."""
    present = set(np.unique(new[new > 0]).tolist())
    lost = [int(i) for i in np.unique(orig_ids[orig_ids > 0]) if int(i) not in present]
    if not lost:
        return 0
    centres = ndimage.center_of_mass(orig_ids > 0, orig_ids, lost)
    size = np.array(ref.GetSize())
    rescued = 0
    for i, centre in zip(lost, centres):
        phys = orig_img.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in centre[::-1]))
        idx = np.floor(np.array(ref.TransformPhysicalPointToContinuousIndex(phys)) + 0.5).astype(int)
        if np.all(idx >= 0) and np.all(idx < size):
            z, y, x = int(idx[2]), int(idx[1]), int(idx[0])
            if new[z, y, x] == 0:
                new[z, y, x] = i
                rescued += 1
    return rescued


def _reconnect(new: np.ndarray, ids_img: sitk.Image, ref: sitk.Image, cls: int) -> int:
    """Cada componente de la clase que la rejilla nueva dejo en trozos recupera los voxeles de fondo que el
    componente original atraviesa (fraccion > 0), salvo los que tocan a otro componente de su clase. Un conjunto
    conexo atraviesa voxeles conexos, asi que eso lo reconecta salvo que lo impida un vecino. Modifica `new`;
    devuelve cuantos siguen partidos."""
    mask = (new // BASE) == cls
    if not mask.any():
        return 0
    still = 0
    for i in np.unique(new[mask]):
        own = new == i
        if ndimage.label(own, structure=STRUCT_26)[1] <= 1:
            continue
        sl = _bbox(own, 2)
        lo = [s.start for s in sl][::-1]
        hi = [s.stop for s in sl][::-1]
        touched = fractions(ids_img, ref, lo, hi, (int(i),))[0] > 0
        sub = new[sl]
        others = ((sub // BASE) == cls) & (sub != i)
        add = touched & (sub == 0) & ~ndimage.binary_dilation(others, structure=STRUCT_26)
        sub[add] = i
        if ndimage.label(sub == i, structure=STRUCT_26)[1] > 1:
            still += 1
    return still


def _separate(ids: np.ndarray, cls: int) -> int:
    """Donde dos componentes distintos de la clase se tocan (26-vecindad), el de identificador mayor (el de mas
    volumen) cede los voxeles de contacto. Modifica `ids`; devuelve cuantos voxeles se quitaron."""
    mask = (ids // BASE) == cls
    if not mask.any():
        return 0
    sl = _bbox(mask, 1)
    sub = ids[sl]
    m = (sub // BASE) == cls
    big = np.iinfo(np.int32).max
    own = np.where(m, sub, big)
    lowest = ndimage.minimum_filter(own, footprint=STRUCT_26, mode="constant", cval=big)
    touch = m & (lowest < sub)
    sub[touch] = 0
    return int(touch.sum())


def resample_nodes(lab_img: sitk.Image, ref: sitk.Image) -> Tuple[np.ndarray, Dict[str, int]]:
    """Etiqueta en la rejilla de trabajo remuestreada componente por componente (ver el modulo)."""
    crop = crop_to_label(lab_img)
    orig_ids = node_ids(sitk.GetArrayFromImage(crop))
    ids_img = sitk.GetImageFromArray(orig_ids)
    ids_img.CopyInformation(crop)
    new = _nearest(ids_img, ref, sitk.sitkInt32).astype(np.int32)
    info = {"rescatados": _rescue(new, orig_ids, crop, ref), "partidos": 0, "voxeles_separados": 0}
    for c in CLASSES:
        info["partidos"] += _reconnect(new, ids_img, ref, c)
        info["voxeles_separados"] += _separate(new, c)
    return (new // BASE).astype(np.uint8), info


def soft_fractions(lab_img: sitk.Image, ref: sitk.Image, factors_zyx: Sequence[int] = FACTORS_ZYX, pad: int = 2) -> np.ndarray:
    """(2, Z, Y, X) float32: fraccion de cada voxel de `ref` que ocupan la clase 1 y la clase 2 en la etiqueta
    original. Fuera de la caja de la etiqueta la fraccion es 0."""
    shape = ref.GetSize()[::-1]
    out = np.zeros((len(CLASSES), *shape), dtype=np.float32)
    arr = sitk.GetArrayFromImage(lab_img)
    nz = np.argwhere(arr > 0)
    if len(nz) == 0:
        return out
    lo_o, hi_o = nz.min(axis=0)[::-1] - 0.5, nz.max(axis=0)[::-1] + 0.5  # bordes, (x, y, z) de la rejilla original
    corners = [lab_img.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
               for x in (lo_o[0], hi_o[0]) for y in (lo_o[1], hi_o[1]) for z in (lo_o[2], hi_o[2])]
    ci = np.array([ref.TransformPhysicalPointToContinuousIndex(c) for c in corners])
    size = np.array(ref.GetSize())
    lo = np.clip(np.floor(ci.min(axis=0) + 0.5).astype(int) - pad, 0, size)
    hi = np.clip(np.floor(ci.max(axis=0) + 0.5).astype(int) + 1 + pad, 0, size)
    if np.any(hi <= lo):
        return out
    out[:, lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = fractions(lab_img, ref, lo, hi, CLASSES, factors_zyx)
    return out


def soft_to_probs(frac: np.ndarray) -> np.ndarray:
    """(2, Z, Y, X) fracciones -> (3, Z, Y, X) probabilidades con el fondo como resto."""
    bg = np.clip(1.0 - frac.sum(axis=0), 0.0, 1.0)
    return np.concatenate([bg[None], frac]).astype(np.float32)
