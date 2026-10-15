"""Preprocesado v2: el de mambalite3d con tres opciones, todas apagadas por defecto.

Opciones en el config (`data:`):
  roi.ancla: absoluta | relativa
      absoluta (por defecto): como mambalite3d, el ancla del ROI es el corte mas craneal con PET > 2 (SUV) y el cuerpo
      incluye PET > 0,5.
      relativa: los dos umbrales se expresan en veces el tejido blando del propio paciente (mediana del PET donde el
      CT esta entre 0 y 100 HU). Con el tejido tipico (0,82) dan los mismos 2 y 0,5; en un PET que no esta en SUV
      (CHUM-010) se escalan con el. La imagen no se toca: el z-score del PET ya es independiente de la escala.
  etiquetas: cercano | nodos | suave
      cercano (por defecto): vecino mas cercano, como mambalite3d.
      nodos: ganglio por ganglio (bifusion3d.data.labels.resample_nodes): sin fusiones ni particiones.
      suave: la etiqueta dura es la de nodos (muestreo, validacion) y ademas se guarda la fraccion de cada voxel que
      es GTVp y GTVn (case.soft, uint8 0-255, recortada a la caja de la etiqueta; case.soft_lo, su esquina zyx).

Con las opciones por defecto load_case_v2 devuelve exactamente lo mismo que mambalite3d.data.io.load_case.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk

from mambalite3d.data import io as mio

from . import labels as L
from .pet_scale import pet_stats

TEJIDO_REF = 0.8207  # mediana del tejido blando en los 680 casos (runs/experimento_preproc/resumen.json)
ANCLAS = ("absoluta", "relativa")
ETIQUETAS = ("cercano", "nodos", "suave")


def roi_cfg_relativo(ct: sitk.Image, pet: sitk.Image, roi_cfg: dict, ref: float = TEJIDO_REF) -> Tuple[dict, float]:
    """roi_cfg con body_suv y suv_anchor multiplicados por tejido / ref, y ese factor (1 si no se puede medir)."""
    t = pet_stats(ct, pet, roi_cfg)["suv_tejido"]
    escala = t / ref if (math.isfinite(t) and t > 0) else 1.0
    out = dict(roi_cfg)
    out["body_suv"] = float(roi_cfg["body_suv"]) * escala
    out["suv_anchor"] = float(roi_cfg["suv_anchor"]) * escala
    return out, float(escala)


def compute_roi_v2(ct: sitk.Image, pet: sitk.Image, roi_cfg: dict, spacing_xyz):
    ancla = roi_cfg.get("ancla", "absoluta")
    if ancla not in ANCLAS:
        raise ValueError(f"roi.ancla desconocida {ancla!r}: usa {' o '.join(ANCLAS)}")
    if ancla == "relativa":
        roi_cfg, _ = roi_cfg_relativo(ct, pet, roi_cfg)
    return mio.compute_roi(ct, pet, roi_cfg, spacing_xyz)


def _fg_coords(label: np.ndarray, max_fg_coords: int, seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for c in (1, 2):
        coords = np.argwhere(label == c)
        if len(coords) > max_fg_coords:
            coords = coords[rng.choice(len(coords), max_fg_coords, replace=False)]
        if len(coords):
            out[c] = coords.astype(np.int32)
    return out


def _crop_soft(frac: np.ndarray):
    """(2, Z, Y, X) float -> (uint8 recortado a donde hay fraccion, esquina zyx)."""
    nz = np.argwhere(frac.max(axis=0) > 0)
    if len(nz) == 0:
        return np.zeros((2, 0, 0, 0), np.uint8), (0, 0, 0)
    lo, hi = nz.min(axis=0), nz.max(axis=0) + 1
    sub = frac[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return np.round(np.clip(sub, 0, 1) * 255).astype(np.uint8), tuple(int(v) for v in lo)


def load_case_v2(files: mio.CaseFiles, data_cfg: dict, with_label: bool = True, max_fg_coords: int = 20000,
                 seed: int = 0) -> mio.CaseData:
    modo = data_cfg.get("etiquetas", "cercano")
    if modo not in ETIQUETAS:
        raise ValueError(f"data.etiquetas desconocido {modo!r}: usa {', '.join(ETIQUETAS)}")
    spacing = tuple(float(s) for s in data_cfg["target_spacing_xyz"])
    ct = mio.read_lps(files.ct)
    pet = mio.read_lps(files.pet)
    roi_lo, roi_hi = compute_roi_v2(ct, pet, data_cfg["roi"], spacing)
    size = np.maximum(np.floor((roi_hi - roi_lo) / np.array(spacing)).astype(int) + 1, 1)
    ref = mio._grid(roi_lo, spacing, size)

    ct_a = sitk.GetArrayFromImage(mio._resample(ct, ref, sitk.sitkLinear, -1024, sitk.sitkFloat32))
    pet_a = sitk.GetArrayFromImage(mio._resample(pet, ref, sitk.sitkLinear, 0, sitk.sitkFloat32))
    ct_n, pet_n = mio._normalize(ct_a, pet_a, data_cfg)
    image = np.stack([ct_n, pet_n]).astype(np.float16)

    label, fg, soft = None, {}, None
    if with_label and files.label is not None:
        lab = mio.read_lps(files.label)
        if modo == "cercano":
            label = sitk.GetArrayFromImage(mio._resample(lab, ref, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
        else:
            label, _ = L.resample_nodes(lab, ref)
            if modo == "suave":
                soft = _crop_soft(L.soft_fractions(lab, ref))
        fg = _fg_coords(label, max_fg_coords, seed)
    pad_values = (float(image[0].min()), float(image[1].min()))
    case = mio.CaseData(files.case_id, files.center, image, label, spacing, tuple(float(v) for v in roi_lo), fg,
                        pad_values, files)
    if soft is not None:
        case.soft, case.soft_lo = soft
    return case


def soft_patch(case, center, size) -> np.ndarray:
    """(2, *size) uint8 de fracciones alrededor de `center`, con la misma convencion que sampler._extract."""
    out = np.zeros((2, *size), np.uint8)
    soft = getattr(case, "soft", None)
    if soft is None or soft.size == 0:
        return out
    lo = [int(center[i]) - size[i] // 2 - case.soft_lo[i] for i in range(3)]  # en coordenadas de case.soft
    src_lo = [max(lo[i], 0) for i in range(3)]
    src_hi = [min(lo[i] + size[i], soft.shape[1 + i]) for i in range(3)]
    if any(src_hi[i] <= src_lo[i] for i in range(3)):
        return out
    d = [src_lo[i] - lo[i] for i in range(3)]
    out[:, d[0]:d[0] + src_hi[0] - src_lo[0], d[1]:d[1] + src_hi[1] - src_lo[1], d[2]:d[2] + src_hi[2] - src_lo[2]] = \
        soft[:, src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


def with_options(data_cfg: dict, ancla=None, etiquetas=None, down_mm=None) -> dict:
    """Copia de data_cfg con las opciones cambiadas (las que no son None)."""
    out = copy.deepcopy(data_cfg)
    if ancla is not None:
        out["roi"]["ancla"] = ancla
    if etiquetas is not None:
        out["etiquetas"] = etiquetas
    if down_mm is not None:
        out["roi"]["down_mm"] = float(down_mm)
    return out


def install() -> None:
    """Sustituye load_case (solo en memoria) donde mambalite3d lo usa. Sin opciones nuevas en el config, igual."""
    import importlib

    for name in ("mambalite3d.data.io", "mambalite3d.data.sampler"):
        importlib.import_module(name).load_case = load_case_v2
