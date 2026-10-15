"""Auditoria del preprocesamiento en todos los casos (Fase 0 del plan). Solo CPU; nada se escribe en modelos/mamba.

    python -m bifusion3d.tools.audit_data --config configs/siam_full.yaml --out runs/auditoria
    python -m bifusion3d.tools.audit_data --config configs/siam_full.yaml --out runs/auditoria_humo --cases CHUM-001 MDA-075

Usa el mismo codigo de mambalite3d, solo para leer (read_lps, compute_roi, su remuestreo y evaluate_case), y mide en
cada caso cuanto pierde el preprocesamiento actual:

- geometria: espaciados de CT, PET y etiqueta, si la etiqueta esta en la rejilla del CT y su campo de vision;
- ROI: tamano, si encontro el ancla del cerebro (SUV > suv_anchor) y la cobertura de GTVp y de GTVn por separado;
- ida y vuelta de la etiqueta real a la rejilla del ROI (target_spacing_xyz) y de vuelta a la suya, con las metricas
  HECKTOR contra la original: lo que sacaria un modelo perfecto. Tres versiones:
    vecino    lo de hoy: vecino mas cercano a la ida (load_case) y a la vuelta (restore_to_original);
    fraccion  etiqueta por fraccion de volumen (vecino a una rejilla fina de 1 x 1 x 0,5 mm, media por bloques y clase
              mayoritaria; empates: vecino) y vuelta con vecino, como quedaria con label_resample: fraction;
    suave     las fracciones sin decidir, vuelta lineal y argmax en la rejilla original (techo de la Fase 1);
- HU y SUV dentro de cada clase; fraccion de la etiqueta bajo -900 HU (borrado de rostro) y fuera de la ventana de CT;
- escala del z-score del PET: media y desviacion dentro del ROI, como _normalize, y el valor que toma un SUV de 10.

Salida en --out: casos.jsonl (una linea por caso; si se corta, al relanzar salta los hechos), manifest.csv,
resumen.json (por centro, espaciado z de la etiqueta y split, con las puertas G0a-G0e del plan) y marcados.csv con
qc/<caso>.png de los casos marcados.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..guard import refuse_inside_mamba  # importa bifusion3d, que anade modelos/mamba a sys.path

import SimpleITK as sitk  # noqa: E402
from scipy import ndimage  # noqa: E402

from mambalite3d.config import load_config, resolve  # noqa: E402
from mambalite3d.data import io as mio  # noqa: E402
from mambalite3d.data.splits import select_subset, split_cases  # noqa: E402
from mambalite3d.eval.metrics import STRUCT_26, CaseResult, aggregate, evaluate_case  # noqa: E402

VERSIONS = ("vecino", "fraccion", "suave")
FINE_MM = (1.0, 1.0, 0.5)  # rejilla fina (xyz) para la fraccion de volumen: 6 subcortes por voxel de 3 mm
SUV_BIN, SUV_BINS = 0.25, 400  # histograma del SUV dentro de las etiquetas: 0-100 SUV, lo de mas va al ultimo
COV_MIN = 0.999  # puerta G0a
G0B_DSC, G0B_NODES = 0.005, 0.01  # puerta G0b
FACE_HU, FACE_FRAC = -900.0, 0.01  # puerta G0e
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
Bbox = Tuple[Tuple[int, int, int], Tuple[int, int, int]]  # (z0, y0, x0), (z1, y1, x1), incluidos


def _r(x, nd: int = 4):
    """Numero redondeado para el JSON; None si falta o no es finito."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _same_grid(a: sitk.Image, b: sitk.Image, tol: float = 1e-3) -> bool:
    return (
        tuple(a.GetSize()) == tuple(b.GetSize())
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=10 * tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


# ---------------------------------------------------------------------------------------------------------------
# medidas de un caso
# ---------------------------------------------------------------------------------------------------------------

def geometry(ct: sitk.Image, pet: sitk.Image, lab: sitk.Image) -> dict:
    g = {}
    for name, img in (("ct", ct), ("pet", pet), ("etiq", lab)):
        g[f"{name}_espaciado"] = " x ".join(f"{s:.4g}" for s in img.GetSpacing())
        g[f"{name}_tamano"] = " x ".join(str(s) for s in img.GetSize())
    g["ct_dz_mm"] = _r(ct.GetSpacing()[2], 3)
    g["etiq_dz_mm"] = _r(lab.GetSpacing()[2], 3)
    g["etiq_en_rejilla_ct"] = bool(_same_grid(lab, ct))
    g["etiq_con_z_del_pet"] = bool(not g["etiq_en_rejilla_ct"] and abs(lab.GetSpacing()[2] - pet.GetSpacing()[2]) < 1e-3)
    lo_l, hi_l = mio._physical_bounds(lab)
    lo_c, hi_c = mio._physical_bounds(ct)
    tol = np.asarray(ct.GetSpacing(), dtype=float)
    g["etiq_fov_menor"] = bool(np.any(lo_l > lo_c + tol) or np.any(hi_l < hi_c - tol))
    g["etiq_fuera_del_ct_mm"] = _r(max(0.0, float(np.max(lo_c - lo_l)), float(np.max(hi_l - hi_c))), 1)
    return g


def anchor_info(ct: sitk.Image, pet: sitk.Image, roi_cfg: dict) -> dict:
    """Repite la parte gruesa de compute_roi para saber si encontro el ancla (SUV > suv_anchor dentro del cuerpo)."""
    lo, hi = mio._physical_bounds(ct)
    size = np.maximum(np.ceil((hi - lo) / mio.COARSE_MM).astype(int) + 1, 1)
    ref = mio._grid(lo, (mio.COARSE_MM,) * 3, size)
    ct_c = sitk.GetArrayFromImage(mio._resample(ct, ref, sitk.sitkLinear, -1024, sitk.sitkFloat32))
    pet_c = sitk.GetArrayFromImage(mio._resample(pet, ref, sitk.sitkLinear, 0, sitk.sitkFloat32))
    body = (ct_c > roi_cfg["body_hu"]) | (pet_c > roi_cfg["body_suv"])
    if roi_cfg.get("main_body", True):
        body = mio.main_body(body, roi_cfg.get("open_iters", 2))
    z_mm = lo[2] + np.arange(body.shape[0]) * mio.COARSE_MM
    body_z = body.any(axis=(1, 2))
    active = ((pet_c > roi_cfg["suv_anchor"]) & body).any(axis=(1, 2))
    out = {"ancla": bool(active.any()), "ancla_bajo_el_tope_mm": None}
    if body_z.any() and active.any():
        out["ancla_bajo_el_tope_mm"] = _r(z_mm[body_z].max() - z_mm[active].max(), 1)
    return out


def roi_info(ct: sitk.Image, roi_lo, roi_hi, size) -> dict:
    lo_c, hi_c = mio._physical_bounds(ct)
    return {
        "roi_zyx": f"{size[2]} x {size[1]} x {size[0]}",
        "roi_mvox": _r(float(np.prod(size)) / 1e6, 2),
        "roi_z_mm": _r(roi_hi[2] - roi_lo[2], 1),
        "roi_tope_en_borde_ct": bool(roi_hi[2] >= hi_c[2] - 1e-3),
        "roi_base_en_borde_ct": bool(roi_lo[2] <= lo_c[2] + 1e-3),
    }


def coverage(lab: sitk.Image, lab_arr: np.ndarray, bbox: Optional[Bbox], roi_lo, size, spacing) -> dict:
    """Fraccion de los voxeles de cada clase que cae dentro de la caja del ROI (la cuenta de io.label_coverage, por clase)."""
    out = {"gtvp_ml": 0.0, "gtvn_ml": 0.0, "cobertura_gtvp": None, "cobertura_gtvn": None, "gtvp_fuera_ml": 0.0,
           "gtvn_fuera_ml": 0.0, "gtvn_componentes": 0, "gtvn_componentes_fuera": 0, "gtvn_componentes_cortados": 0}
    if bbox is None:
        return out
    (z0, y0, x0), (z1, y1, x1) = bbox
    crop = lab_arr[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    spacing = np.asarray(spacing, dtype=float)
    lo = np.asarray(roi_lo, dtype=float) - spacing / 2
    hi = lo + np.asarray(size, dtype=float) * spacing
    sp = np.asarray(lab.GetSpacing(), dtype=float)
    org = np.asarray(lab.GetOrigin(), dtype=float)
    dirm = np.asarray(lab.GetDirection(), dtype=float).reshape(3, 3)
    vox_ml = float(np.prod(sp)) / 1000.0
    for c, name in ((1, "gtvp"), (2, "gtvn")):
        idx = np.argwhere(crop == c)
        if len(idx) == 0:
            continue
        pts = org + ((idx + np.array([z0, y0, x0]))[:, ::-1] * sp) @ dirm.T  # (N, 3) xyz en mm
        inside = np.all((pts >= lo - 1e-3) & (pts <= hi + 1e-3), axis=1)
        out[f"{name}_ml"] = _r(len(idx) * vox_ml, 3)
        out[f"cobertura_{name}"] = _r(inside.mean(), 5)
        out[f"{name}_fuera_ml"] = _r((~inside).sum() * vox_ml, 3)
        if c == 2:
            comp, n = ndimage.label(crop == 2, structure=STRUCT_26)
            ids = comp[tuple(idx.T)]
            total = np.bincount(ids, minlength=n + 1)[1:]
            ins = np.bincount(ids, weights=inside.astype(float), minlength=n + 1)[1:]
            out["gtvn_componentes"] = int(n)
            out["gtvn_componentes_fuera"] = int(np.sum(ins == 0))
            out["gtvn_componentes_cortados"] = int(np.sum((ins > 0) & (ins < total)))
    return out


def fractions(lab: sitk.Image, origin, spacing, size) -> np.ndarray:
    """Fraccion de volumen de fondo, GTVp y GTVn en cada voxel de una rejilla (origen, espaciado, tamano en xyz).

    La etiqueta se lleva con vecino a una rejilla fina (FINE_MM; con 1x1x3 mm, 6 subcortes de 0,5 mm por voxel) y se
    promedia por bloques. Devuelve (3, Z, Y, X) float32.
    """
    spacing = np.asarray(spacing, dtype=float)
    size = np.asarray(size, dtype=int)
    f = np.maximum(1, np.rint(spacing / np.asarray(FINE_MM, dtype=float))).astype(int)  # xyz
    fine_sp = spacing / f
    fine_org = np.asarray(origin, dtype=float) - spacing / 2 + fine_sp / 2
    fine = mio._grid(fine_org, fine_sp, size * f)
    arr = sitk.GetArrayFromImage(mio._resample(lab, fine, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
    blk = arr.reshape(size[2], f[2], size[1], f[1], size[0], f[0])
    n = float(np.prod(f))
    return np.stack([(blk == c).sum(axis=(1, 3, 5), dtype=np.float32) / n for c in (0, 1, 2)])


def majority(fr: np.ndarray, nn: np.ndarray) -> np.ndarray:
    """Clase con mayor fraccion; en un empate, la del vecino mas cercano al centro del voxel."""
    best = fr.max(axis=0)
    out = fr.argmax(axis=0).astype(np.uint8)
    tie = (fr >= best[None] - 1e-6).sum(axis=0) > 1
    out[tie] = nn[tie]
    return out


def _roi_image(arr: np.ndarray, origin, spacing) -> sitk.Image:
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    img.SetSpacing([float(s) for s in spacing])
    img.SetOrigin([float(o) for o in origin])
    img.SetDirection(IDENTITY)
    return img


def _back_nearest(arr: np.ndarray, origin, spacing, ref: sitk.Image) -> np.ndarray:
    """Como io.restore_to_original: vecino mas cercano de la mascara a la rejilla de la etiqueta (aqui, a ref)."""
    img = _roi_image(arr.astype(np.uint8), origin, spacing)
    out = sitk.Resample(img, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out)


def _back_linear(fr: np.ndarray, origin, spacing, ref: sitk.Image) -> np.ndarray:
    """Cada fraccion con interpolacion lineal a la rejilla de la etiqueta y argmax alli (fuera del ROI: fondo)."""
    maps = []
    for c in range(fr.shape[0]):
        img = _roi_image(fr[c].astype(np.float32), origin, spacing)
        out = sitk.Resample(img, ref, sitk.Transform(), sitk.sitkLinear, 1.0 if c == 0 else 0.0, sitk.sitkFloat32)
        maps.append(sitk.GetArrayFromImage(out))
    return np.argmax(np.stack(maps), axis=0).astype(np.uint8)


def _crop_fg(pred: np.ndarray, gt: np.ndarray):
    """Recorta a la caja de lo que no es fondo en pred o gt: las metricas HECKTOR no cambian."""
    m = (pred > 0) | (gt > 0)
    if not m.any():
        z = np.zeros((1, 1, 1), dtype=np.uint8)
        return z, z
    sl = tuple(slice(int(a.min()), int(a.max()) + 1) for a in np.nonzero(m))
    return pred[sl], gt[sl]


def _roi_window(lab_sub: sitk.Image, roi_lo, size, spacing) -> Tuple[np.ndarray, np.ndarray]:
    """Indices [j0, j1] (xyz, incluidos) de la rejilla del ROI que cubren la caja de lab_sub, con 2 voxeles de margen."""
    sz = lab_sub.GetSize()
    pts = np.array([lab_sub.TransformContinuousIndexToPhysicalPoint((float(a), float(b), float(c)))
                    for a in (-0.5, sz[0] - 0.5) for b in (-0.5, sz[1] - 0.5) for c in (-0.5, sz[2] - 0.5)])
    k = (pts - np.asarray(roi_lo, dtype=float)) / np.asarray(spacing, dtype=float)
    j0 = np.maximum(np.floor(k.min(axis=0)).astype(int) - 2, 0)
    j1 = np.minimum(np.ceil(k.max(axis=0)).astype(int) + 2, np.asarray(size, dtype=int) - 1)
    return j0, j1


def round_trips(case_id: str, lab: sitk.Image, lab_arr: np.ndarray, bbox: Optional[Bbox], roi_lo, size, spacing,
                full: bool = False) -> dict:
    """Etiqueta -> rejilla del ROI -> rejilla de la etiqueta en las tres versiones, con evaluate_case contra la original.

    Solo se remuestrea alrededor de lo etiquetado (fuera de ahi las tres versiones dan fondo), y las metricas se
    calculan en el recorte de lo que no es fondo: es exacto e igual que en la rejilla completa. full=True hace lo
    mismo sobre las rejillas completas (solo para la prueba de que el recorte no cambia nada).
    """
    spacing = np.asarray(spacing, dtype=float)
    size = np.asarray(size, dtype=int)
    lab_size = np.asarray(lab.GetSize(), dtype=int)
    vox_mm3 = float(np.prod(lab.GetSpacing()))
    if bbox is None and not full:
        empty = np.zeros((1, 1, 1), dtype=np.uint8)
        res = asdict(evaluate_case(case_id, empty, empty, vox_mm3))
        return {f"iv_{v}": dict(res) for v in VERSIONS}
    if full:
        i0, i1 = np.zeros(3, dtype=int), lab_size - 1
    else:
        margin = np.ceil(2 * spacing / np.asarray(lab.GetSpacing(), dtype=float)).astype(int) + 1
        (z0, y0, x0), (z1, y1, x1) = bbox
        i0 = np.maximum(np.array([x0, y0, z0]) - margin, 0)
        i1 = np.minimum(np.array([x1, y1, z1]) + margin, lab_size - 1)
    lab_sub = sitk.RegionOfInterest(lab, [int(v) for v in i1 - i0 + 1], [int(v) for v in i0])
    gt = lab_arr[i0[2]:i1[2] + 1, i0[1]:i1[1] + 1, i0[0]:i1[0] + 1]
    if full:
        j0, j1 = np.zeros(3, dtype=int), size - 1
    else:
        j0, j1 = _roi_window(lab_sub, roi_lo, size, spacing)
    if np.any(j1 < j0):  # lo etiquetado queda entero fuera del ROI
        preds = {v: np.zeros_like(gt) for v in VERSIONS}
    else:
        sub_size = j1 - j0 + 1
        origin = np.asarray(roi_lo, dtype=float) + j0 * spacing  # mismos centros de voxel que la rejilla de load_case
        roi_sub = mio._grid(origin, spacing, sub_size)
        nn = sitk.GetArrayFromImage(mio._resample(lab, roi_sub, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
        fr = fractions(lab, origin, spacing, sub_size)
        preds = {
            "vecino": _back_nearest(nn, origin, spacing, lab_sub),
            "fraccion": _back_nearest(majority(fr, nn), origin, spacing, lab_sub),
            "suave": _back_linear(fr, origin, spacing, lab_sub),
        }
    out = {}
    for v in VERSIONS:
        p, g = _crop_fg(preds[v], gt)
        out[f"iv_{v}"] = asdict(evaluate_case(case_id, p, g, vox_mm3))
    return out


def intensities(ct: sitk.Image, pet: sitk.Image, lab: sitk.Image, lab_crop: np.ndarray, start_xyz, window):
    """HU y SUV dentro de cada clase, muestreados en la rejilla de la etiqueta (recorte de lo etiquetado)."""
    sub = sitk.RegionOfInterest(lab, [int(lab_crop.shape[2]), int(lab_crop.shape[1]), int(lab_crop.shape[0])],
                                [int(v) for v in start_xyz])
    ct_s = sitk.GetArrayFromImage(sitk.Resample(ct, sub, sitk.Transform(), sitk.sitkLinear, -1024.0, sitk.sitkFloat32))
    pet_s = sitk.GetArrayFromImage(sitk.Resample(pet, sub, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32))
    out, hist = {}, {}
    for c, name in ((1, "gtvp"), (2, "gtvn")):
        m = lab_crop == c
        if not m.any():
            continue
        hu, suv = ct_s[m], pet_s[m]
        for q, v in zip((1, 50, 99), np.percentile(hu, [1, 50, 99])):
            out[f"{name}_hu_p{q}"] = _r(v, 1)
        for q, v in zip(("50", "99", "999"), np.percentile(suv, [50, 99, 99.9])):
            out[f"{name}_suv_p{q}"] = _r(v, 2)
        out[f"{name}_suv_max"] = _r(suv.max(), 2)
        bins = np.minimum((np.maximum(suv, 0.0) / SUV_BIN).astype(int), SUV_BINS - 1)
        counts = np.bincount(bins, minlength=SUV_BINS)
        hist[name] = {int(k): int(n) for k, n in enumerate(counts) if n}
    m = lab_crop > 0
    if m.any():
        hu = ct_s[m]
        w_lo, w_hi = (float(v) for v in window)
        out["etiq_frac_hu_bajo_900"] = _r((hu < FACE_HU).mean(), 4)
        out["etiq_frac_fuera_ventana"] = _r(((hu < w_lo) | (hu > w_hi)).mean(), 4)
    return out, hist


def pet_scale(pet: sitk.Image, ref: sitk.Image) -> dict:
    """Media y desviacion del PET dentro del ROI, las que usa el z-score de io._normalize."""
    a = sitk.GetArrayFromImage(mio._resample(pet, ref, sitk.sitkLinear, 0, sitk.sitkFloat32))
    mean, std = float(a.mean()), float(a.std())
    return {
        "pet_roi_media": _r(mean, 3),
        "pet_roi_desv": _r(std, 3),
        "pet_roi_frac_suv_mayor_2": _r((a > 2).mean(), 4),
        "pet_roi_p999": _r(np.percentile(a, 99.9), 2),
        "zscore_de_suv_10": _r((10.0 - mean) / max(std, 1e-6), 2),
    }


def flags(row: dict) -> List[str]:
    f = []
    for name in ("gtvp", "gtvn"):
        cov = row.get(f"cobertura_{name}")
        if cov is not None and cov < COV_MIN:
            f.append(f"cobertura {name.upper()} {cov:.3f}")
    if not row.get("ancla", True):
        f.append("sin ancla del cerebro")
    face = row.get("etiq_frac_hu_bajo_900") or 0.0
    if face > FACE_FRAC:
        f.append(f"{100 * face:.1f} % de la etiqueta bajo -900 HU")
    if row.get("etiq_fov_menor"):
        f.append("etiqueta con campo de vision menor que el CT")
    if (row.get("etiq_fuera_del_ct_mm") or 0.0) > 1.0:
        f.append("etiqueta fuera del campo de vision del CT")
    if row.get("etiq_valores_raros"):
        f.append(f"valores de etiqueta raros {row['etiq_valores_raros']}")
    d = (row.get("iv_vecino") or {}).get("dice_gtvp")
    if row.get("gtvp_ml") and d is not None and d < 0.85:
        f.append(f"ida y vuelta GTVp {d:.3f}")
    return f


def qc_png(path: Path, ct, pet, lab, roi_lo, roi_hi, title: str) -> None:
    """PET coronal (MIP) con la caja del ROI y las etiquetas, y CT coronal y axial por el centro de la etiqueta."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    lo, hi = mio._physical_bounds(ct)
    step = 2.0
    size = np.maximum(np.ceil((hi - lo) / step).astype(int) + 1, 1)
    q = mio._grid(lo, (step,) * 3, size)
    ct_q = sitk.GetArrayFromImage(mio._resample(ct, q, sitk.sitkLinear, -1024, sitk.sitkFloat32))
    pet_q = sitk.GetArrayFromImage(mio._resample(pet, q, sitk.sitkLinear, 0, sitk.sitkFloat32))
    lab_q = sitk.GetArrayFromImage(mio._resample(lab, q, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
    xs, ys, zs = (lo[i] + np.arange(size[i]) * step for i in range(3))
    half = step / 2
    ext_xz = [xs[0] - half, xs[-1] + half, zs[0] - half, zs[-1] + half]
    ext_xy = [xs[0] - half, xs[-1] + half, ys[-1] + half, ys[0] - half]  # anterior arriba
    fgq = lab_q > 0
    if fgq.any():  # los cortes con mas voxeles etiquetados
        zc, yc = int(np.argmax(fgq.sum(axis=(1, 2)))), int(np.argmax(fgq.sum(axis=(0, 2))))
    else:
        zc, yc = int(size[2] // 2), int(size[1] // 2)
    colors = ((1, "tab:orange"), (2, "tab:cyan"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes[0]
    ax.imshow(pet_q.max(axis=1), cmap="gray_r", vmin=0, vmax=10, origin="lower", extent=ext_xz)
    for c, col in colors:
        proj = (lab_q == c).any(axis=1)
        if proj.any():
            ax.contour(xs, zs, proj.astype(float), levels=[0.5], colors=col, linewidths=1)
    ax.add_patch(Rectangle((roi_lo[0], roi_lo[2]), roi_hi[0] - roi_lo[0], roi_hi[2] - roi_lo[2], fill=False,
                           ec="tab:red", lw=1.5))
    ax.set_title("PET (MIP coronal, SUV 0-10): ROI en rojo", fontsize=9)
    ax = axes[1]
    ax.imshow(ct_q[:, yc, :], cmap="gray", vmin=-200, vmax=400, origin="lower", extent=ext_xz)
    for c, col in colors:
        sl = lab_q[:, yc, :] == c
        if sl.any():
            ax.contour(xs, zs, sl.astype(float), levels=[0.5], colors=col, linewidths=1)
    ax.add_patch(Rectangle((roi_lo[0], roi_lo[2]), roi_hi[0] - roi_lo[0], roi_hi[2] - roi_lo[2], fill=False,
                           ec="tab:red", lw=1.0))
    ax.set_title(f"CT coronal con mas etiqueta (y = {ys[yc]:.0f} mm)", fontsize=9)
    ax = axes[2]
    ax.imshow(ct_q[zc], cmap="gray", vmin=-200, vmax=400, extent=ext_xy)
    for c, col in colors:
        sl = lab_q[zc] == c
        if sl.any():
            ax.contour(xs, ys, sl.astype(float), levels=[0.5], colors=col, linewidths=1)
    ax.add_patch(Rectangle((roi_lo[0], roi_lo[1]), roi_hi[0] - roi_lo[0], roi_hi[1] - roi_lo[1], fill=False,
                           ec="tab:red", lw=1.0))
    ax.set_title(f"CT axial con mas etiqueta (z = {zs[zc]:.0f} mm); GTVp naranja, GTVn cian", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(str(path), dpi=70)
    plt.close(fig)


def audit_case(files: mio.CaseFiles, data_cfg: dict, split_name: str = "", qc_dir: Optional[str] = None) -> dict:
    t0 = time.time()
    spacing = tuple(float(s) for s in data_cfg["target_spacing_xyz"])
    row: Dict = {"case_id": files.case_id, "centro": files.center, "split": split_name}
    ct, pet, lab = mio.read_lps(files.ct), mio.read_lps(files.pet), mio.read_lps(files.label)
    row.update(geometry(ct, pet, lab))
    roi_lo, roi_hi = mio.compute_roi(ct, pet, data_cfg["roi"], spacing)
    size = np.maximum(np.floor((roi_hi - roi_lo) / np.array(spacing)).astype(int) + 1, 1)  # como load_case
    ref = mio._grid(roi_lo, spacing, size)
    row.update(anchor_info(ct, pet, data_cfg["roi"]))
    row.update(roi_info(ct, roi_lo, roi_hi, size))
    lab_arr = sitk.GetArrayFromImage(lab)
    if lab_arr.dtype != np.uint8:
        lab_arr = np.rint(lab_arr).clip(0, 255).astype(np.uint8)
    nz = np.nonzero(lab_arr)
    fg = lab_arr[nz]  # solo lo etiquetado (evita copias del volumen entero)
    row["etiq_valores_raros"] = [int(v) for v in np.unique(fg[fg > 2])]
    bbox = (tuple(int(a.min()) for a in nz), tuple(int(a.max()) for a in nz)) if len(nz[0]) else None
    row.update(coverage(lab, lab_arr, bbox, roi_lo, size, spacing))
    row.update(round_trips(files.case_id, lab, lab_arr, bbox, roi_lo, size, spacing))
    hist = {}
    if bbox is not None:
        (z0, y0, x0), (z1, y1, x1) = bbox
        vals, hist = intensities(ct, pet, lab, lab_arr[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1], (x0, y0, z0),
                                 data_cfg["ct_window"])
        row.update(vals)
    row.update(pet_scale(pet, ref))
    row["roi_lo_mm"] = [_r(v, 2) for v in roi_lo]
    row["roi_hi_mm"] = [_r(v, 2) for v in roi_hi]
    row["marcas"] = flags(row)
    if qc_dir and row["marcas"]:
        try:
            title = f"{files.case_id} ({files.center}, {split_name or 'sin split'}): " + "; ".join(row["marcas"])
            qc_png(Path(qc_dir) / f"{files.case_id}.png", ct, pet, lab, roi_lo, roi_hi, title)
            row["qc"] = f"qc/{files.case_id}.png"
        except Exception as e:  # noqa: BLE001  (la imagen es accesoria)
            row["qc_error"] = repr(e)
    row["hist_suv"] = hist
    row["segundos"] = _r(time.time() - t0, 1)
    return row


def _worker(files: mio.CaseFiles, data_cfg: dict, split_name: str, qc_dir: Optional[str], threads: int) -> dict:
    try:
        try:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(threads))
        except AttributeError:  # versiones antiguas de SimpleITK
            sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(int(threads))
        return audit_case(files, data_cfg, split_name, qc_dir)
    except Exception:  # noqa: BLE001  (un caso raro no detiene la auditoria)
        return {"case_id": files.case_id, "centro": files.center, "split": split_name,
                "error": traceback.format_exc(limit=8)}


# ---------------------------------------------------------------------------------------------------------------
# resumen
# ---------------------------------------------------------------------------------------------------------------

def _low_cov(row: dict) -> bool:
    return any(row.get(f"cobertura_{n}") is not None and row[f"cobertura_{n}"] < COV_MIN for n in ("gtvp", "gtvn"))


def _results(rows: Sequence[dict], version: str) -> List[CaseResult]:
    return [CaseResult(**r[f"iv_{version}"]) for r in rows if r.get(f"iv_{version}")]


def round_trip_metrics(rows: Sequence[dict]) -> dict:
    keys = ("n_cases", "gtvp_dice_mean", "gtvp_dice_mean_ref_only", "gtvn_dice_agg", "gtvn_f1_agg",
            "gtvn_tp", "gtvn_fp", "gtvn_fn")
    out = {}
    for v in VERSIONS:
        a = aggregate(_results(rows, v)) if rows else {}
        out[v] = {k: (_r(a[k]) if isinstance(a.get(k), float) else a.get(k)) for k in keys}
    for v in ("fraccion", "suave"):
        out[f"{v}_menos_vecino"] = {
            k: _r(out[v][k] - out["vecino"][k]) if out[v][k] is not None and out["vecino"][k] is not None else None
            for k in ("gtvp_dice_mean", "gtvn_dice_agg", "gtvn_f1_agg")
        }
    return out


def _dz_group(row: dict) -> str:
    dz = row.get("etiq_dz_mm")
    return "dz <= 2 mm" if dz is not None and dz <= 2.0 else "dz > 2 mm"


def _groups(rows: Sequence[dict], key) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return dict(sorted(out.items()))


def _group_summary(rows: Sequence[dict]) -> dict:
    return {
        "n_casos": len(rows),
        "casos_con_cobertura_baja": sum(1 for r in rows if _low_cov(r)),
        "ml_fuera_del_roi": {n: _r(sum(r.get(f"{n}_fuera_ml") or 0.0 for r in rows), 2) for n in ("gtvp", "gtvn")},
        "ida_y_vuelta": round_trip_metrics(rows),
    }


def _pct(values, qs=(5, 50, 95), nd: int = 3):
    v = np.array([x for x in values if x is not None and math.isfinite(float(x))], dtype=float)
    return [_r(x, nd) for x in np.percentile(v, qs)] if len(v) else None


def _hist_sum(rows: Sequence[dict], classes=("gtvp", "gtvn")) -> np.ndarray:
    total = np.zeros(SUV_BINS)
    for r in rows:
        for c in classes:
            for k, n in ((r.get("hist_suv") or {}).get(c) or {}).items():
                total[int(k)] += n  # las claves vuelven del JSON como texto
    return total


def _hist_pct(counts: np.ndarray, qs) -> List[Optional[float]]:
    n = counts.sum()
    if n == 0:
        return [None for _ in qs]
    cum = np.cumsum(counts) / n
    return [_r((int(np.searchsorted(cum, q / 100.0)) + 1) * SUV_BIN, 2) for q in qs]


def gates(ok: Sequence[dict]) -> dict:
    low = [r for r in ok if _low_cov(r)]
    g0a = {
        "criterio": f"algun caso con cobertura de GTVp o GTVn < {COV_MIN}",
        "se_cumple": bool(low),
        "consecuencia": ("probar un ROI corregido (R), primero solo en inferencia" if low
                         else "el ROI actual cubre todas las etiquetas: R no hace falta"),
        "por_split": dict(Counter(r.get("split") or "sin split" for r in low)),
        "casos": [{k: r.get(k) for k in ("case_id", "split", "cobertura_gtvp", "cobertura_gtvn", "gtvp_fuera_ml",
                                         "gtvn_fuera_ml", "gtvn_componentes_fuera", "gtvn_componentes_cortados")}
                  for r in low],
    }
    m = round_trip_metrics(ok)
    v, f = m["vecino"], m["fraccion"]
    nodes = (v["gtvn_tp"] or 0) + (v["gtvn_fn"] or 0)
    recovered = ((v["gtvn_fn"] or 0) - (f["gtvn_fn"] or 0)) / nodes if nodes else 0.0
    d_dsc = (f["gtvn_dice_agg"] or 0.0) - (v["gtvn_dice_agg"] or 0.0)
    g0b_ok = bool(d_dsc >= G0B_DSC or recovered >= G0B_NODES)
    g0b = {
        "criterio": f"la etiqueta por fraccion sube el DSCagg de GTVn de la ida y vuelta >= {G0B_DSC} "
                    f"o recupera >= {100 * G0B_NODES:.0f} % de los ganglios",
        "se_cumple": g0b_ok,
        "consecuencia": "L entra en el 2x2" if g0b_ok else "L no entra: la Fase 3 queda en B0 frente a P",
        "dscagg_gtvn": {"vecino": v["gtvn_dice_agg"], "fraccion": f["gtvn_dice_agg"], "diferencia": _r(d_dsc)},
        "ganglios": {"total": nodes, "fn_vecino": v["gtvn_fn"], "fn_fraccion": f["gtvn_fn"],
                     "recuperados_pct": _r(100 * recovered, 2)},
        "dsc_gtvp": {"vecino": v["gtvp_dice_mean"], "fraccion": f["gtvp_dice_mean"]},
    }
    g0c = {"criterio": "ida y vuelta suave (fracciones, vuelta lineal) frente a la actual: techo de la Fase 1"}
    for name, rows in [("todos", list(ok))] + list(_groups(ok, _dz_group).items()):
        mm = round_trip_metrics(rows)
        g0c[name] = {"n_casos": len(rows), "vecino": mm["vecino"], "suave": mm["suave"],
                     "suave_menos_vecino": mm["suave_menos_vecino"]}
    train = [r for r in ok if r.get("split") == "train"]
    base = train or list(ok)
    p99, p999 = _hist_pct(_hist_sum(base), (99, 99.9))
    g0d = {
        "criterio": "percentil 99,9 del SUV dentro de las etiquetas (GTVp y GTVn) de los casos de train",
        "casos_usados": "train" if train else "todos (no hay split)",
        "p99": p99,
        "p99_9": p999,
        "c_propuesto": int(5 * math.ceil(p999 / 5.0)) if p999 else 20,
        "por_clase": {c: dict(zip(("p50", "p99", "p99_9"), _hist_pct(_hist_sum(base, (c,)), (50, 99, 99.9))))
                      for c in ("gtvp", "gtvn")},
        "todos_los_casos": dict(zip(("p99", "p99_9"), _hist_pct(_hist_sum(ok), (99, 99.9)))),
    }
    face = [r for r in ok if (r.get("etiq_frac_hu_bajo_900") or 0.0) > FACE_FRAC]
    g0e = {
        "criterio": f"mas del {100 * FACE_FRAC:.0f} % de los voxeles etiquetados bajo {FACE_HU:.0f} HU",
        "n": len(face),
        "consecuencia": "lista para el analisis por caso; el preprocesamiento no cambia",
        "casos": [{"case_id": r["case_id"], "split": r.get("split"), "fraccion": r.get("etiq_frac_hu_bajo_900")}
                  for r in sorted(face, key=lambda r: -(r.get("etiq_frac_hu_bajo_900") or 0.0))],
    }
    return {"G0a_roi": g0a, "G0b_etiquetas": g0b, "G0c_restauracion": g0c, "G0d_suv": g0d, "G0e_rostro": g0e}


def summarize(rows: Sequence[dict], meta: dict) -> dict:
    ok = [r for r in rows if "error" not in r]
    err = [r for r in rows if "error" in r]
    dz = Counter(f"{r['ct_dz_mm']:.2f}" for r in ok if r.get("ct_dz_mm") is not None)
    return {
        **meta,
        "n_casos": len(rows),
        "n_con_error": len(err),
        "errores": [{"case_id": r["case_id"], "error": r["error"].strip().splitlines()[-1]} for r in err],
        "puertas": gates(ok) if ok else {},
        "todos": _group_summary(ok),
        "por_centro": {k: _group_summary(v) for k, v in _groups(ok, lambda r: r["centro"]).items()},
        "por_espaciado_z": {k: _group_summary(v) for k, v in _groups(ok, _dz_group).items()},
        "por_split": {k: _group_summary(v) for k, v in _groups(ok, lambda r: r.get("split") or "sin split").items()},
        "geometria": {
            "ct_dz_mm": dict(sorted(dz.items(), key=lambda kv: -kv[1])),
            "etiqueta_fuera_de_la_rejilla_del_ct": sum(1 for r in ok if not r.get("etiq_en_rejilla_ct")),
            "etiqueta_con_z_del_pet": sum(1 for r in ok if r.get("etiq_con_z_del_pet")),
            "etiqueta_con_fov_menor": [r["case_id"] for r in ok if r.get("etiq_fov_menor")],
            "etiqueta_fuera_del_ct": [r["case_id"] for r in ok if (r.get("etiq_fuera_del_ct_mm") or 0) > 1],
            "valores_de_etiqueta_raros": {r["case_id"]: r["etiq_valores_raros"] for r in ok if r.get("etiq_valores_raros")},
        },
        "roi": {
            "mvox_p5_p50_p95": _pct([r.get("roi_mvox") for r in ok], nd=2),
            "z_mm_p5_p50_p95": _pct([r.get("roi_z_mm") for r in ok], nd=1),
            "sin_ancla": [r["case_id"] for r in ok if not r.get("ancla", True)],
            "ancla_bajo_el_tope_mm_p50_p95_max": _pct([r.get("ancla_bajo_el_tope_mm") for r in ok], (50, 95, 100), 1),
            "tope_en_el_borde_del_ct": sum(1 for r in ok if r.get("roi_tope_en_borde_ct")),
            "base_en_el_borde_del_ct": sum(1 for r in ok if r.get("roi_base_en_borde_ct")),
        },
        "escala_zscore_pet": {
            "media_p5_p50_p95": _pct([r.get("pet_roi_media") for r in ok]),
            "desviacion_p5_p50_p95": _pct([r.get("pet_roi_desv") for r in ok]),
            "suv10_en_zscore_p5_p50_p95": _pct([r.get("zscore_de_suv_10") for r in ok], nd=2),
            "suv10_en_zscore_mediana_por_centro": {k: (_pct([r.get("zscore_de_suv_10") for r in v], (50,), 2) or [None])[0]
                                                   for k, v in _groups(ok, lambda r: r["centro"]).items()},
        },
        "intensidades": {
            "gtvp_hu_p50_p5_p50_p95": _pct([r.get("gtvp_hu_p50") for r in ok], nd=1),
            "gtvn_hu_p50_p5_p50_p95": _pct([r.get("gtvn_hu_p50") for r in ok], nd=1),
            "etiqueta_fuera_de_la_ventana_ct_p50_p95_max": _pct([r.get("etiq_frac_fuera_ventana") for r in ok],
                                                                 (50, 95, 100), 4),
        },
        "segundos_por_caso_p50_max": _pct([r.get("segundos") for r in ok], (50, 100), 1),
    }


FIRST = ("case_id", "centro", "split", "marcas")
NESTED = ("hist_suv", "marcas", "roi_lo_mm", "roi_hi_mm", "etiq_valores_raros") + tuple(f"iv_{v}" for v in VERSIONS)


def _flat(row: dict) -> dict:
    d = {k: v for k, v in row.items() if k not in NESTED}
    d["marcas"] = "; ".join(row.get("marcas") or [])
    d["etiq_valores_raros"] = " ".join(str(v) for v in row.get("etiq_valores_raros") or [])
    for v in VERSIONS:
        m = row.get(f"iv_{v}") or {}
        d[f"iv_{v}_dsc_gtvp"] = _r(m.get("dice_gtvp"))
        d[f"iv_{v}_dsc_gtvn"] = _r(m.get("dice_gtvn_case"))
        d[f"iv_{v}_ganglios_tp_fp_fn"] = f"{m.get('gtvn_tp')}/{m.get('gtvn_fp')}/{m.get('gtvn_fn')}" if m else ""
    return d


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields: List[str] = [k for k in FIRST if any(k in r for r in rows)]
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_outputs(rows: Sequence[dict], out: Path, meta: dict) -> dict:
    ok = [r for r in rows if "error" not in r]
    _write_csv(out / "manifest.csv", [_flat(r) for r in ok])
    marked = [r for r in ok if r.get("marcas")]
    _write_csv(out / "marcados.csv", [{"case_id": r["case_id"], "centro": r["centro"], "split": r.get("split"),
                                      "marcas": "; ".join(r["marcas"]), "qc": r.get("qc", "")} for r in marked])
    summary = summarize(rows, meta)
    tmp = out / "resumen.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out / "resumen.json")
    return summary


def print_summary(s: dict) -> None:
    p = s.get("puertas") or {}
    print("=" * 70)
    print(f"auditoria: {s['n_casos']} casos ({s['n_con_error']} con error)")
    if not p:
        return
    a, b, c, d, e = (p[k] for k in ("G0a_roi", "G0b_etiquetas", "G0c_restauracion", "G0d_suv", "G0e_rostro"))
    print(f"G0a ROI: {len(a['casos'])} casos con cobertura < {COV_MIN} {a['por_split']} -> {a['consecuencia']}")
    print(f"G0b etiquetas: DSCagg GTVn {b['dscagg_gtvn']}, ganglios {b['ganglios']} -> {b['consecuencia']}")
    for k in ("todos", "dz <= 2 mm", "dz > 2 mm"):
        if k in c:
            vv, ss = c[k]["vecino"], c[k]["suave"]
            print(f"G0c restauracion [{k}, {c[k]['n_casos']} casos]: GTVp DSC {vv['gtvp_dice_mean']} -> "
                  f"{ss['gtvp_dice_mean']}, GTVn DSCagg {vv['gtvn_dice_agg']} -> {ss['gtvn_dice_agg']}, "
                  f"F1 {vv['gtvn_f1_agg']} -> {ss['gtvn_f1_agg']}")
    print(f"G0d SUV ({d['casos_usados']}): p99 {d['p99']}, p99,9 {d['p99_9']} -> c = {d['c_propuesto']}")
    print(f"G0e rostro: {e['n']} casos con mas del {100 * FACE_FRAC:.0f} % de la etiqueta bajo {FACE_HU:.0f} HU")
    z = s["escala_zscore_pet"]
    print(f"z-score del PET: un SUV de 10 pasa a {z['suv10_en_zscore_p5_p50_p95']} (p5, p50, p95 entre casos)")
    print("=" * 70)


# ---------------------------------------------------------------------------------------------------------------
# programa
# ---------------------------------------------------------------------------------------------------------------

def _read_jsonl(path: Path) -> Dict[str, dict]:
    done: Dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:  # ultima linea a medias si se corto
                    continue
                done[row["case_id"]] = row
    return done


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/siam_full.yaml", help="config del que se toman data.* y la semilla")
    ap.add_argument("--out", default="runs/auditoria")
    ap.add_argument("--workers", type=int, default=max(1, min(3, (os.cpu_count() or 2) // 2)),
                    help="procesos en paralelo (por defecto, la mitad de los nucleos, hasta 3)")
    ap.add_argument("--threads", type=int, default=2, help="hilos de SimpleITK por proceso")
    ap.add_argument("--cases", nargs="+", default=None, help="solo estos casos (p. ej. para una prueba corta)")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--no-qc", action="store_true", help="sin imagenes de los casos marcados")
    args = ap.parse_args(argv)
    refuse_inside_mamba(argv, flags=("--out",))

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    root = resolve(data_cfg["root"])
    all_cases = mio.find_cases(root)
    subset = select_subset(all_cases, data_cfg["cases_per_center"], data_cfg["centers"], cfg["seed"])
    train, val, test = split_cases(subset, data_cfg["split"], cfg["seed"])
    split_of = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        split_of.update({c.case_id: name for c in part})
    cases = [c for c in all_cases if c.label is not None]
    no_label = [c.case_id for c in all_cases if c.label is None]
    if args.cases:
        wanted = set(args.cases)
        missing = wanted - {c.case_id for c in cases}
        if missing:
            print(f"AVISO: no se encontraron (o no tienen etiqueta): {sorted(missing)}")
        cases = [c for c in cases if c.case_id in wanted]
    if args.max_cases:
        cases = cases[: args.max_cases]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qc_dir = None
    if not args.no_qc:
        (out / "qc").mkdir(exist_ok=True)
        qc_dir = str(out / "qc")
    jsonl = out / "casos.jsonl"
    done = _read_jsonl(jsonl)
    todo = [c for c in cases if c.case_id not in done or "error" in done[c.case_id]]
    print(f"auditoria de {root}: {len(cases)} casos con etiqueta ({len(no_label)} sin etiqueta), "
          f"{len(cases) - len(todo)} ya hechos, {len(todo)} por hacer, {args.workers} procesos", flush=True)

    # el worker se pasa desde el modulo importado por su nombre (no desde __main__), para que funcione con spawn
    mod = importlib.import_module("bifusion3d.tools.audit_data")
    t0 = time.time()
    with open(jsonl, "a", encoding="utf-8") as fh:
        def keep(row: dict, i: int) -> None:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            done[row["case_id"]] = row
            if "error" in row:
                print(f"  ERROR en {row['case_id']}: {row['error'].strip().splitlines()[-1]}", flush=True)
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} casos ({time.time() - t0:.0f} s)", flush=True)

        if args.workers <= 1:
            for i, c in enumerate(todo, 1):
                keep(mod._worker(c, data_cfg, split_of.get(c.case_id, ""), qc_dir, args.threads), i)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(mod._worker, c, data_cfg, split_of.get(c.case_id, ""), qc_dir, args.threads): c
                        for c in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    c = futs[fut]
                    try:
                        row = fut.result()
                    except Exception as e:  # noqa: BLE001  (p. ej. un proceso muerto por memoria)
                        row = {"case_id": c.case_id, "centro": c.center, "split": split_of.get(c.case_id, ""),
                               "error": f"el proceso fallo: {e!r}"}
                    keep(row, i)

    rows = [done[c.case_id] for c in cases if c.case_id in done]
    meta = {"config": str(args.config), "raiz_datos": str(root), "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
            "segundos_totales": round(time.time() - t0, 1), "sin_etiqueta": no_label,
            "target_spacing_xyz": data_cfg["target_spacing_xyz"], "roi": data_cfg["roi"]}
    summary = write_outputs(rows, out, meta)
    print_summary(summary)
    print(f"guardado en {out}: manifest.csv, resumen.json, marcados.csv" + ("" if args.no_qc else ", qc/"))
    return 0 if summary["n_con_error"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
