"""Reading HECKTOR cases and building the head-and-neck ROI on the fly (nothing is written to disk).

Pipeline per case (tumour-agnostic, the label is never used to place the ROI):
1. Read CT, PET and label and reorient them to LPS (+z = superior).
2. On a coarse 4 mm grid: body = (CT > body_hu) | (PET > body_suv), reduced to the patient's main
   connected component (drops the scanner table, head rests and PET noise; `roi.main_body`).
   The top of the ROI is the most superior slice with SUV > suv_anchor inside the body
   (brain/head); the ROI goes down `down_mm` from there.
   X/Y come from the body inside that z range. Margins and a minimum size are added.
3. CT, PET and label are resampled once, directly onto the fine target grid covering the ROI.
4. CT is windowed to [0, 1]; PET is z-scored inside the ROI (or clipped SUV, see config).

Arrays are returned in numpy (z, y, x) order; spacings in configs are given in SimpleITK (x, y, z).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

COARSE_MM = 4.0


@dataclass
class CaseFiles:
    case_id: str
    center: str
    ct: Path
    pet: Path
    label: Optional[Path]


@dataclass
class CaseData:
    case_id: str
    center: str
    image: np.ndarray  # (2, Z, Y, X) float16: CT, PET
    label: Optional[np.ndarray]  # (Z, Y, X) uint8 or None
    spacing_xyz: Tuple[float, float, float]
    roi_origin_xyz: Tuple[float, float, float]
    fg_coords: Dict[int, np.ndarray] = field(default_factory=dict)  # class -> (N, 3) zyx voxel coords
    pad_values: Tuple[float, float] = (0.0, 0.0)
    files: Optional[CaseFiles] = None


def center_of(case_id: str) -> str:
    return case_id.split("-")[0]


def find_cases(root) -> List[CaseFiles]:
    root = Path(root)
    cases = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        cid = d.name
        ct, pet, lab = d / f"{cid}__CT.nii.gz", d / f"{cid}__PT.nii.gz", d / f"{cid}.nii.gz"
        if ct.exists() and pet.exists():
            cases.append(CaseFiles(cid, center_of(cid), ct, pet, lab if lab.exists() else None))
    return cases


def read_lps(path) -> sitk.Image:
    return sitk.DICOMOrient(sitk.ReadImage(str(path)), "LPS")


def _physical_bounds(img: sitk.Image) -> Tuple[np.ndarray, np.ndarray]:
    size = img.GetSize()
    corners = []
    for i in (0, size[0] - 1):
        for j in (0, size[1] - 1):
            for k in (0, size[2] - 1):
                corners.append(img.TransformIndexToPhysicalPoint((i, j, k)))
    corners = np.array(corners)
    return corners.min(axis=0), corners.max(axis=0)


def _grid(origin_xyz, spacing_xyz, size_xyz) -> sitk.Image:
    ref = sitk.Image([int(s) for s in size_xyz], sitk.sitkUInt8)
    ref.SetOrigin([float(o) for o in origin_xyz])
    ref.SetSpacing([float(s) for s in spacing_xyz])
    ref.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return ref


def _resample(img, ref, interp, default, pixel_type) -> sitk.Image:
    return sitk.Resample(img, ref, sitk.Transform(), interp, float(default), pixel_type)


def main_body(body: np.ndarray, open_iters: int = 2) -> np.ndarray:
    """Keeps the largest connected component of the body mask after removing thin structures.

    The scanner table, thermoplastic masks and isolated PET noise are thinner than
    2 * open_iters coarse voxels (16 mm by default) and/or disconnected from the patient.
    """
    from scipy import ndimage as ndi

    opened = ndi.binary_opening(body, iterations=open_iters) if open_iters > 0 else body
    lab, n = ndi.label(opened)
    if n == 0:
        return body
    sizes = np.bincount(lab.ravel())[1:]
    keep = lab == (int(np.argmax(sizes)) + 1)
    return ndi.binary_dilation(keep, iterations=1) & body  # recover boundary voxels lost by the opening


def compute_roi(ct: sitk.Image, pet: sitk.Image, roi_cfg: dict, spacing_xyz) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the ROI as physical (lo, hi) corners in LPS millimetres, (x, y, z) order."""
    lo, hi = _physical_bounds(ct)
    size = np.maximum(np.ceil((hi - lo) / COARSE_MM).astype(int) + 1, 1)
    ref = _grid(lo, (COARSE_MM,) * 3, size)
    ct_c = sitk.GetArrayFromImage(_resample(ct, ref, sitk.sitkLinear, -1024, sitk.sitkFloat32))
    pet_c = sitk.GetArrayFromImage(_resample(pet, ref, sitk.sitkLinear, 0, sitk.sitkFloat32))
    body = (ct_c > roi_cfg["body_hu"]) | (pet_c > roi_cfg["body_suv"])  # (z, y, x)
    if roi_cfg.get("main_body", True):
        body = main_body(body, roi_cfg.get("open_iters", 2))

    z_idx = np.arange(body.shape[0])
    z_mm = lo[2] + z_idx * COARSE_MM
    body_z = body.any(axis=(1, 2))
    if not body_z.any():
        return lo, hi
    z_lo, z_hi = z_mm[body_z].min(), z_mm[body_z].max()
    active = ((pet_c > roi_cfg["suv_anchor"]) & body).any(axis=(1, 2))
    if active.any():
        z_hi = min(z_mm[active].max(), z_hi)
        z_lo = max(z_hi - roi_cfg["down_mm"], z_lo)
    in_z = (z_mm >= z_lo) & (z_mm <= z_hi)
    body_xy = body[in_z].any(axis=0)  # (y, x)
    if not body_xy.any():
        body_xy = body.any(axis=0)
    yy, xx = np.nonzero(body_xy)
    roi_lo = np.array([lo[0] + xx.min() * COARSE_MM, lo[1] + yy.min() * COARSE_MM, z_lo])
    roi_hi = np.array([lo[0] + xx.max() * COARSE_MM, lo[1] + yy.max() * COARSE_MM, z_hi])

    margin = np.array(roi_cfg["margin_vox"], dtype=float) * np.array(spacing_xyz, dtype=float)
    roi_lo -= margin + COARSE_MM / 2
    roi_hi += margin + COARSE_MM / 2
    min_mm = np.array(roi_cfg["min_size_vox"], dtype=float) * np.array(spacing_xyz, dtype=float)
    extent = roi_hi - roi_lo
    grow = np.maximum(min_mm - extent, 0) / 2
    roi_lo -= grow
    roi_hi += grow
    # stay inside the CT field of view
    roi_lo = np.maximum(roi_lo, lo)
    roi_hi = np.minimum(roi_hi, hi)
    return roi_lo, roi_hi


def _normalize(ct: np.ndarray, pet: np.ndarray, norm_cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    w_lo, w_hi = norm_cfg["ct_window"]
    ct_n = (np.clip(ct, w_lo, w_hi) - w_lo) / (w_hi - w_lo)
    if norm_cfg["pet_norm"] == "zscore":
        mean, std = float(pet.mean()), float(pet.std())
        pet_n = (pet - mean) / max(std, 1e-6)
    elif norm_cfg["pet_norm"] == "clip":
        c = float(norm_cfg["suv_clip"])
        pet_n = np.clip(pet, 0, c) / c
    else:
        raise ValueError(f"unknown pet_norm {norm_cfg['pet_norm']}")
    return ct_n.astype(np.float32), pet_n.astype(np.float32)


def load_case(files: CaseFiles, data_cfg: dict, with_label: bool = True, max_fg_coords: int = 20000, seed: int = 0) -> CaseData:
    spacing = tuple(float(s) for s in data_cfg["target_spacing_xyz"])
    ct = read_lps(files.ct)
    pet = read_lps(files.pet)
    roi_lo, roi_hi = compute_roi(ct, pet, data_cfg["roi"], spacing)
    size = np.maximum(np.floor((roi_hi - roi_lo) / np.array(spacing)).astype(int) + 1, 1)
    ref = _grid(roi_lo, spacing, size)

    ct_a = sitk.GetArrayFromImage(_resample(ct, ref, sitk.sitkLinear, -1024, sitk.sitkFloat32))
    pet_a = sitk.GetArrayFromImage(_resample(pet, ref, sitk.sitkLinear, 0, sitk.sitkFloat32))
    ct_n, pet_n = _normalize(ct_a, pet_a, data_cfg)
    image = np.stack([ct_n, pet_n]).astype(np.float16)

    label = None
    fg_coords: Dict[int, np.ndarray] = {}
    if with_label and files.label is not None:
        lab = read_lps(files.label)
        label = sitk.GetArrayFromImage(_resample(lab, ref, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
        rng = np.random.default_rng(seed)
        for c in (1, 2):
            coords = np.argwhere(label == c)
            if len(coords) > max_fg_coords:
                coords = coords[rng.choice(len(coords), max_fg_coords, replace=False)]
            if len(coords):
                fg_coords[c] = coords.astype(np.int32)
    pad_values = (float(image[0].min()), float(image[1].min()))
    return CaseData(
        files.case_id, files.center, image, label, spacing, tuple(float(v) for v in roi_lo), fg_coords, pad_values, files
    )


def roi_image(case: CaseData, array_zyx: np.ndarray) -> sitk.Image:
    img = sitk.GetImageFromArray(array_zyx)
    img.SetSpacing(case.spacing_xyz)
    img.SetOrigin(case.roi_origin_xyz)
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


def restore_to_original(case: CaseData, pred_zyx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Nearest-neighbour resampling of an ROI prediction onto the original label grid (LPS).

    Returns (prediction, reference label, voxel volume in mm^3) on the original grid.
    """
    if case.files is None or case.files.label is None:
        raise ValueError(f"{case.case_id}: no original label to evaluate against")
    ref = read_lps(case.files.label)
    pred_img = roi_image(case, pred_zyx.astype(np.uint8))
    pred_orig = sitk.Resample(pred_img, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)
    gt = sitk.GetArrayFromImage(ref).astype(np.uint8)
    voxel_mm3 = float(np.prod(ref.GetSpacing()))
    return sitk.GetArrayFromImage(pred_orig), gt, voxel_mm3


def label_coverage(case: CaseData) -> float:
    """Fraction of labelled voxels of the original label that fall inside the ROI (physical check)."""
    if case.files is None or case.files.label is None:
        return 1.0
    lab = read_lps(case.files.label)
    arr = sitk.GetArrayFromImage(lab)
    idx = np.argwhere(arr > 0)
    if len(idx) == 0:
        return 1.0
    spacing = np.array(lab.GetSpacing())
    origin = np.array(lab.GetOrigin())
    direction = np.array(lab.GetDirection()).reshape(3, 3)
    pts = origin + (idx[:, ::-1] * spacing) @ direction.T  # (N, 3) xyz
    lo = np.array(case.roi_origin_xyz) - np.array(case.spacing_xyz) / 2
    shape_xyz = np.array(case.image.shape[1:][::-1])
    hi = lo + shape_xyz * np.array(case.spacing_xyz)
    inside = np.all((pts >= lo - 1e-3) & (pts <= hi + 1e-3), axis=1)
    return float(inside.mean())
