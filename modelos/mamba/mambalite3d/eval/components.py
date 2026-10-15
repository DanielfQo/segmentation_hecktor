"""Connected-component statistics of a prediction, used to study and tune post-processing.

`component_stats` summarises one case in a small JSON-friendly dict: for each class (1 = GTVp,
2 = GTVn) every predicted component with its size, model confidence and overlap with the reference,
and every reference component with its size. `score_cases` recomputes the HECKTOR metrics after
dropping components (any keep-rule), so post-processing rules can be tuned on validation without
running the network again.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

STRUCT_26 = np.ones((3, 3, 3), dtype=bool)


def component_stats(pred: np.ndarray, gt: np.ndarray, probs: Optional[np.ndarray] = None,
                    pet: Optional[np.ndarray] = None, voxel_mm3: float = 1.0) -> Dict:
    """pred, gt: (Z, Y, X) uint8 in {0, 1, 2}; probs: (3, Z, Y, X) softmax; pet: (Z, Y, X) normalised PET."""
    out = {"voxel_mm3": voxel_mm3, "confusion": np.bincount(gt.astype(np.int64).ravel() * 3 + pred.ravel(), minlength=9).reshape(3, 3).tolist()}
    for c in (1, 2):
        lab_p, n_p = ndimage.label(pred == c, structure=STRUCT_26)
        lab_g, n_g = ndimage.label(gt == c, structure=STRUCT_26)
        size_p = np.bincount(lab_p.ravel(), minlength=n_p + 1)[1:]
        size_g = np.bincount(lab_g.ravel(), minlength=n_g + 1)[1:]
        idx = np.arange(1, n_p + 1)
        comps = []
        if n_p:
            both = (lab_p > 0) & (lab_g > 0)
            pair = lab_p[both].astype(np.int64) * (n_g + 1) + lab_g[both].astype(np.int64)
            ov = np.bincount(pair, minlength=(n_p + 1) * (n_g + 1)).reshape(n_p + 1, n_g + 1)[1:, 1:]
            other = ndimage.sum_labels(gt == (3 - c), lab_p, index=idx)
            maxp = ndimage.maximum(probs[c], lab_p, index=idx) if probs is not None else np.ones(n_p)
            meanp = ndimage.mean(probs[c], lab_p, index=idx) if probs is not None else np.ones(n_p)
            petmax = ndimage.maximum(pet, lab_p, index=idx) if pet is not None else np.zeros(n_p)
            coms = ndimage.center_of_mass(np.ones_like(lab_p, dtype=np.uint8), lab_p, idx)
            for i in range(n_p):
                nz = np.nonzero(ov[i])[0]
                comps.append({
                    "vox": int(size_p[i]),
                    "inter": int(ov[i].sum()),
                    "other": int(other[i]),
                    "maxp": round(float(maxp[i]), 4),
                    "meanp": round(float(meanp[i]), 4),
                    "petmax": round(float(petmax[i]), 3),
                    "com": [round(float(v), 1) for v in coms[i]],
                    "ov": [[int(j), int(ov[i, j])] for j in nz],
                })
        out[str(c)] = {"pred": comps, "gt": [int(s) for s in size_g]}
    return out


def _dice(inter: float, sp: float, sg: float) -> float:
    return 1.0 if sp + sg == 0 else 2.0 * inter / (sp + sg)


def _f1_counts(kept: List[dict], gt_sizes: List[int], iou_thr: float = 0.3):
    n_p, n_g = len(kept), len(gt_sizes)
    if n_p == 0 or n_g == 0:
        return 0, n_p, n_g
    iou = np.zeros((n_g, n_p))
    for i, comp in enumerate(kept):
        for j, inter in comp["ov"]:
            iou[j, i] = inter / (gt_sizes[j] + comp["vox"] - inter)
    rows, cols = linear_sum_assignment(-iou)
    tp = int(np.sum(iou[rows, cols] > iou_thr))
    return tp, n_p - tp, n_g - tp


def score_cases(cases: List[Dict], keep_p: Callable = None, keep_n: Callable = None) -> Dict[str, float]:
    """HECKTOR metrics after keeping only the components accepted by keep_p / keep_n.

    keep_x(comp, comps_of_case) -> bool. GTVp: mean DSC over all patients (1 if both empty);
    GTVn: aggregated DSC and aggregated F1 (IoU > 0.3, Hungarian matching). Also the HECKTOR 2022
    metric: GTVp aggregated DSC and dsc_agg_mean (mean of both aggregated DSC).
    Note: removed voxels become background (the other class is not filled in).
    """
    keep_p = keep_p or (lambda comp, comps: True)
    keep_n = keep_n or (lambda comp, comps: True)
    d_p, d_p_ref = [], []
    inter_p = sum_p = 0.0
    inter_n = sum_n = 0.0
    tp = fp = fn = 0
    for case in cases:
        v = case["voxel_mm3"]
        comps_p = case["1"]["pred"]
        kept = [c for c in comps_p if keep_p(c, comps_p)]
        gt_p = sum(case["1"]["gt"])
        d = _dice(sum(c["inter"] for c in kept), sum(c["vox"] for c in kept), gt_p)
        inter_p += sum(c["inter"] for c in kept) * v
        sum_p += (sum(c["vox"] for c in kept) + gt_p) * v
        d_p.append(d)
        if gt_p > 0:
            d_p_ref.append(d)
        comps_n = case["2"]["pred"]
        kept_n = [c for c in comps_n if keep_n(c, comps_n)]
        inter_n += sum(c["inter"] for c in kept_n) * v
        sum_n += (sum(c["vox"] for c in kept_n) + sum(case["2"]["gt"])) * v
        a, b, c_ = _f1_counts(kept_n, case["2"]["gt"])
        tp, fp, fn = tp + a, fp + b, fn + c_
    gtvp_agg = float(2 * inter_p / sum_p) if sum_p > 0 else 1.0
    gtvn_agg = float(2 * inter_n / sum_n) if sum_n > 0 else 1.0
    return {
        "gtvp_dice_mean": float(np.mean(d_p)),
        "gtvp_dice_mean_ref_only": float(np.mean(d_p_ref)) if d_p_ref else float("nan"),
        "gtvn_dice_agg": gtvn_agg,
        "gtvn_f1_agg": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0,
        "gtvn_tp": tp, "gtvn_fp": fp, "gtvn_fn": fn,
        "gtvp_dice_agg": gtvp_agg,
        "dsc_agg_mean": 0.5 * (gtvp_agg + gtvn_agg),
    }


def postprocess_prediction(pred: np.ndarray, probs: Optional[np.ndarray], rules: Dict, voxel_mm3: float) -> np.ndarray:
    """Applies the rules of tools/tune_postprocess (same semantics) to a voxel prediction (Z, Y, X)."""
    out = pred.copy()
    for cls, key in ((1, "gtvp"), (2, "gtvn")):
        r = rules.get(key) or {}
        if not r:
            continue
        lab, n = ndimage.label(pred == cls, structure=STRUCT_26)
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        ml = np.bincount(lab.ravel(), minlength=n + 1)[1:] * voxel_mm3 / 1000.0
        maxp = ndimage.maximum(probs[cls], lab, index=idx) if probs is not None else np.ones(n)
        keep = (ml >= r.get("min_ml", 0)) & (maxp >= r.get("min_maxp", 0))
        if cls == 1 and r.get("largest") and keep.any():
            largest = int(np.argmax(np.where(keep, ml, -1)))
            keep = np.zeros(n, bool)
            keep[largest] = True
        if cls == 1 and ml[keep].sum() < r.get("case_min_ml", 0):
            keep[:] = False
        drop = idx[~keep]
        if len(drop):
            out[np.isin(lab, drop)] = 0
    return out
