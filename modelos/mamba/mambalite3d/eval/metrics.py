"""HECKTOR Task 1 metrics (2025 ranking, plus the 2022 ranking metric for comparison with published work).

HECKTOR 2025:
- GTVp: Dice per patient, averaged. Primary convention: Dice = 1 when prediction and reference are
  both empty, 0 when only one is. Secondary: mean over patients whose reference contains GTVp.
- GTVn aggregated Dice: 2 * sum(|P n G|) / sum(|P| + |G|) over all patients, volumes in mm^3
  (as DiceAggScore in the official HECKTOR 2025 baseline repository).
- GTVn aggregated F1 (detection): connected components (26-connectivity), one-to-one matching
  with the Hungarian algorithm, a match counts as TP when IoU > 0.3; TP/FP/FN summed over patients.

HECKTOR 2022 (overview paper, Andrearczyk et al.): aggregated Dice for both classes, ranked by their mean:
- gtvp_dice_agg: same aggregated formula as GTVn, applied to GTVp.
- dsc_agg_mean = (gtvp_dice_agg + gtvn_dice_agg) / 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

STRUCT_26 = np.ones((3, 3, 3), dtype=bool)


def dice_binary(p: np.ndarray, g: np.ndarray) -> float:
    ps, gs = int(p.sum()), int(g.sum())
    if ps + gs == 0:
        return 1.0
    return 2.0 * int(np.logical_and(p, g).sum()) / (ps + gs)


def match_components(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.3):
    """Returns (TP, FP, FN) for binary masks using 26-connected components."""
    lp, n_p = ndimage.label(pred, structure=STRUCT_26)
    lg, n_g = ndimage.label(gt, structure=STRUCT_26)
    if n_p == 0 or n_g == 0:
        return 0, n_p, n_g
    size_p = np.bincount(lp.ravel(), minlength=n_p + 1)
    size_g = np.bincount(lg.ravel(), minlength=n_g + 1)
    both = (lp > 0) & (lg > 0)
    pair = lg[both].astype(np.int64) * (n_p + 1) + lp[both].astype(np.int64)
    inter = np.bincount(pair, minlength=(n_g + 1) * (n_p + 1)).reshape(n_g + 1, n_p + 1)[1:, 1:]
    union = size_g[1:, None] + size_p[None, 1:] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    rows, cols = linear_sum_assignment(-iou)
    tp = int(np.sum(iou[rows, cols] > iou_threshold))
    return tp, n_p - tp, n_g - tp


@dataclass
class CaseResult:
    case_id: str
    dice_gtvp: float
    gtvp_in_ref: bool
    gtvn_inter_mm3: float
    gtvn_sum_mm3: float
    gtvn_tp: int
    gtvn_fp: int
    gtvn_fn: int
    dice_gtvn_case: float
    # GTVp volumes for the aggregated (HECKTOR 2022) Dice; NaN when read from older per-case CSVs.
    gtvp_inter_mm3: float = float("nan")
    gtvp_sum_mm3: float = float("nan")


def evaluate_case(case_id: str, pred: np.ndarray, gt: np.ndarray, voxel_mm3: float) -> CaseResult:
    p1, g1 = pred == 1, gt == 1
    p2, g2 = pred == 2, gt == 2
    inter1 = int(np.logical_and(p1, g1).sum()) * voxel_mm3
    sum1 = (int(p1.sum()) + int(g1.sum())) * voxel_mm3
    inter2 = int(np.logical_and(p2, g2).sum()) * voxel_mm3
    sum2 = (int(p2.sum()) + int(g2.sum())) * voxel_mm3
    tp, fp, fn = match_components(p2, g2)
    return CaseResult(
        case_id=case_id,
        dice_gtvp=dice_binary(p1, g1),
        gtvp_in_ref=bool(g1.any()),
        gtvn_inter_mm3=inter2,
        gtvn_sum_mm3=sum2,
        gtvn_tp=tp,
        gtvn_fp=fp,
        gtvn_fn=fn,
        dice_gtvn_case=dice_binary(p2, g2),
        gtvp_inter_mm3=inter1,
        gtvp_sum_mm3=sum1,
    )


def _agg_dice(inter: float, total: float) -> float:
    return float(2 * inter / total) if total > 0 else 1.0


def aggregate(results: List[CaseResult]) -> Dict[str, float]:
    if not results:
        return {}
    dsc_p = [r.dice_gtvp for r in results]
    dsc_p_ref = [r.dice_gtvp for r in results if r.gtvp_in_ref]
    inter = sum(r.gtvn_inter_mm3 for r in results)
    total = sum(r.gtvn_sum_mm3 for r in results)
    tp = sum(r.gtvn_tp for r in results)
    fp = sum(r.gtvn_fp for r in results)
    fn = sum(r.gtvn_fn for r in results)
    gtvn_agg = _agg_dice(inter, total)
    vols_p = [(r.gtvp_inter_mm3, r.gtvp_sum_mm3) for r in results]
    if all(np.isfinite(i) and np.isfinite(s) for i, s in vols_p):
        gtvp_agg = _agg_dice(sum(i for i, _ in vols_p), sum(s for _, s in vols_p))
    else:  # older per-case files without GTVp volumes
        gtvp_agg = float("nan")
    return {
        "n_cases": len(results),
        "gtvp_dice_mean": float(np.mean(dsc_p)),
        "gtvp_dice_mean_ref_only": float(np.mean(dsc_p_ref)) if dsc_p_ref else float("nan"),
        "gtvn_dice_agg": gtvn_agg,
        "gtvn_f1_agg": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 1.0,
        "gtvn_tp": tp,
        "gtvn_fp": fp,
        "gtvn_fn": fn,
        "gtvp_dice_agg": gtvp_agg,
        "dsc_agg_mean": float(0.5 * (gtvp_agg + gtvn_agg)),
    }
