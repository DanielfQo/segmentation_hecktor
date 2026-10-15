"""Official HECKTOR Task 1 metrics of the 2022 and 2025 editions, and how each edition ranks submissions.

HECKTOR 2022 (overview by Andrearczyk et al.; hecktor.grand-challenge.org/Evaluation):
    aggregated Dice for GTVp and for GTVn, DSCagg = 2 * sum_i |P_i n G_i| / sum_i (|P_i| + |G_i|) over all
    patients; submissions are ranked by the mean of the two.
HECKTOR 2025 (hecktor25.grand-challenge.org/tasks-and-evaluation):
    GTVp: Dice per patient averaged over all patients (1 if prediction and reference are both empty, 0 if only
    one of them is); GTVn: aggregated Dice and aggregated detection F1 = 2TP / (2TP + FP + FN), a node counting
    as detected when IoU > 0.3. Ranking: one rank on GTVp, one rank on GTVn (Borda count of the DSCagg and F1agg
    ranks) and the Borda count of those two ranks.

The values themselves are computed by eval.metrics (evaluate_case / aggregate); this module names them by edition
and ranks a set of models. Among a handful of our own models the Borda count only orders those models.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy.stats import rankdata

NAN = float("nan")


def hecktor2025(agg: Dict) -> Dict[str, float]:
    """The three HECKTOR 2025 Task 1 metrics from an eval.metrics.aggregate() dict."""
    return {
        "GTVp_DSC": float(agg["gtvp_dice_mean"]),
        "GTVn_DSCagg": float(agg["gtvn_dice_agg"]),
        "GTVn_F1agg": float(agg["gtvn_f1_agg"]),
    }


def hecktor2022(agg: Dict) -> Dict[str, float]:
    """The HECKTOR 2022 Task 1 metrics; NaN for GTVp when the evaluation predates the GTVp volumes."""
    p = float(agg.get("gtvp_dice_agg", NAN))
    n = float(agg["gtvn_dice_agg"])
    return {"GTVp_DSCagg": p, "GTVn_DSCagg": n, "DSCagg_media": 0.5 * (p + n)}


def _rank(values, higher_is_better: bool = True) -> np.ndarray:
    """1 = best; ties share the average rank; NaN values get NaN and do not take part."""
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), NAN)
    ok = np.isfinite(v)
    if ok.any():
        out[ok] = rankdata(-v[ok] if higher_is_better else v[ok], method="average")
    return out


def ranking_2025(rows: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """rows: model -> hecktor2025() dict. Returns model -> ranks (GTVp, GTVn DSCagg, GTVn F1agg, GTVn, final)."""
    names = list(rows)
    r_p = _rank([rows[n]["GTVp_DSC"] for n in names])
    r_nd = _rank([rows[n]["GTVn_DSCagg"] for n in names])
    r_nf = _rank([rows[n]["GTVn_F1agg"] for n in names])
    r_n = _rank(r_nd + r_nf, higher_is_better=False)  # Borda count of the two GTVn rankings
    final = _rank(r_p + r_n, higher_is_better=False)  # Borda count of the GTVp and GTVn rankings
    # the grand-challenge leaderboard of 2025 also shows the mean position over the three metric columns
    mean_pos = _rank((r_p + r_nd + r_nf) / 3, higher_is_better=False)
    return {n: {"puesto_GTVp": r_p[i], "puesto_GTVn_DSCagg": r_nd[i], "puesto_GTVn_F1agg": r_nf[i],
                "puesto_GTVn": r_n[i], "puesto_final": final[i], "puesto_posicion_media": mean_pos[i]}
            for i, n in enumerate(names)}


def ranking_2022(rows: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """rows: model -> hecktor2022() dict. Returns model -> rank by the mean aggregated Dice (NaN if missing)."""
    names = list(rows)
    r = _rank([rows[n]["DSCagg_media"] for n in names])
    return {n: r[i] for i, n in enumerate(names)}


def selection_score(agg: Dict, edition: str = "2025") -> float:
    """Single number used to choose a checkpoint on validation: 2025 -> mean of GTVp DSC and GTVn DSCagg
    (our summary; the official 2025 ranking has no single score); 2022 -> mean aggregated Dice."""
    if edition == "2022":
        return hecktor2022(agg)["DSCagg_media"]
    m = hecktor2025(agg)
    return 0.5 * (m["GTVp_DSC"] + m["GTVn_DSCagg"])


def fmt(x: float, nd: int = 3) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}".replace(".", ",")


def fmt_rank(x: float) -> str:
    """Ranks: whole numbers without decimals, shared (tied) ranks with one decimal."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return str(int(x)) if float(x).is_integer() else fmt(x, 1)
