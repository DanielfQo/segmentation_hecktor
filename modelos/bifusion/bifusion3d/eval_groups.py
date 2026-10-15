"""Deteccion de ganglios con la regla de grupos del articulo del dataset HECKTOR 2025 (Saeed et al., 2025, Fig. 5).

Las componentes conexas de prediccion y de referencia que se tocan forman un grupo (1-1, 1-N, M-1 o M-N). Un grupo
con prediccion y referencia cuenta como UN verdadero positivo si su IoU compuesto (union de sus predicciones frente a
union de sus referencias) es >= 0,3; si no, todas sus predicciones son FP y todas sus referencias FN. Una referencia
sin prediccion es FN y una prediccion sin referencia es FP.

mambalite3d.eval.metrics.match_components empareja uno a uno (algoritmo hungaro, IoU > 0,3): con una prediccion que
une dos ganglios cuenta como mucho 1 TP y deja el otro como FN, mientras que aqui el grupo cuenta como 1 TP. El codigo
oficial de la evaluacion no es publico; las dos cuentas se informan juntas.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage

STRUCT_26 = np.ones((3, 3, 3), dtype=bool)


def match_groups(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.3) -> Tuple[int, int, int]:
    """(TP, FP, FN) de mascaras binarias con componentes de 26 vecinos agrupadas por solape."""
    lp, n_p = ndimage.label(pred, structure=STRUCT_26)
    lg, n_g = ndimage.label(gt, structure=STRUCT_26)
    if n_p == 0 or n_g == 0:
        return 0, int(n_p), int(n_g)
    size_p = np.bincount(lp.ravel(), minlength=n_p + 1)
    size_g = np.bincount(lg.ravel(), minlength=n_g + 1)
    both = (lp > 0) & (lg > 0)
    pairs, inter = np.unique(np.stack([lg[both], lp[both]]), axis=1, return_counts=True)

    parent = list(range(n_g + n_p + 1))  # nodos: referencias 1..n_g, predicciones n_g+1..n_g+n_p

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for g, p in pairs.T:
        ra, rb = find(int(g)), find(n_g + int(p))
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for g in range(1, n_g + 1):
        groups.setdefault(find(g), [[], [], 0])[0].append(g)
    for p in range(1, n_p + 1):
        groups.setdefault(find(n_g + p), [[], [], 0])[1].append(p)
    for (g, p), n in zip(pairs.T, inter):
        groups[find(int(g))][2] += int(n)
    tp = fp = fn = 0
    for gs, ps, n_inter in groups.values():
        if not ps:
            fn += len(gs)
        elif not gs:
            fp += len(ps)
        else:
            union = int(size_g[gs].sum() + size_p[ps].sum()) - n_inter
            if union > 0 and n_inter / union >= iou_threshold:
                tp += 1
            else:
                fp += len(ps)
                fn += len(gs)
    return tp, fp, fn
