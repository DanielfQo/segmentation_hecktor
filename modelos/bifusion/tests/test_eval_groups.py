"""Regla de grupos para la deteccion de ganglios (articulo del dataset HECKTOR 2025, Fig. 5) frente al emparejamiento
hungaro de mambalite3d, en mascaras sinteticas."""
import numpy as np

from bifusion3d.eval_groups import match_groups
from mambalite3d.eval.metrics import match_components

SHAPE = (40, 60, 60)


def ball(center, r):
    z, y, x = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    return (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= r * r


def test_uno_a_uno_igual_en_las_dos_reglas():
    gt = ball((20, 30, 20), 5)
    pred = ball((20, 30, 21), 5)
    assert match_groups(pred, gt) == (1, 0, 0) == match_components(pred, gt)


def test_una_prediccion_que_une_dos_ganglios_es_un_tp_por_grupos():
    g1, g2 = ball((20, 30, 18), 4), ball((20, 30, 30), 4)
    pred = np.zeros(SHAPE, bool)
    pred[16:25, 26:35, 14:35] = True  # una caja que tapa los dos ganglios
    gt = g1 | g2
    inter, union = (pred & gt).sum(), (pred | gt).sum()
    assert inter / union >= 0.3  # IoU compuesto del grupo
    assert match_groups(pred, gt) == (1, 0, 0)
    tp, fp, fn = match_components(pred, gt)  # hungaro: la caja no llega a IoU 0,3 con ninguno de los dos
    assert (tp, fp, fn) == (0, 1, 2)


def test_un_ganglio_partido_en_dos_predicciones_es_un_tp_por_grupos():
    gt = ball((20, 30, 30), 8)
    pred = gt.copy()
    pred[:, :, 30] = False  # parte la prediccion en dos mitades sin contacto
    assert match_groups(pred, gt) == (1, 0, 0)
    assert match_components(pred, gt) == (1, 1, 0)  # hungaro: una mitad empareja, la otra es FP


def test_grupo_con_poco_solape_cuenta_fp_y_fn():
    gt = ball((20, 30, 30), 6)
    pred = ball((20, 30, 40), 6)  # se tocan poco: IoU < 0,3
    assert match_groups(pred, gt) == (0, 1, 1)


def test_aislados_y_vacios():
    gt = ball((10, 15, 15), 3)
    pred = ball((30, 45, 45), 3)
    assert match_groups(pred, gt) == (0, 1, 1)
    assert match_groups(np.zeros(SHAPE, bool), gt) == (0, 0, 1)
    assert match_groups(pred, np.zeros(SHAPE, bool)) == (0, 1, 0)
    assert match_groups(np.zeros(SHAPE, bool), np.zeros(SHAPE, bool)) == (0, 0, 0)
