"""Remuestreo de la etiqueta a 1 x 1 x 3 mm: el actual, ganglio por ganglio y suave, en etiquetas de geometria conocida."""
import numpy as np
import SimpleITK as sitk

from bifusion3d.data import labels
from mambalite3d.data import io as mio

SP_ORIG = (1.0, 1.0, 1.0)  # etiqueta original a 1 mm
SP_WORK = (1.0, 1.0, 3.0)  # rejilla de trabajo
N = (40, 40, 60)  # x, y, z


def _image(arr, spacing=SP_ORIG):
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def _work_ref():
    # misma caja fisica que la etiqueta (bordes en -0,5 y 59,5 mm en z): centros de 3 mm en 1, 4, ..., 58
    return mio._grid((0.0, 0.0, 1.0), SP_WORK, (N[0], N[1], N[2] // 3))


def _box(arr, z0, z1, y0=15, y1=25, x0=15, x1=25, value=2):
    arr[z0:z1, y0:y1, x0:x1] = value


def test_dos_ganglios_a_un_corte_se_funden_con_el_actual_y_no_ganglio_por_ganglio():
    arr = np.zeros(N[::-1], np.uint8)
    _box(arr, 10, 23)  # z 10..22 mm
    _box(arr, 24, 36)  # z 24..35 mm: 1 mm de hueco; los centros de 3 mm (22 y 25) caen uno en cada bloque
    lab, ref = _image(arr), _work_ref()
    assert labels.count_components(arr, 2) == 2
    near = labels.resample_nearest(lab, ref)
    assert labels.count_components(near, 2) == 1  # el remuestreo actual los funde
    nodes, info = labels.resample_nodes(lab, ref)
    assert labels.count_components(nodes, 2) == 2 and info["voxeles_separados"] > 0
    # solo se quita la capa de contacto: el volumen apenas cambia
    assert (nodes == 2).sum() >= (near == 2).sum() - 10 * 10


def test_un_ganglio_que_desaparece_se_rescata():
    arr = np.zeros(N[::-1], np.uint8)
    arr[11, 20, 20] = 2  # 1 mm3, entre dos centros de la rejilla de 3 mm (z 10 y 13)
    _box(arr, 30, 45, value=1)
    lab, ref = _image(arr), _work_ref()
    assert labels.count_components(labels.resample_nearest(lab, ref), 2) == 0
    nodes, info = labels.resample_nodes(lab, ref)
    assert info["rescatados"] == 1 and labels.count_components(nodes, 2) == 1
    assert labels.count_components(nodes, 1) == 1


def test_un_ganglio_en_diagonal_que_se_parte_se_reconecta():
    arr = np.zeros(N[::-1], np.uint8)
    arr[13:20, 15:25, 10:15] = 2  # z 13..19, x 10..14
    arr[20, 15:25, 12:17] = 2  # escalones en diagonal: un solo componente en el original
    arr[21, 15:25, 14:19] = 2
    arr[22:32, 15:25, 16:21] = 2  # z 22..31, x 16..20
    lab, ref = _image(arr), _work_ref()
    assert labels.count_components(arr, 2) == 1
    assert labels.count_components(labels.resample_nearest(lab, ref), 2) == 2  # centros 19 y 22: x 10..14 y 16..20
    nodes, info = labels.resample_nodes(lab, ref)
    assert labels.count_components(nodes, 2) == 1 and info["partidos"] == 0


def test_clases_distintas_no_se_separan():
    arr = np.zeros(N[::-1], np.uint8)
    _box(arr, 10, 22, value=1)
    _box(arr, 22, 34, value=2)  # GTVn pegado al GTVp en el original
    lab, ref = _image(arr), _work_ref()
    nodes, info = labels.resample_nodes(lab, ref)
    assert info["voxeles_separados"] == 0
    assert np.array_equal(nodes, labels.resample_nearest(lab, ref))


def test_igual_al_actual_cuando_no_hay_nada_que_arreglar():
    arr = np.zeros(N[::-1], np.uint8)
    _box(arr, 9, 30, value=1)
    _box(arr, 39, 51, x0=2, x1=10)
    lab, ref = _image(arr), _work_ref()
    nodes, info = labels.resample_nodes(lab, ref)
    assert np.array_equal(nodes, labels.resample_nearest(lab, ref))
    assert info == {"rescatados": 0, "partidos": 0, "voxeles_separados": 0}


def test_etiqueta_suave_conserva_el_volumen_y_marca_los_bordes():
    arr = np.zeros(N[::-1], np.uint8)
    _box(arr, 11, 20, value=2)  # z 11..19 mm: empieza a mitad del voxel de trabajo centrado en 10 (9..12)
    lab, ref = _image(arr), _work_ref()
    frac = labels.soft_fractions(lab, ref, factors_zyx=(3, 1, 1))
    assert frac.shape == (2, 20, 40, 40)
    assert np.isclose(frac[1].sum() * 3.0, (arr == 2).sum() * 1.0)  # volumen exacto en mm3
    assert np.isclose(frac[1][3, 20, 20], 1 / 3)  # voxel de z 9..12: solo el corte 11 es ganglio
    assert np.isclose(frac[1][4, 20, 20], 1.0)
    probs = labels.soft_to_probs(frac)
    assert probs.shape == (3, 20, 40, 40) and np.allclose(probs.sum(axis=0), 1.0)
