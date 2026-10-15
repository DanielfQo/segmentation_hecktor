"""Vuelta de la prediccion del ROI a la rejilla original: la actual (cercana) y la lineal de las probabilidades."""
import numpy as np
import SimpleITK as sitk

from bifusion3d.data import restore
from mambalite3d.config import load_config
from mambalite3d.data import io as mio
from test_audit import make_case


def test_la_cercana_es_igual_a_restore_to_original(tmp_path):
    files = make_case(tmp_path, "SYN-001")
    case = mio.load_case(files, load_config(None)["data"])
    ref = mio.read_lps(files.label)
    expected, gt, vox = mio.restore_to_original(case, case.label)
    got = restore.restore_label_nearest(case.label, case.roi_origin_xyz, case.spacing_xyz, ref)
    assert np.array_equal(got, expected)
    probs = restore.one_hot(case.label)
    p2, gt2, vox2 = restore.restore_case(case, probs, "nearest")
    assert np.array_equal(p2, expected) and np.array_equal(gt2, gt) and vox2 == vox


def test_lineal_de_una_etiqueta_dura_se_parece_a_la_cercana_y_respeta_el_roi(tmp_path):
    files = make_case(tmp_path, "SYN-001")
    case = mio.load_case(files, load_config(None)["data"])
    near, gt, _ = restore.restore_case(case, restore.one_hot(case.label), "nearest")
    lin, gt_l, _ = restore.restore_case(case, restore.one_hot(case.label), "linear")
    assert np.array_equal(gt, gt_l)
    for c in (1, 2):
        a, b = near == c, lin == c
        assert 2 * (a & b).sum() / (a.sum() + b.sum()) > 0.9
    # nada fuera del ROI: el ganglio de debajo sigue sin prediccion en las dos
    lo, hi = restore.roi_box(case.roi_origin_xyz, case.spacing_xyz, case.label.shape)
    zmm = np.arange(gt.shape[0]) * 1.5
    assert not lin[zmm < lo[2] - 1.5].any() and not near[zmm < lo[2] - 1.5].any()


def test_empate_va_al_primer_plano_y_la_caja_recorta():
    ref = mio._grid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (10, 10, 10))
    probs = np.zeros((3, 4, 10, 10), np.float32)
    probs[0] = 0.5
    probs[2] = 0.5  # empate fondo / ganglio en todo el ROI
    lab = restore.restore_probs_linear(probs, (0.0, 0.0, 0.0), (1.0, 1.0, 3.0), ref)
    assert set(np.unique(lab)) == {2}  # el ROI de 4 cortes de 3 mm cubre z -1,5..10,5: toda la rejilla
    box = (np.array([-0.5, -0.5, -0.5]), np.array([4.5, 4.5, 4.5]))
    cut = restore.restore_probs_linear(probs, (0.0, 0.0, 0.0), (1.0, 1.0, 3.0), ref, box)
    assert cut[:5, :5, :5].all()  # la caja cubre los indices 0..4; se remuestrea con 1 voxel de holgura
    assert not cut[6:].any() and not cut[:, 6:].any() and not cut[:, :, 6:].any()


def test_index_box_cubre_la_caja():
    img = sitk.Image([20, 20, 20], sitk.sitkUInt8)
    img.SetSpacing((2.0, 2.0, 2.0))
    lo, hi = restore.index_box(img, np.array([3.0, 3.0, 3.0]), np.array([9.0, 9.0, 9.0]), pad=0)
    assert list(lo) == [2, 2, 2] and list(hi) == [5, 5, 5]  # centros en 4, 6 y 8 mm
