"""Preprocesado v2: por defecto identico a mambalite3d; ancla relativa, etiquetas ganglio por ganglio y suaves."""
import numpy as np

from bifusion3d.data import labels as L
from bifusion3d.data import preproc
from mambalite3d.config import load_config
from mambalite3d.data import io as mio
from test_audit import make_case
from test_experimento_preproc import _scale_pet


def _cfg(**kw):
    return preproc.with_options(load_config(None)["data"], **kw)


def test_por_defecto_es_load_case_de_mamba(tmp_path):
    files = make_case(tmp_path, "SYN-001")
    a = mio.load_case(files, _cfg())
    b = preproc.load_case_v2(files, _cfg())
    assert np.array_equal(a.image, b.image) and np.array_equal(a.label, b.label)
    assert a.roi_origin_xyz == b.roi_origin_xyz and a.pad_values == b.pad_values
    assert set(a.fg_coords) == set(b.fg_coords)
    assert all(np.array_equal(a.fg_coords[c], b.fg_coords[c]) for c in a.fg_coords)
    assert not hasattr(b, "soft")


def test_ancla_relativa_no_depende_de_la_escala_del_pet(tmp_path):
    files = make_case(tmp_path, "SYN-001", node_below=False)
    rel = preproc.load_case_v2(files, _cfg(ancla="relativa"))
    absol = preproc.load_case_v2(files, _cfg())
    _scale_pet(files, 0.05)  # PET fuera de SUV: con el umbral fijo de 2 no hay ancla
    rel2 = preproc.load_case_v2(files, _cfg(ancla="relativa"))
    abs2 = preproc.load_case_v2(files, _cfg())
    assert rel2.roi_origin_xyz == rel.roi_origin_xyz and rel2.image.shape == rel.image.shape
    assert abs2.image.shape[1] > absol.image.shape[1]  # sin ancla el ROI toma todo el cuerpo en z
    assert np.allclose(rel2.image[1].astype(float), rel.image[1].astype(float), atol=2e-2)  # z-score: igual


def test_ancla_relativa_con_tejido_tipico_da_los_umbrales_de_siempre(tmp_path):
    files = make_case(tmp_path, "SYN-001")
    ct, pet = mio.read_lps(files.ct), mio.read_lps(files.pet)
    roi = _cfg()["roi"]
    t = preproc.pet_stats(ct, pet, roi)["suv_tejido"]
    cfg, escala = preproc.roi_cfg_relativo(ct, pet, roi, ref=t)
    assert escala == 1.0 and cfg["suv_anchor"] == roi["suv_anchor"] and cfg["body_suv"] == roi["body_suv"]


def test_etiquetas_nodos_y_suaves(tmp_path):
    files = make_case(tmp_path, "SYN-001")
    near = preproc.load_case_v2(files, _cfg())
    nodos = preproc.load_case_v2(files, _cfg(etiquetas="nodos"))
    suave = preproc.load_case_v2(files, _cfg(etiquetas="suave"))
    ref = mio.roi_image(near, near.label)
    expected, _ = L.resample_nodes(mio.read_lps(files.label), ref)
    assert np.array_equal(nodos.label, expected) and np.array_equal(suave.label, expected)
    assert suave.soft.dtype == np.uint8 and suave.soft.shape[0] == 2
    full = L.soft_fractions(mio.read_lps(files.label), ref)
    # un parche que cubre todo el ROI devuelve las fracciones completas (redondeadas a 1/255)
    shape = near.label.shape
    center = [s // 2 for s in shape]
    patch = preproc.soft_patch(suave, center, shape).astype(float) / 255
    assert np.abs(patch - full).max() <= 0.5 / 255 + 1e-6
    # y donde la fraccion es alta, la etiqueta dura es de esa clase
    for c in (1, 2):
        assert (suave.label[full[c - 1] > 0.9] == c).mean() > 0.95
    # un parche fuera de la etiqueta es todo ceros
    assert not preproc.soft_patch(suave, (0, 0, 0), (4, 4, 4)).any()
