"""Deteccion de un PET fuera de SUV y su correccion, en un caso sintetico con el tejido blando a SUV 1."""
import numpy as np
import SimpleITK as sitk

from bifusion3d.data import pet_scale
from mambalite3d.config import load_config
from mambalite3d.data import io as mio
from test_audit import make_case


def test_un_pet_en_suv_no_se_toca(tmp_path):
    files = make_case(tmp_path, "SYN-001")
    roi_cfg = load_config(None)["data"]["roi"]
    st = pet_scale.pet_stats(mio.read_lps(files.ct), mio.read_lps(files.pet), roi_cfg)
    assert abs(st["suv_tejido"] - 1.0) < 0.05  # cuerpo a 40 HU y SUV 1
    assert st["suv_craneal_p99"] > 5  # el "cerebro" del caso sintetico (SUV 6)
    assert pet_scale.scale_factor(st["suv_tejido"]) == 1.0


def test_un_pet_fuera_de_suv_se_detecta_y_se_corrige(tmp_path):
    files = make_case(tmp_path, "SYN-002")
    ct, pet = mio.read_lps(files.ct), mio.read_lps(files.pet)
    small = pet_scale.scaled(pet, 1 / 12)  # como CHUM-010
    roi_cfg = load_config(None)["data"]["roi"]
    st = pet_scale.pet_stats(ct, small, roi_cfg)
    factor = pet_scale.scale_factor(st["suv_tejido"])
    assert abs(factor - pet_scale.REFERENCIA * 12) < 0.5
    fixed = pet_scale.scaled(small, factor)
    assert np.allclose(sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(pet) * factor / 12, rtol=1e-5)
    assert fixed.GetSpacing() == pet.GetSpacing() and fixed.GetOrigin() == pet.GetOrigin()


def test_rango_y_valores_raros():
    assert pet_scale.scale_factor(0.8) == 1.0
    assert pet_scale.scale_factor(5.0) == pet_scale.REFERENCIA / 5.0
    assert pet_scale.scale_factor(float("nan")) == 1.0
    assert pet_scale.scale_factor(0.0) == 1.0
