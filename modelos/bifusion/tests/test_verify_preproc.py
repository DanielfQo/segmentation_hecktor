"""Verificacion del preprocesamiento real (load_case) en casos sinteticos de geometria conocida."""
import json

import numpy as np
import SimpleITK as sitk

from bifusion3d.tools import verify_preproc as vp
from mambalite3d.config import load_config
from mambalite3d.data import io as mio
from test_audit import make_case


def _scale_pet(files, factor):
    pet = sitk.ReadImage(str(files.pet))
    out = sitk.GetImageFromArray(sitk.GetArrayFromImage(pet) * np.float32(factor))
    out.SetSpacing(pet.GetSpacing())
    out.SetOrigin(pet.GetOrigin())
    sitk.WriteImage(out, str(files.pet))


def test_la_formula_coincide_con_compute_roi(tmp_path):
    data_cfg = load_config(None)["data"]
    spacing = tuple(float(s) for s in data_cfg["target_spacing_xyz"])
    for cid, scale in (("SYN-001", 1.0), ("SYN-002", 0.1)):  # con ancla y sin ancla (SUV maximo 1,2)
        files = make_case(tmp_path, cid)
        if scale != 1.0:
            _scale_pet(files, scale)
        ct, pet = mio.read_lps(files.ct), mio.read_lps(files.pet)
        f = vp.roi_formula(ct, pet, data_cfg["roi"], spacing)
        lo, hi = mio.compute_roi(ct, pet, data_cfg["roi"], spacing)
        assert np.allclose(f["lo"], lo) and np.allclose(f["hi"], hi)
        assert f["ancla"] is (scale == 1.0)


def test_un_caso_correcto_pasa_todo(tmp_path):
    files = make_case(tmp_path, "SYN-001")
    row = vp.verify_case(files, load_config(None)["data"], "train", det=True)
    assert row["fallos"] == []
    assert row["ancla"] and row["determinismo"]
    assert "roi_cobertura" in row["avisos"]  # el ganglio de debajo del ROI queda fuera, como en el caso sintetico
    assert row["roi_formula_dif_mm"] <= vp.ROI_TOL_MM
    assert row["gtvn_componentes"][0] == 3 and row["gtvn_componentes"][1] == 2
    assert "etiqueta_componentes" not in row["avisos"]  # el ganglio que falta esta fuera del ROI, no perdido
    assert "etiqueta_volumen" not in row["avisos"]  # el volumen se compara con lo que cae dentro del ROI
    tp, fp, fn = row["vuelta"]["hungaro_tp_fp_fn"]
    assert (fp, fn) == (0, 1)
    json.dumps(row)


def test_detecta_un_roi_que_no_sigue_la_formula(tmp_path, monkeypatch):
    files = make_case(tmp_path, "SYN-003", node_below=False)
    real = mio.compute_roi

    def shifted(ct, pet, roi_cfg, spacing):
        lo, hi = real(ct, pet, roi_cfg, spacing)
        return lo + np.array([0.0, 0.0, 6.0]), hi  # un ROI 6 mm mas corto por abajo
    monkeypatch.setattr(mio, "compute_roi", shifted)
    row = vp.verify_case(files, load_config(None)["data"], "train")
    assert "roi_formula" in row["fallos"]


def test_avisa_de_un_pet_que_no_esta_en_suv(tmp_path):
    files = make_case(tmp_path, "SYN-004", node_below=False)
    _scale_pet(files, 0.01)
    row = vp.verify_case(files, load_config(None)["data"], "train")
    assert {"pet_suv", "roi_ancla"} <= set(row["avisos"])
    assert "pet_zscore" not in row["fallos"]  # el z-score lo disimula, por eso hace falta el aviso


def test_main_escribe_y_reanuda(tmp_path):
    root = tmp_path / "datos"
    make_case(root, "SYN-001")
    make_case(root, "SYN-002", face=False, node_below=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"seed: 1\ndata:\n  root: {json.dumps(root.as_posix())}\n", encoding="utf-8")
    out = tmp_path / "out"
    assert vp.main(["--config", str(cfg), "--out", str(out), "--workers", "1", "--det-every", "1"]) == 0
    s = json.loads((out / "resumen.json").read_text(encoding="utf-8"))
    assert s["n_casos"] == 2 and s["veredicto"] == "ok"
    assert s["comprobaciones"]["roi_formula"]["estado"] == "ok"
    assert s["comprobaciones"]["determinismo"]["casos_comprobados"] == 2
    assert s["comprobaciones"]["roi_cobertura"]["casos"] == ["SYN-001"]
    assert s["comprobaciones"]["etiqueta_volumen"]["estado"] == "ok"
    assert vp.main(["--config", str(cfg), "--out", str(out), "--workers", "1"]) == 0  # relanzar no repite casos
    assert len((out / "casos.jsonl").read_text(encoding="utf-8").splitlines()) == 2
