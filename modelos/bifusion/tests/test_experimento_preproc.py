"""Experimento de preprocesado (escala del PET, profundidad del ROI y etiquetas) sobre casos sinteticos."""
import json

import numpy as np
import SimpleITK as sitk

from bifusion3d.tools import experimento_preproc as ep
from bifusion3d.tools import verify_preproc as vp
from mambalite3d.config import load_config
from test_audit import make_case


def _scale_pet(files, factor):
    pet = sitk.ReadImage(str(files.pet))
    out = sitk.GetImageFromArray(sitk.GetArrayFromImage(pet) * np.float32(factor))
    out.SetSpacing(pet.GetSpacing())
    out.SetOrigin(pet.GetOrigin())
    sitk.WriteImage(out, str(files.pet))


def test_caso_con_ganglio_bajo_el_roi(tmp_path):
    files = make_case(tmp_path, "SYN-001")
    data_cfg = load_config(None)["data"]
    row = ep.experiment_case(files, data_cfg, "val")
    assert row["pet"]["factor"] == 1.0 and "ancla_antes" not in row["pet"]
    assert row["roi_270"]["cobertura_gtvn"] < 1.0  # el ganglio de debajo queda fuera con 270 mm
    assert row["roi_300"]["cobertura_gtvn"] == 1.0 and row["roi_300"]["voxeles"] > row["roi_270"]["voxeles"]
    assert set(row["vuelta"]) == set(ep.VARIANTES)
    # la vuelta actual coincide con la de la verificacion (otro camino: load_case + restore_to_original)
    v = vp.verify_case(files, data_cfg, "val")["vuelta"]
    ref = row["vuelta"]["cercano+cercano"]
    assert abs(ref["dice_gtvp"] - v["dsc_gtvp"]) < 1e-4
    assert [ref["tp"], ref["fp"], ref["fn"]] == list(v["hungaro_tp_fp_fn"])
    assert [ref["tp_grupos"], ref["fp_grupos"], ref["fn_grupos"]] == list(v["grupos_tp_fp_fn"])
    o = row["etiquetas"]["original"]
    assert o["gtvn_comp"] == 3 and row["etiquetas"]["cercano"]["gtvn_comp"] == 2  # el de debajo esta fuera del ROI
    json.dumps(row)


def test_pet_fuera_de_suv_recupera_el_ancla(tmp_path):
    files = make_case(tmp_path, "SYN-004", node_below=False)
    _scale_pet(files, 0.01)
    row = ep.experiment_case(files, load_config(None)["data"], "train")
    p = row["pet"]
    assert p["factor"] > 50 and p["ancla_antes"] is False and p["ancla_despues"] is True


def test_main_escribe_resume_y_reanuda(tmp_path):
    root = tmp_path / "datos"
    make_case(root, "SYN-001")
    make_case(root, "SYN-002", face=False, node_below=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"seed: 1\ndata:\n  root: {json.dumps(root.as_posix())}\n", encoding="utf-8")
    out = tmp_path / "out"
    assert ep.main(["--config", str(cfg), "--out", str(out), "--workers", "1"]) == 0
    s = json.loads((out / "resumen.json").read_text(encoding="utf-8"))
    assert s["n_casos"] == 2 and not s["errores"]
    assert s["roi"]["270"]["casos_con_etiqueta_fuera"] == 1 and s["roi"]["300"]["casos_con_etiqueta_fuera"] == 0
    assert s["pet"]["corregidos"] == []
    assert set(s["vuelta"]["todos"]) == set(ep.VARIANTES)
    assert (out / "casos.csv").read_text(encoding="utf-8").count("\n") == 3
    assert ep.main(["--config", str(cfg), "--out", str(out), "--workers", "1"]) == 0  # relanzar no repite casos
    assert len((out / "casos.jsonl").read_text(encoding="utf-8").splitlines()) == 2
