"""Auditoria del preprocesamiento (Fase 0) sobre casos sinteticos de geometria conocida.

Un cilindro de cuerpo con "cerebro" arriba (SUV 6), un GTVp esferico de 10 mm de radio (60 HU, SUV 12) y tres
ganglios; el tercero queda por debajo del ROI (270 mm bajo el ancla), asi que la cobertura de GTVn no es 1.
Con face=True, el casquete superior del GTVp tiene -1000 HU, como si lo hubiera tocado el borrado de rostro.
"""
import csv
import json

import numpy as np
import pytest
import SimpleITK as sitk

from bifusion3d.tools import audit_data as ad
from mambalite3d.config import load_config
from mambalite3d.data import io as mio

CT_SP, CT_N = (0.98, 0.98, 1.5), (100, 100, 220)  # xyz: 98 x 98 x 330 mm
PET_SP, PET_N = (4.0, 4.0, 3.0), (25, 25, 110)
BODY_C, BODY_R = (49.0, 49.0), 40.0
GTVP = ((49.0, 49.0, 180.0), 10.0)
NODES = (((69.0, 49.0, 150.0), 4.0), ((29.0, 49.0, 120.0), 2.5))
NODE_BELOW = ((49.0, 30.0, 12.0), 4.0)


def _grid_mm(n, sp):
    x = (np.arange(n[0]) * sp[0])[None, None, :]
    y = (np.arange(n[1]) * sp[1])[None, :, None]
    z = (np.arange(n[2]) * sp[2])[:, None, None]
    return x, y, z


def _ball(x, y, z, centre, r):
    return (x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2 <= r * r


def _write(arr, spacing, path):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(img, str(path))


def make_case(root, cid, face=True, node_below=True):
    d = root / cid
    d.mkdir(parents=True)
    nodes = NODES + ((NODE_BELOW,) if node_below else ())
    x, y, z = _grid_mm(CT_N, CT_SP)
    body = (x - BODY_C[0]) ** 2 + (y - BODY_C[1]) ** 2 <= BODY_R ** 2
    ct = np.where(body, 40, -1000).astype(np.int16) + np.zeros(z.shape, np.int16)
    lab = np.zeros(ct.shape, np.uint8)
    p = _ball(x, y, z, *GTVP)
    ct[p] = 60
    lab[p] = 1
    if face:
        ct[p & (z > GTVP[0][2] + 6.0)] = -1000
    for centre, r in nodes:
        lab[_ball(x, y, z, centre, r)] = 2
    _write(ct, CT_SP, d / f"{cid}__CT.nii.gz")
    _write(lab, CT_SP, d / f"{cid}.nii.gz")
    x, y, z = _grid_mm(PET_N, PET_SP)
    body = ((x - BODY_C[0]) ** 2 + (y - BODY_C[1]) ** 2 <= BODY_R ** 2) & (z >= 0)
    pet = np.where(body, 1.0, 0.0).astype(np.float32)
    pet[body & (z >= 270.0)] = 6.0  # cerebro: el ancla del ROI
    pet[_ball(x, y, z, GTVP[0], GTVP[1] + 4.0)] = 12.0  # captacion algo mas ancha que la etiqueta, como el PET real
    for (centre, r), suv in zip(nodes, (6.0, 4.0, 5.0)):
        pet[_ball(x, y, z, centre, r + 2.0)] = suv
    _write(pet, PET_SP, d / f"{cid}__PT.nii.gz")
    return mio.CaseFiles(cid, cid.split("-")[0], d / f"{cid}__CT.nii.gz", d / f"{cid}__PT.nii.gz", d / f"{cid}.nii.gz")


def _prepared(files):
    data_cfg = load_config(None)["data"]
    spacing = tuple(float(s) for s in data_cfg["target_spacing_xyz"])
    ct, pet, lab = (mio.read_lps(p) for p in (files.ct, files.pet, files.label))
    roi_lo, roi_hi = mio.compute_roi(ct, pet, data_cfg["roi"], spacing)
    size = np.maximum(np.floor((roi_hi - roi_lo) / np.array(spacing)).astype(int) + 1, 1)
    arr = sitk.GetArrayFromImage(lab)
    nz = np.nonzero(arr)
    bbox = (tuple(int(a.min()) for a in nz), tuple(int(a.max()) for a in nz))
    return data_cfg, spacing, lab, arr, bbox, roi_lo, size


def test_audit_case_mide_lo_que_se_sabe(tmp_path):
    files = make_case(tmp_path, "SYN-001")
    qc = tmp_path / "qc"
    qc.mkdir()
    row = ad.audit_case(files, load_config(None)["data"], "train", str(qc))
    assert row["ancla"] is True
    assert row["etiq_en_rejilla_ct"] is True and row["etiq_fov_menor"] is False
    assert row["cobertura_gtvp"] == 1.0
    assert (row["gtvn_componentes"], row["gtvn_componentes_fuera"], row["gtvn_componentes_cortados"]) == (3, 1, 0)
    assert 0.4 < row["cobertura_gtvn"] < ad.COV_MIN
    vec, frac, suave = (row[f"iv_{v}"] for v in ad.VERSIONS)
    assert 0.85 < vec["dice_gtvp"] < 1.0
    assert suave["dice_gtvp"] >= vec["dice_gtvp"] - 0.01
    for m in (vec, frac, suave):
        assert m["gtvn_fn"] >= 1  # el ganglio de debajo del ROI se pierde en las tres versiones
    assert row["gtvp_hu_p50"] == pytest.approx(60, abs=1)
    assert row["etiq_frac_hu_bajo_900"] > ad.FACE_FRAC
    assert 11.0 <= row["gtvp_suv_p50"] <= 12.01
    assert set(row["hist_suv"]) == {"gtvp", "gtvn"}
    assert any("bajo -900 HU" in m for m in row["marcas"]) and any("cobertura GTVN" in m for m in row["marcas"])
    assert (qc / "SYN-001.png").exists()
    json.dumps(row)


def test_cobertura_coincide_con_label_coverage_de_mamba(tmp_path):
    files = make_case(tmp_path, "SYN-001")
    data_cfg = load_config(None)["data"]
    row = ad.audit_case(files, data_cfg, "train", None)
    case = mio.load_case(files, data_cfg)
    total = row["gtvp_ml"] + row["gtvn_ml"]
    ours = (row["cobertura_gtvp"] * row["gtvp_ml"] + row["cobertura_gtvn"] * row["gtvn_ml"]) / total
    assert ours == pytest.approx(mio.label_coverage(case), abs=1e-3)


def test_el_recorte_no_cambia_las_metricas(tmp_path):
    files = make_case(tmp_path, "SYN-002", face=False)
    _, spacing, lab, arr, bbox, roi_lo, size = _prepared(files)
    small = ad.round_trips("SYN-002", lab, arr, bbox, roi_lo, size, spacing)
    full = ad.round_trips("SYN-002", lab, arr, bbox, roi_lo, size, spacing, full=True)
    assert small == full


def test_la_vuelta_con_vecino_es_la_de_restore_to_original(tmp_path):
    """La version "vecino" reproduce load_case + restore_to_original de mamba con la etiqueta como prediccion."""
    files = make_case(tmp_path, "SYN-003", face=False)
    data_cfg = load_config(None)["data"]
    _, spacing, lab, arr, bbox, roi_lo, size = _prepared(files)
    case = mio.load_case(files, data_cfg)
    pred_o, gt_o, vox = mio.restore_to_original(case, case.label)
    from mambalite3d.eval.metrics import evaluate_case
    from dataclasses import asdict

    ref = asdict(evaluate_case("SYN-003", pred_o, gt_o, vox))
    ours = ad.round_trips("SYN-003", lab, arr, bbox, roi_lo, size, spacing)["iv_vecino"]
    assert ours == ref


def test_fracciones_suman_uno(tmp_path):
    files = make_case(tmp_path, "SYN-004", face=False, node_below=False)
    _, spacing, lab, arr, bbox, roi_lo, size = _prepared(files)
    fr = ad.fractions(lab, roi_lo, spacing, size)
    assert fr.shape == (3, size[2], size[1], size[0])
    assert np.allclose(fr.sum(axis=0), 1.0)
    assert fr[1].max() == 1.0 and fr[2].max() > 0


def test_mayoria_y_empates():
    fr = np.zeros((3, 1, 1, 3), np.float32)
    fr[:, 0, 0, 0] = [2 / 3, 1 / 3, 0]
    fr[:, 0, 0, 1] = [1 / 3, 2 / 3, 0]
    fr[:, 0, 0, 2] = [1 / 3, 1 / 3, 1 / 3]  # empate a tres: manda el vecino
    nn = np.array([[[1, 0, 2]]], np.uint8)
    assert ad.majority(fr, nn).tolist() == [[[0, 1, 2]]]


def test_main_escribe_y_reanuda(tmp_path):
    root = tmp_path / "datos"
    make_case(root, "SYN-001")
    make_case(root, "SYN-002", face=False, node_below=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"seed: 1\ndata:\n  root: {json.dumps(root.as_posix())}\n", encoding="utf-8")
    out = tmp_path / "out"
    assert ad.main(["--config", str(cfg), "--out", str(out), "--workers", "1"]) == 0
    s = json.loads((out / "resumen.json").read_text(encoding="utf-8"))
    assert s["n_casos"] == 2 and s["n_con_error"] == 0
    assert set(s["puertas"]) == {"G0a_roi", "G0b_etiquetas", "G0c_restauracion", "G0d_suv", "G0e_rostro"}
    assert s["puertas"]["G0a_roi"]["se_cumple"] is True
    assert [c["case_id"] for c in s["puertas"]["G0e_rostro"]["casos"]] == ["SYN-001"]
    assert s["puertas"]["G0d_suv"]["casos_usados"] == "train"
    with open(out / "manifest.csv", encoding="utf-8-sig") as f:
        assert [r["case_id"] for r in csv.DictReader(f)] == ["SYN-001", "SYN-002"]
    assert (out / "qc" / "SYN-001.png").exists()
    assert ad.main(["--config", str(cfg), "--out", str(out), "--workers", "1"]) == 0  # relanzar no repite casos
    assert len((out / "casos.jsonl").read_text(encoding="utf-8").splitlines()) == 2
