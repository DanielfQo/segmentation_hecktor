"""Verifica, caso por caso y en TODO el dataset, el preprocesamiento que usa de verdad el entrenamiento. Solo CPU.

    python -m bifusion3d.tools.verify_preproc --config configs/siam_full.yaml --out runs/verificacion
    python -m bifusion3d.tools.verify_preproc --config configs/siam_full.yaml --out runs/verificacion_humo --cases CHUM-010 MDA-264

Para cada caso con etiqueta (los 680: train, val y test) llama a `mambalite3d.data.io.load_case`, la misma funcion con la
que se cargan los casos para entrenar, validar, evaluar e inferir, y comprueba:

  roi_formula      el ROI de load_case coincide con la formula, recalculada aparte paso a paso desde su definicion:
                   cuerpo = (HU > body_hu) | (SUV > body_suv) en una rejilla de 4 mm, su componente principal; tope = el
                   corte mas alto con SUV > suv_anchor dentro del cuerpo (el cerebro); base = tope - down_mm (sin bajar
                   del cuerpo); x/y = el cuerpo en esos cortes; margenes, tamano minimo y recorte al campo del CT
  roi_contiene     el ROI contiene el ancla, todo el cuerpo de esos cortes y queda dentro del CT
  roi_ancla        se encontro el ancla del cerebro (si no, la formula cae a todo el cuerpo: aviso)
  roi_cobertura    fraccion de cada clase de la etiqueta original que cae dentro del ROI (aviso si < 0,999)
  imagen           forma (2, Z, Y, X) igual a la del ROI, float16, sin NaN/Inf
  ct_rango         canal CT dentro de [0, 1] (ventana [-200, 400])
  pet_zscore       canal PET con media ~0 y desviacion ~1 en el ROI (o en [0, 1] si pet_norm no es zscore)
  pet_suv          el PET parece SUV (percentil 99,9 en el ROI >= 2; aviso si no, como CHUM-010)
  etiqueta         valores solo 0, 1 y 2; volumen y numero de componentes por clase frente a la etiqueta original
  fg_coords        las coordenadas de muestreo apuntan a su clase y son min(n, 20000)
  vuelta           la etiqueta del ROI devuelta a su rejilla con restore_to_original (el camino de la evaluacion):
                   DSC por clase y ganglios TP/FP/FN, con el emparejamiento hungaro y con la regla de grupos
  determinismo     cada --det-every casos, una segunda carga por iter_cases (el camino de la evaluacion) da lo mismo

Salida en --out: casos.jsonl (si se corta, al relanzar salta los hechos), verificacion.csv, resumen.json y un
informe en pantalla con cada comprobacion como ok / aviso (propiedad de los datos) / fallo (el preprocesamiento no
hace lo que dice). No escribe nada en modelos/mamba.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..guard import refuse_inside_mamba  # importa bifusion3d, que anade modelos/mamba a sys.path

import SimpleITK as sitk  # noqa: E402
from scipy import ndimage  # noqa: E402

from mambalite3d.config import load_config, resolve  # noqa: E402
from mambalite3d.data import io as mio  # noqa: E402
from mambalite3d.data.sampler import iter_cases  # noqa: E402
from mambalite3d.data.splits import select_subset, split_cases  # noqa: E402
from mambalite3d.eval.metrics import STRUCT_26, evaluate_case  # noqa: E402

from ..eval_groups import match_groups  # noqa: E402
from .audit_data import _r, coverage  # noqa: E402

ROI_TOL_MM = 1e-3
COV_MIN = 0.999
PET_TOL = 0.02
VOL_RATIO = (0.8, 1.25)  # aviso fuera de este rango, solo para clases de >= 0,5 ml
VOL_MIN_ML = 0.5
TRIP_DSC_MIN = 0.85  # aviso por debajo, solo para clases de >= 1 ml
SUV_P999_MIN = 2.0
MAX_FG = 20000  # max_fg_coords de load_case

FALLOS = ("roi_formula", "roi_contiene", "imagen", "ct_rango", "pet_zscore", "etiqueta_valores", "fg_coords", "determinismo")
AVISOS = ("roi_ancla", "roi_cobertura", "pet_suv", "etiqueta_volumen", "etiqueta_componentes", "vuelta")


# ---------------------------------------------------------------------------------------------------------------
# la formula del ROI, recalculada aparte
# ---------------------------------------------------------------------------------------------------------------

def roi_formula(ct: sitk.Image, pet: sitk.Image, roi_cfg: dict, spacing_xyz) -> dict:
    """El ROI que define la formula, paso a paso, con que regla actuo en cada paso. Coordenadas LPS en mm (x, y, z)."""
    C = mio.COARSE_MM
    lo_ct, hi_ct = mio._physical_bounds(ct)
    size = np.maximum(np.ceil((hi_ct - lo_ct) / C).astype(int) + 1, 1)
    ref = sitk.Image([int(s) for s in size], sitk.sitkUInt8)
    ref.SetOrigin([float(v) for v in lo_ct])
    ref.SetSpacing((C, C, C))
    ref.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    ct_c = sitk.GetArrayFromImage(sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear, -1024.0, sitk.sitkFloat32))
    pet_c = sitk.GetArrayFromImage(sitk.Resample(pet, ref, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32))
    body = (ct_c > roi_cfg["body_hu"]) | (pet_c > roi_cfg["body_suv"])
    if roi_cfg.get("main_body", True):
        body = mio.main_body(body, roi_cfg.get("open_iters", 2))  # componente principal: quita la camilla y el ruido
    z_mm = lo_ct[2] + np.arange(body.shape[0]) * C
    out = {"ancla": False, "ancla_z_mm": None, "base_limitada_por_cuerpo": False, "xy_de_todo_el_cuerpo": False,
           "crece_por_minimo": [False, False, False], "recortado_por_ct": [], "body": body, "z_mm": z_mm,
           "lo_ct": lo_ct, "hi_ct": hi_ct}
    body_z = body.any(axis=(1, 2))
    if not body_z.any():
        out.update(lo=lo_ct, hi=hi_ct, sin_cuerpo=True, z_rango=(lo_ct[2], hi_ct[2]))
        return out
    z_lo, z_hi = z_mm[body_z].min(), z_mm[body_z].max()
    active = ((pet_c > roi_cfg["suv_anchor"]) & body).any(axis=(1, 2))
    if active.any():
        anchor = float(z_mm[active].max())
        out["ancla"], out["ancla_z_mm"] = True, anchor
        z_hi = min(anchor, z_hi)
        base = z_hi - roi_cfg["down_mm"]
        out["base_limitada_por_cuerpo"] = bool(base < z_lo)
        z_lo = max(base, z_lo)
    in_z = (z_mm >= z_lo) & (z_mm <= z_hi)
    body_xy = body[in_z].any(axis=0)
    if not body_xy.any():
        body_xy = body.any(axis=0)
        out["xy_de_todo_el_cuerpo"] = True
    yy, xx = np.nonzero(body_xy)
    lo = np.array([lo_ct[0] + xx.min() * C, lo_ct[1] + yy.min() * C, z_lo], dtype=float)
    hi = np.array([lo_ct[0] + xx.max() * C, lo_ct[1] + yy.max() * C, z_hi], dtype=float)
    margin = np.array(roi_cfg["margin_vox"], dtype=float) * np.array(spacing_xyz, dtype=float)
    lo -= margin + C / 2
    hi += margin + C / 2
    grow = np.maximum(np.array(roi_cfg["min_size_vox"], dtype=float) * np.array(spacing_xyz, dtype=float) - (hi - lo), 0) / 2
    out["crece_por_minimo"] = [bool(g > 0) for g in grow]
    lo -= grow
    hi += grow
    for i, ax in enumerate("xyz"):
        if lo[i] < lo_ct[i]:
            out["recortado_por_ct"].append(f"{ax}_min")
        if hi[i] > hi_ct[i]:
            out["recortado_por_ct"].append(f"{ax}_max")
    out.update(lo=np.maximum(lo, lo_ct), hi=np.minimum(hi, hi_ct), z_rango=(float(z_lo), float(z_hi)), in_z=in_z, body_xy=body_xy)
    return out


def roi_contains(f: dict, roi_lo, roi_hi) -> List[str]:
    """Postcondiciones de la formula sobre el ROI que uso load_case. Devuelve los problemas encontrados."""
    problems = []
    lo, hi = np.asarray(roi_lo, float), np.asarray(roi_hi, float)
    if np.any(lo < f["lo_ct"] - ROI_TOL_MM) or np.any(hi > f["hi_ct"] + ROI_TOL_MM):
        problems.append("ROI fuera del campo del CT")
    if f.get("sin_cuerpo"):
        return problems
    if f["ancla"] and not f["recortado_por_ct"].count("z_max") and hi[2] < f["ancla_z_mm"] - ROI_TOL_MM:
        problems.append("el ROI no llega al ancla")
    C = mio.COARSE_MM
    yy, xx = np.nonzero(f["body_xy"])
    bx = (f["lo_ct"][0] + xx * C, f["lo_ct"][1] + yy * C)
    inside = (bx[0] >= lo[0] - C / 2 - ROI_TOL_MM) & (bx[0] <= hi[0] + C / 2 + ROI_TOL_MM) & \
             (bx[1] >= lo[1] - C / 2 - ROI_TOL_MM) & (bx[1] <= hi[1] + C / 2 + ROI_TOL_MM)
    clipped_xy = any(s in f["recortado_por_ct"] for s in ("x_min", "x_max", "y_min", "y_max"))
    if not inside.all() and not clipped_xy:
        problems.append(f"{int((~inside).sum())} voxeles de cuerpo fuera del ROI en x/y")
    return problems


# ---------------------------------------------------------------------------------------------------------------
# un caso
# ---------------------------------------------------------------------------------------------------------------

def _components(arr: np.ndarray, cls: int) -> int:
    m = arr == cls
    if not m.any():
        return 0
    nz = np.nonzero(m)
    sl = tuple(slice(max(int(a.min()) - 1, 0), int(a.max()) + 2) for a in nz)  # solo la caja de la clase
    return int(ndimage.label(m[sl], structure=STRUCT_26)[1])


def verify_case(files: mio.CaseFiles, data_cfg: dict, split_name: str = "", det: bool = False) -> dict:
    t0 = time.time()
    spacing = tuple(float(s) for s in data_cfg["target_spacing_xyz"])
    row: Dict = {"case_id": files.case_id, "centro": files.center, "split": split_name, "fallos": [], "avisos": []}
    ct, pet, lab = mio.read_lps(files.ct), mio.read_lps(files.pet), mio.read_lps(files.label)

    # --- el preprocesamiento real ---
    case = mio.load_case(files, data_cfg, with_label=True, seed=0)
    roi_lo = np.asarray(case.roi_origin_xyz, dtype=float)
    shape_zyx = tuple(int(s) for s in case.image.shape[1:])
    roi_hi = roi_lo + (np.array(shape_zyx[::-1], dtype=float) - 1) * np.array(spacing)
    row["roi_lo_mm"] = [_r(v, 2) for v in roi_lo]
    row["roi_zyx"] = list(shape_zyx)

    # --- R: la formula ---
    f = roi_formula(ct, pet, data_cfg["roi"], spacing)
    size_f = np.maximum(np.floor((f["hi"] - f["lo"]) / np.array(spacing)).astype(int) + 1, 1)
    d_origin = float(np.max(np.abs(f["lo"] - roi_lo)))
    row["roi_formula_dif_mm"] = _r(d_origin, 6)
    row["roi_formula_forma"] = [int(v) for v in size_f[::-1]]
    if d_origin > ROI_TOL_MM or tuple(size_f[::-1]) != shape_zyx:
        row["fallos"].append("roi_formula")
    problems = roi_contains(f, roi_lo, roi_hi)
    row["roi_contiene"] = problems
    if problems:
        row["fallos"].append("roi_contiene")
    row["ancla"] = bool(f["ancla"])
    row["ancla_bajo_tope_roi_mm"] = _r(roi_hi[2] - f["ancla_z_mm"], 1) if f["ancla"] else None
    row["base_limitada_por_cuerpo"] = bool(f["base_limitada_por_cuerpo"])
    row["crece_por_minimo"] = f["crece_por_minimo"]
    row["recortado_por_ct"] = f["recortado_por_ct"]
    row["roi_mm"] = [_r(v, 1) for v in (roi_hi - roi_lo + np.array(spacing))]
    if not f["ancla"]:
        row["avisos"].append("roi_ancla")

    # --- I: la imagen ---
    img = case.image
    ok_img = img.dtype == np.float16 and img.ndim == 4 and img.shape[0] == 2 and bool(np.isfinite(img).all())
    if not ok_img:
        row["fallos"].append("imagen")
    ct_n, pet_n = img[0].astype(np.float32), img[1].astype(np.float32)
    row["ct_min_max"] = [_r(ct_n.min(), 4), _r(ct_n.max(), 4)]
    if ct_n.min() < -1e-3 or ct_n.max() > 1 + 1e-3:
        row["fallos"].append("ct_rango")
    row["pet_media_desv"] = [_r(pet_n.mean(), 4), _r(pet_n.std(), 4)]
    if data_cfg.get("pet_norm", "zscore") == "zscore":
        bad_pet = abs(float(pet_n.mean())) > PET_TOL or abs(float(pet_n.std()) - 1) > PET_TOL
    else:
        bad_pet = pet_n.min() < -1e-3 or pet_n.max() > 1 + 1e-3
    if bad_pet:
        row["fallos"].append("pet_zscore")
    if tuple(round(float(v), 4) for v in case.pad_values) != (round(float(ct_n.min()), 4), round(float(pet_n.min()), 4)):
        row["fallos"].append("imagen")
    ref = mio._grid(roi_lo, spacing, shape_zyx[::-1])
    pet_raw = sitk.GetArrayFromImage(sitk.Resample(pet, ref, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32))
    row["pet_p999_suv"] = _r(np.percentile(pet_raw, 99.9), 2)
    if row["pet_p999_suv"] is not None and row["pet_p999_suv"] < SUV_P999_MIN:
        row["avisos"].append("pet_suv")

    # --- L: la etiqueta ---
    lab_arr = sitk.GetArrayFromImage(lab)
    roi_lab = case.label
    vals = sorted(int(v) for v in np.unique(roi_lab))
    row["etiqueta_valores"] = vals
    if any(v not in (0, 1, 2) for v in vals):
        row["fallos"].append("etiqueta_valores")
    nz = np.nonzero(lab_arr)
    bbox = None if len(nz[0]) == 0 else (tuple(int(a.min()) for a in nz), tuple(int(a.max()) for a in nz))
    cov = coverage(lab, lab_arr, bbox, roi_lo, shape_zyx[::-1], spacing)
    row.update({k: cov[k] for k in ("gtvp_ml", "gtvn_ml", "cobertura_gtvp", "cobertura_gtvn", "gtvn_componentes_fuera")})
    if any(cov[k] is not None and cov[k] < COV_MIN for k in ("cobertura_gtvp", "cobertura_gtvn")):
        row["avisos"].append("roi_cobertura")
    vox_o = float(np.prod(lab.GetSpacing())) / 1000.0
    vox_r = float(np.prod(spacing)) / 1000.0
    vol_flag = comp_flag = False
    for c, name in ((1, "gtvp"), (2, "gtvn")):
        v_o = float((lab_arr == c).sum()) * vox_o * float(cov[f"cobertura_{name}"] or 0.0)  # lo que cae dentro del ROI
        v_r = float((roi_lab == c).sum()) * vox_r
        row[f"{name}_ml_roi"] = _r(v_r, 3)
        if v_o >= VOL_MIN_ML and not VOL_RATIO[0] <= v_r / v_o <= VOL_RATIO[1]:
            vol_flag = True
        n_o, n_r = _components(lab_arr, c), _components(roi_lab, c)
        row[f"{name}_componentes"] = [n_o, n_r]
        lost = n_o - (cov["gtvn_componentes_fuera"] if c == 2 else 0) - n_r
        if lost != 0:
            comp_flag = True
    if vol_flag:
        row["avisos"].append("etiqueta_volumen")
    if comp_flag:
        row["avisos"].append("etiqueta_componentes")

    # --- fg_coords ---
    fg_ok = True
    for c in (1, 2):
        n = int((roi_lab == c).sum())
        coords = case.fg_coords.get(c)
        if n == 0:
            fg_ok &= coords is None
        else:
            fg_ok &= coords is not None and len(coords) == min(n, MAX_FG) and bool((roi_lab[tuple(coords.T)] == c).all())
    if not fg_ok:
        row["fallos"].append("fg_coords")

    # --- vuelta por el camino de la evaluacion ---
    pred_o, gt_o, vox = mio.restore_to_original(case, roi_lab)
    r = evaluate_case(files.case_id, pred_o, gt_o, vox)
    tpg, fpg, fng = match_groups(pred_o == 2, gt_o == 2)
    dsc_n = 2 * r.gtvn_inter_mm3 / r.gtvn_sum_mm3 if r.gtvn_sum_mm3 > 0 else None
    row["vuelta"] = {"dsc_gtvp": _r(r.dice_gtvp), "dsc_gtvn": _r(dsc_n), "hungaro_tp_fp_fn": [r.gtvn_tp, r.gtvn_fp, r.gtvn_fn],
                     "grupos_tp_fp_fn": [tpg, fpg, fng], "gtvn_inter_mm3": _r(r.gtvn_inter_mm3, 1),
                     "gtvn_suma_mm3": _r(r.gtvn_sum_mm3, 1)}
    low = (row["gtvp_ml"] >= 1 and r.dice_gtvp < TRIP_DSC_MIN) or (row["gtvn_ml"] >= 1 and dsc_n is not None and dsc_n < TRIP_DSC_MIN)
    if low:
        row["avisos"].append("vuelta")

    # --- determinismo, por el camino de la evaluacion (iter_cases) ---
    if det:
        again = list(iter_cases([files], data_cfg, threads=1))[0]
        same = (np.array_equal(again.image, case.image) and np.array_equal(again.label, case.label)
                and again.roi_origin_xyz == case.roi_origin_xyz
                and all(np.array_equal(again.fg_coords.get(c), case.fg_coords.get(c)) for c in (1, 2)))
        row["determinismo"] = bool(same)
        if not same:
            row["fallos"].append("determinismo")
    row["fallos"] = sorted(set(row["fallos"]))
    row["avisos"] = sorted(set(row["avisos"]))
    row["segundos"] = _r(time.time() - t0, 1)
    return row


def _worker(files, data_cfg, split_name, det, threads):
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(threads))
    except AttributeError:
        try:
            sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(int(threads))
        except AttributeError:
            pass
    try:
        return verify_case(files, data_cfg, split_name, det)
    except Exception:  # noqa: BLE001
        return {"case_id": files.case_id, "centro": files.center, "split": split_name, "error": traceback.format_exc()}


# ---------------------------------------------------------------------------------------------------------------
# resumen
# ---------------------------------------------------------------------------------------------------------------

def summarize(rows: Sequence[dict], meta: dict) -> dict:
    ok = [r for r in rows if "error" not in r]
    s = {**meta, "n_casos": len(rows), "n_con_error": len(rows) - len(ok),
         "errores": [{"case_id": r["case_id"], "error": r["error"].strip().splitlines()[-1]} for r in rows if "error" in r],
         "por_split": dict(Counter(r["split"] for r in rows)), "comprobaciones": {}}
    for name in FALLOS + AVISOS:
        kind = "fallo" if name in FALLOS else "aviso"
        key = "fallos" if kind == "fallo" else "avisos"
        hit = [r for r in ok if name in r[key]]
        checked = [r for r in ok if name != "determinismo" or "determinismo" in r]
        s["comprobaciones"][name] = {"tipo": kind, "casos_comprobados": len(checked), "casos_marcados": len(hit),
                                     "estado": "ok" if not hit else kind,
                                     "casos": [r["case_id"] for r in hit][:50],
                                     "por_split": dict(Counter(r["split"] for r in hit))}
    s["roi"] = {
        "ancla_encontrada": sum(r["ancla"] for r in ok),
        "base_limitada_por_cuerpo": sum(r["base_limitada_por_cuerpo"] for r in ok),
        "crece_por_minimo_xyz": [sum(r["crece_por_minimo"][i] for r in ok) for i in range(3)],
        "recortado_por_ct": dict(Counter(x for r in ok for x in r["recortado_por_ct"])),
        "dif_maxima_con_la_formula_mm": max((r["roi_formula_dif_mm"] or 0.0) for r in ok) if ok else None,
        "tamano_mm_p5_p50_p95": [[_r(np.percentile([r["roi_mm"][i] for r in ok], q), 1) for q in (5, 50, 95)] for i in range(3)] if ok else None,
    }
    if ok:
        def agg(rs, key):
            tp = sum(r["vuelta"][key][0] for r in rs)
            fp = sum(r["vuelta"][key][1] for r in rs)
            fn = sum(r["vuelta"][key][2] for r in rs)
            return {"tp_fp_fn": [tp, fp, fn], "f1": _r(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None}
        inter = sum(r["vuelta"]["gtvn_inter_mm3"] or 0 for r in ok)
        tot = sum(r["vuelta"]["gtvn_suma_mm3"] or 0 for r in ok)
        s["vuelta"] = {"dsc_gtvp_medio": _r(np.mean([r["vuelta"]["dsc_gtvp"] for r in ok])),
                       "gtvn_dscagg": _r(2 * inter / tot) if tot else None,
                       "ganglios_hungaro": agg(ok, "hungaro_tp_fp_fn"), "ganglios_grupos": agg(ok, "grupos_tp_fp_fn")}
    s["veredicto"] = "fallo" if any(v["estado"] == "fallo" for v in s["comprobaciones"].values()) or s["n_con_error"] else "ok"
    return s


def _flat(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}_{k2}"] = "/".join(str(x) for x in v2) if isinstance(v2, (list, tuple)) else v2
        elif isinstance(v, (list, tuple)):
            out[k] = "; ".join(str(x) for x in v)
        else:
            out[k] = v
    return out


def write_outputs(rows: Sequence[dict], out: Path, meta: dict) -> dict:
    rows = sorted(rows, key=lambda r: r["case_id"])
    flat = [_flat(r) for r in rows]
    keys = []
    for fr in flat:
        keys += [k for k in fr if k not in keys]
    with open(out / "verificacion.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    s = summarize(rows, meta)
    (out / "resumen.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    return s


def print_summary(s: dict) -> None:
    print("=" * 78)
    print(f"verificacion del preprocesamiento: {s['n_casos']} casos {s['por_split']}, errores {s['n_con_error']}")
    for name, c in s["comprobaciones"].items():
        mark = {"ok": "ok   ", "aviso": "AVISO", "fallo": "FALLO"}[c["estado"]]
        extra = f"  {c['casos'][:8]}" if c["casos_marcados"] else ""
        print(f"  [{mark}] {name:22s} {c['casos_marcados']:4d} de {c['casos_comprobados']}{extra}")
    if "roi" in s:
        print(f"  ROI: ancla en {s['roi']['ancla_encontrada']} casos; diferencia maxima con la formula "
              f"{s['roi']['dif_maxima_con_la_formula_mm']} mm; recortes por el CT {s['roi']['recortado_por_ct']}")
    if "vuelta" in s:
        v = s["vuelta"]
        print(f"  vuelta: DSC GTVp {v['dsc_gtvp_medio']}, DSCagg GTVn {v['gtvn_dscagg']}, F1 hungaro "
              f"{v['ganglios_hungaro']['f1']}, F1 por grupos {v['ganglios_grupos']['f1']}")
    print(f"veredicto: {s['veredicto'].upper()}")
    print("=" * 78)


def _read_jsonl(path: Path) -> Dict[str, dict]:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "error" not in row:
                done[row["case_id"]] = row
    return done


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    refuse_inside_mamba(argv, flags=("--out",))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs/verificacion")
    ap.add_argument("--workers", type=int, default=2, help="procesos en paralelo (2 por defecto, por la temperatura)")
    ap.add_argument("--threads", type=int, default=2, help="hilos de SimpleITK por proceso")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--det-every", type=int, default=20, help="cada cuantos casos se repite la carga (0 = nunca)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    root = resolve(data_cfg["root"])
    all_cases = [c for c in mio.find_cases(root) if c.label is not None]
    subset = select_subset(all_cases, data_cfg["cases_per_center"], data_cfg["centers"], cfg["seed"])
    tr, va, te = split_cases(subset, data_cfg["split"], cfg["seed"])
    split_of = {**{c.case_id: "train" for c in tr}, **{c.case_id: "val" for c in va}, **{c.case_id: "test" for c in te}}
    todo = all_cases if args.cases is None else [c for c in all_cases if c.case_id in set(args.cases)]
    if args.max_cases:
        todo = todo[: args.max_cases]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "casos.jsonl"
    done = _read_jsonl(jsonl)
    pending = [c for c in todo if c.case_id not in done]
    order = {c.case_id: i for i, c in enumerate(all_cases)}
    print(f"{len(all_cases)} casos con etiqueta en {root}; por verificar {len(pending)} (ya hechos {len(done)})", flush=True)
    t0 = time.time()
    mod = importlib.import_module("bifusion3d.tools.verify_preproc")  # por nombre: los procesos hijos lo importan igual
    with open(jsonl, "a", encoding="utf-8") as fj:
        def keep(row):
            fj.write(json.dumps(row, ensure_ascii=False) + "\n")
            fj.flush()
            if "error" not in row:
                done[row["case_id"]] = row
            flags = (" FALLO " + ",".join(row["fallos"])) if row.get("fallos") else ""
            warn = (" aviso " + ",".join(row["avisos"])) if row.get("avisos") else ""
            err = " ERROR" if "error" in row else ""
            print(f"  {row['case_id']} ({row['split']}) {row.get('segundos', '-')} s{flags}{warn}{err}", flush=True)

        det = lambda c: bool(args.det_every) and order[c.case_id] % args.det_every == 0  # noqa: E731
        if args.workers <= 1:
            for c in pending:
                keep(mod._worker(c, data_cfg, split_of.get(c.case_id, "fuera"), det(c), args.threads))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(mod._worker, c, data_cfg, split_of.get(c.case_id, "fuera"), det(c), args.threads) for c in pending]
                for fut in as_completed(futs):
                    keep(fut.result())
    rows = [done[c.case_id] for c in todo if c.case_id in done]
    errs = {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if "error" in e and e["case_id"] not in done:
            errs[e["case_id"]] = e  # el ultimo intento de cada caso
    errs = list(errs.values())
    meta = {"config": args.config, "raiz_datos": str(root), "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
            "segundos_totales": round(time.time() - t0, 1), "target_spacing_xyz": data_cfg["target_spacing_xyz"],
            "roi": data_cfg["roi"], "pet_norm": data_cfg.get("pet_norm")}
    s = write_outputs(rows + errs, out, meta)
    print_summary(s)
    return 0 if s["veredicto"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
