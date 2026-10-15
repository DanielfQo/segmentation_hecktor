"""Experimento sin entrenar, en los 680 casos: escala del PET, profundidad del ROI y remuestreo de la etiqueta. Solo CPU.

    python -m bifusion3d.tools.experimento_preproc --config configs/siam_full.yaml --out runs/experimento_preproc
    python -m bifusion3d.tools.experimento_preproc --config configs/siam_full.yaml --out runs/exp_humo --cases CHUM-010 MDA-264

Mide, caso por caso, cada arreglo de los avisos de la verificacion frente al preprocesamiento actual:

  pet        mediana del PET en el tejido blando (CT entre 0 y 100 HU) y percentil 99 craneal (bifusion3d.data.pet_scale);
             factor de la regla de escala. Si el factor no es 1, el ROI se recalcula con el PET corregido para ver si
             aparece el ancla del cerebro.
  roi        fraccion de GTVp y GTVn de la etiqueta original dentro del ROI con down_mm = 270, 300 y 330 mm, y su
             tamano (voxeles y altura).
  etiquetas  en la rejilla de trabajo (ROI de 270 mm, 1 x 1 x 3 mm): componentes y volumen de cada clase en la
             etiqueta original, en la actual (voxel mas cercano) y en la remuestreada ganglio por ganglio; volumen de
             la etiqueta suave.
  vuelta     la etiqueta de trabajo devuelta a la rejilla original, como si el modelo acertara todo. Variantes:
             cercano+cercano (la actual), cercano+lineal, suave+lineal, nodos+cercano y nodos+lineal. Metricas
             oficiales: DSC de GTVp, DSCagg de GTVn y F1 de ganglios, con el emparejamiento hungaro y por grupos.

Salida en --out: casos.jsonl (si se corta, al relanzar sigue), casos.csv y resumen.json. No escribe en modelos/mamba.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..guard import refuse_inside_mamba  # importa bifusion3d, que anade modelos/mamba a sys.path

import SimpleITK as sitk  # noqa: E402

from mambalite3d.config import load_config, resolve  # noqa: E402
from mambalite3d.data import io as mio  # noqa: E402
from mambalite3d.data.splits import select_subset, split_cases  # noqa: E402
from mambalite3d.eval.metrics import CaseResult, aggregate, evaluate_case  # noqa: E402

from ..data import labels, pet_scale, restore  # noqa: E402
from ..eval_groups import match_groups  # noqa: E402
from .audit_data import _r, coverage  # noqa: E402
from .verify_preproc import roi_formula  # noqa: E402

DOWNS = (270, 300, 330)
VARIANTES = ("cercano+cercano", "cercano+lineal", "suave+lineal", "nodos+cercano", "nodos+lineal")
REFERENCIA = "cercano+cercano"
MARGEN_MM = 10.0  # alrededor de la etiqueta original, para la vuelta
COV_MIN = 0.999


def _bbox(arr: np.ndarray):
    nz = np.nonzero(arr)
    if len(nz[0]) == 0:
        return None
    return tuple(int(a.min()) for a in nz), tuple(int(a.max()) for a in nz)


def _phys_box(img: sitk.Image, bbox, margin_mm: float):
    """Caja fisica (x, y, z) de una caja de indices (z, y, x) de `img`, con un margen en mm."""
    (z0, y0, x0), (z1, y1, x1) = bbox
    pts = np.array([img.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
                    for x in (x0 - 0.5, x1 + 0.5) for y in (y0 - 0.5, y1 + 0.5) for z in (z0 - 0.5, z1 + 0.5)])
    return pts.min(axis=0) - margin_mm, pts.max(axis=0) + margin_mm


def _comp_ml(arr: np.ndarray, voxel_ml: float) -> dict:
    return {"gtvp_comp": labels.count_components(arr, 1), "gtvn_comp": labels.count_components(arr, 2),
            "gtvp_ml": _r(float((arr == 1).sum()) * voxel_ml, 4), "gtvn_ml": _r(float((arr == 2).sum()) * voxel_ml, 4)}


def _metrics(case_id: str, pred: np.ndarray, gt: np.ndarray, voxel_mm3: float) -> dict:
    r = evaluate_case(case_id, pred, gt, voxel_mm3)
    tp, fp, fn = match_groups(pred == 2, gt == 2)
    return {"dice_gtvp": _r(r.dice_gtvp, 5), "gtvp_in_ref": r.gtvp_in_ref, "gtvp_inter_mm3": _r(r.gtvp_inter_mm3, 2),
            "gtvp_sum_mm3": _r(r.gtvp_sum_mm3, 2), "gtvn_inter_mm3": _r(r.gtvn_inter_mm3, 2),
            "gtvn_sum_mm3": _r(r.gtvn_sum_mm3, 2), "dice_gtvn_case": _r(r.dice_gtvn_case, 5),
            "tp": r.gtvn_tp, "fp": r.gtvn_fp, "fn": r.gtvn_fn, "tp_grupos": tp, "fp_grupos": fp, "fn_grupos": fn}


def experiment_case(files: mio.CaseFiles, data_cfg: dict, split_name: str = "", downs: Sequence[int] = DOWNS) -> dict:
    t0 = time.time()
    spacing = tuple(float(s) for s in data_cfg["target_spacing_xyz"])
    roi_cfg = data_cfg["roi"]
    ct, pet, lab = mio.read_lps(files.ct), mio.read_lps(files.pet), mio.read_lps(files.label)
    lab_arr = sitk.GetArrayFromImage(lab)
    bbox = _bbox(lab_arr)
    row: Dict = {"case_id": files.case_id, "centro": files.center, "split": split_name,
                 "dz_original_mm": _r(lab.GetSpacing()[2], 3)}

    # --- escala del PET -------------------------------------------------------------------------------------
    st = pet_scale.pet_stats(ct, pet, roi_cfg)
    factor = pet_scale.scale_factor(st["suv_tejido"])
    row["pet"] = {"suv_tejido": _r(st["suv_tejido"], 4), "suv_craneal_p99": _r(st["suv_craneal_p99"], 3),
                  "factor": _r(factor, 3)}
    if factor != 1.0:
        before = roi_formula(ct, pet, roi_cfg, spacing)
        after = roi_formula(ct, pet_scale.scaled(pet, factor), roi_cfg, spacing)
        st2 = pet_scale.pet_stats(ct, pet_scale.scaled(pet, factor), roi_cfg)
        row["pet"].update(ancla_antes=bool(before["ancla"]), ancla_despues=bool(after["ancla"]),
                          roi_z_antes_mm=_r(before["hi"][2] - before["lo"][2], 1),
                          roi_z_despues_mm=_r(after["hi"][2] - after["lo"][2], 1),
                          suv_craneal_p99_despues=_r(st2["suv_craneal_p99"], 3))

    # --- profundidad del ROI ----------------------------------------------------------------------------------
    work = None
    for d in downs:
        lo, hi = mio.compute_roi(ct, pet, dict(roi_cfg, down_mm=float(d)), spacing)
        size = np.maximum(np.floor((hi - lo) / np.array(spacing)).astype(int) + 1, 1)
        cov = coverage(lab, lab_arr, bbox, lo, size, spacing)
        row[f"roi_{d}"] = {"cobertura_gtvp": cov["cobertura_gtvp"], "cobertura_gtvn": cov["cobertura_gtvn"],
                           "gtvp_fuera_ml": cov["gtvp_fuera_ml"], "gtvn_fuera_ml": cov["gtvn_fuera_ml"],
                           "voxeles": int(np.prod(size)), "z_mm": _r(size[2] * spacing[2], 1)}
        if d == 270 or work is None:
            work = (lo, size)
    if bbox is None:
        row["segundos"] = _r(time.time() - t0, 1)
        return row

    # --- etiqueta en la rejilla de trabajo --------------------------------------------------------------------
    roi_lo, size = work
    ref = mio._grid(roi_lo, spacing, size)
    near = labels.resample_nearest(lab, ref)
    nodes, info = labels.resample_nodes(lab, ref)
    soft = labels.soft_fractions(lab, ref)
    work_ml = float(np.prod(spacing)) / 1000.0
    (z0, y0, x0), (z1, y1, x1) = bbox
    crop = lab_arr[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    orig_ml = float(np.prod(lab.GetSpacing())) / 1000.0
    row["etiquetas"] = {"original": _comp_ml(crop, orig_ml), "cercano": _comp_ml(near, work_ml),
                        "nodos": {**_comp_ml(nodes, work_ml), **info},
                        "suave": {"gtvp_ml": _r(float(soft[0].sum()) * work_ml, 4),
                                  "gtvn_ml": _r(float(soft[1].sum()) * work_ml, 4)}}

    # --- vuelta a la rejilla original (como si el modelo acertara todo) -------------------------------------
    box = _phys_box(lab, bbox, MARGEN_MM)
    lo_i, hi_i = restore.index_box(lab, *box, pad=0)
    sl_o = (slice(lo_i[2], hi_i[2]), slice(lo_i[1], hi_i[1]), slice(lo_i[0], hi_i[0]))
    rlo, rhi = restore.index_box(ref, *box, pad=2)  # ventana de la rejilla de trabajo que cubre la caja
    sl_w = (slice(rlo[2], rhi[2]), slice(rlo[1], rhi[1]), slice(rlo[0], rhi[0]))
    origin_w = ref.TransformIndexToPhysicalPoint(tuple(int(v) for v in rlo))
    gt = lab_arr[sl_o]
    vox = float(np.prod(lab.GetSpacing()))
    near_w, nodes_w, soft_w = near[sl_w], nodes[sl_w], soft[(slice(None),) + sl_w]
    preds = {
        "cercano+cercano": lambda: restore.restore_label_nearest(near_w, origin_w, spacing, lab, box),
        "cercano+lineal": lambda: restore.restore_probs_linear(restore.one_hot(near_w), origin_w, spacing, lab, box),
        "suave+lineal": lambda: restore.restore_probs_linear(labels.soft_to_probs(soft_w), origin_w, spacing, lab, box),
        "nodos+cercano": lambda: restore.restore_label_nearest(nodes_w, origin_w, spacing, lab, box),
        "nodos+lineal": lambda: restore.restore_probs_linear(restore.one_hot(nodes_w), origin_w, spacing, lab, box),
    }
    row["vuelta"] = {name: _metrics(files.case_id, fn()[sl_o], gt, vox) for name, fn in preds.items()}
    row["segundos"] = _r(time.time() - t0, 1)
    return row


def _worker(files, data_cfg, split_name, threads):
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(threads))
    except AttributeError:
        pass
    try:
        return experiment_case(files, data_cfg, split_name)
    except Exception:  # noqa: BLE001
        return {"case_id": files.case_id, "centro": files.center, "split": split_name, "error": traceback.format_exc()}


# ---------------------------------------------------------------------------------------------------------------
# resumen
# ---------------------------------------------------------------------------------------------------------------

def _q(values: List[float]) -> dict:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return {}
    return {k: _r(np.percentile(v, p), 4) for k, p in (("p1", 1), ("p5", 5), ("p50", 50), ("p95", 95), ("p99", 99))}


def _aggregate(rows: Sequence[dict], variant: str) -> dict:
    res, tp_g, fp_g, fn_g = [], 0, 0, 0
    for r in rows:
        m = r.get("vuelta", {}).get(variant)
        if not m:
            continue
        res.append(CaseResult(case_id=r["case_id"], dice_gtvp=m["dice_gtvp"], gtvp_in_ref=m["gtvp_in_ref"],
                              gtvn_inter_mm3=m["gtvn_inter_mm3"], gtvn_sum_mm3=m["gtvn_sum_mm3"], gtvn_tp=m["tp"],
                              gtvn_fp=m["fp"], gtvn_fn=m["fn"], dice_gtvn_case=m["dice_gtvn_case"],
                              gtvp_inter_mm3=m["gtvp_inter_mm3"], gtvp_sum_mm3=m["gtvp_sum_mm3"]))
        tp_g, fp_g, fn_g = tp_g + m["tp_grupos"], fp_g + m["fp_grupos"], fn_g + m["fn_grupos"]
    if not res:
        return {}
    a = aggregate(res)
    den = 2 * tp_g + fp_g + fn_g
    return {"n": a["n_cases"], "gtvp_dsc": _r(a["gtvp_dice_mean"], 4), "gtvn_dscagg": _r(a["gtvn_dice_agg"], 4),
            "f1_hungaro": _r(a["gtvn_f1_agg"], 4), "f1_grupos": _r(2 * tp_g / den if den else 1.0, 4),
            "ganglios_fn_hungaro": a["gtvn_fn"], "ganglios_fp_hungaro": a["gtvn_fp"]}


def summarize(rows: Sequence[dict], meta: dict) -> dict:
    ok = [r for r in rows if "error" not in r]
    s: Dict = {**meta, "n_casos": len(rows), "errores": [r["case_id"] for r in rows if "error" in r]}

    # escala del PET
    suv = [r["pet"]["suv_tejido"] for r in ok]
    ref_medida = float(np.median([x for x in suv if x is not None])) if suv else float("nan")
    flagged = [r for r in ok if r["pet"]["factor"] != 1.0]
    order = sorted((r["pet"]["suv_tejido"], r["case_id"]) for r in ok if r["pet"]["suv_tejido"] is not None)
    s["pet"] = {"suv_tejido": _q(suv), "referencia_medida": _r(ref_medida, 4), "mas_bajos": order[:5],
                "mas_altos": order[-5:], "rango_regla": list(pet_scale.RANGO),
                "corregidos": [{"caso": r["case_id"], "split": r["split"], **r["pet"],
                                "factor_con_referencia_medida": _r(ref_medida / r["pet"]["suv_tejido"], 3)}
                               for r in flagged]}

    # profundidad del ROI
    roi = {}
    base_vox = sum(r["roi_270"]["voxeles"] for r in ok if "roi_270" in r)
    for d in DOWNS:
        key = f"roi_{d}"
        cut = [r for r in ok if key in r and any(r[key][c] is not None and r[key][c] < COV_MIN
                                                  for c in ("cobertura_gtvn", "cobertura_gtvp"))]
        roi[str(d)] = {
            "casos_con_etiqueta_fuera": len(cut),
            "por_split": {sp: sum(1 for r in cut if r["split"] == sp) for sp in ("train", "val", "test")},
            "casos": [{"caso": r["case_id"], "split": r["split"], "cobertura_gtvn": r[key]["cobertura_gtvn"],
                       "cobertura_gtvp": r[key]["cobertura_gtvp"], "gtvn_fuera_ml": r[key]["gtvn_fuera_ml"]} for r in cut],
            "voxeles_frente_a_270": _r(sum(r[key]["voxeles"] for r in ok if key in r) / base_vox, 4) if base_vox else None,
            "altura_mediana_mm": _r(float(np.median([r[key]["z_mm"] for r in ok if key in r])), 1),
        }
    s["roi"] = roi

    # etiquetas en la rejilla de trabajo
    lab_rows = [r for r in ok if "etiquetas" in r]
    lab_sum = {}
    for v in ("cercano", "nodos"):
        d = {"casos_distintos": 0, "fusiones": 0, "particiones": 0, "casos": []}
        for r in lab_rows:
            o, w = r["etiquetas"]["original"], r["etiquetas"][v]
            diffs = {c: w[f"{c}_comp"] - o[f"{c}_comp"] for c in ("gtvp", "gtvn")}
            if any(diffs.values()):
                d["casos_distintos"] += 1
                d["fusiones"] += sum(-x for x in diffs.values() if x < 0)
                d["particiones"] += sum(x for x in diffs.values() if x > 0)
                d["casos"].append({"caso": r["case_id"], "gtvp": [o["gtvp_comp"], w["gtvp_comp"]], "gtvn": [o["gtvn_comp"], w["gtvn_comp"]]})
        lab_sum[v] = d
    lab_sum["nodos"]["rescatados"] = sum(r["etiquetas"]["nodos"]["rescatados"] for r in lab_rows)
    lab_sum["nodos"]["siguen_partidos"] = sum(r["etiquetas"]["nodos"]["partidos"] for r in lab_rows)
    vol = {}
    for v in ("cercano", "nodos", "suave"):
        errs = []
        for r in lab_rows:
            o = r["etiquetas"]["original"]
            for c in ("gtvp", "gtvn"):
                if o[f"{c}_ml"] and o[f"{c}_ml"] < 2.0:  # estructuras pequenas: donde la rejilla de 3 mm pesa
                    errs.append(abs(r["etiquetas"][v][f"{c}_ml"] / o[f"{c}_ml"] - 1.0))
        vol[v] = {"error_relativo_mediano": _r(float(np.median(errs)), 4) if errs else None,
                  "error_relativo_p90": _r(float(np.percentile(errs, 90)), 4) if errs else None, "n": len(errs)}
    lab_sum["volumen_clases_menores_de_2ml"] = vol
    s["etiquetas"] = lab_sum

    # vuelta a la rejilla original
    fine = [r for r in lab_rows if (r.get("dz_original_mm") or 9) <= 2.0]
    s["vuelta"] = {"todos": {v: _aggregate(lab_rows, v) for v in VARIANTES},
                   "cortes_de_2mm_o_menos": {v: _aggregate(fine, v) for v in VARIANTES},
                   "por_split": {sp: {v: _aggregate([r for r in lab_rows if r["split"] == sp], v) for v in VARIANTES}
                                 for sp in ("train", "val", "test")}}
    best = []
    for r in lab_rows:
        ref_m, sl_m = r["vuelta"][REFERENCIA], r["vuelta"]["suave+lineal"]
        if ref_m["gtvp_in_ref"]:
            best.append((round(sl_m["dice_gtvp"] - ref_m["dice_gtvp"], 4), r["case_id"], ref_m["dice_gtvp"], sl_m["dice_gtvp"]))
    best.sort(reverse=True)
    s["vuelta"]["gtvp_mas_mejorados_por_suave_lineal"] = best[:10]
    s["vuelta"]["gtvp_mas_empeorados_por_suave_lineal"] = best[-5:]
    return s


def _flat(r: dict) -> dict:
    out = {"case_id": r["case_id"], "centro": r["centro"], "split": r["split"], "dz_original_mm": r.get("dz_original_mm"),
           "error": "si" if "error" in r else ""}
    if "error" in r:
        return out
    for k, v in r["pet"].items():
        out[f"pet_{k}"] = v
    for d in DOWNS:
        for k, v in r.get(f"roi_{d}", {}).items():
            out[f"roi{d}_{k}"] = v
    for v, m in r.get("vuelta", {}).items():
        for k in ("dice_gtvp", "dice_gtvn_case", "tp", "fp", "fn", "tp_grupos", "fp_grupos", "fn_grupos"):
            out[f"{v}_{k}"] = m[k]
    for v, m in r.get("etiquetas", {}).items():
        for k, val in m.items():
            out[f"lab_{v}_{k}"] = val
    return out


def write_outputs(rows: Sequence[dict], out: Path, meta: dict) -> dict:
    s = summarize(rows, meta)
    (out / "resumen.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    flat = [_flat(r) for r in rows]
    keys: List[str] = []
    for f in flat:
        keys += [k for k in f if k not in keys]
    with open(out / "casos.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    return s


def print_summary(s: dict) -> None:
    print("=" * 78)
    print(f"experimento de preprocesado: {s['n_casos']} casos, errores {len(s['errores'])}")
    p = s["pet"]
    print(f"PET: mediana del tejido blando {p['suv_tejido']}; referencia medida {p['referencia_medida']}")
    for c in p["corregidos"]:
        print(f"  corregido {c['caso']} ({c['split']}): tejido {c['suv_tejido']} -> factor {c['factor']} "
              f"(con la referencia medida {c['factor_con_referencia_medida']}); ancla {c.get('ancla_antes')} -> {c.get('ancla_despues')}")
    for d, r in s["roi"].items():
        print(f"ROI {d} mm: {r['casos_con_etiqueta_fuera']} casos con etiqueta fuera {r['por_split']}; "
              f"voxeles x{r['voxeles_frente_a_270']}; altura mediana {r['altura_mediana_mm']} mm")
    e = s["etiquetas"]
    print(f"componentes distintos del original: cercano {e['cercano']['casos_distintos']} casos "
          f"({e['cercano']['fusiones']} fusiones, {e['cercano']['particiones']} particiones); nodos {e['nodos']['casos_distintos']} "
          f"(rescatados {e['nodos']['rescatados']}, siguen partidos {e['nodos']['siguen_partidos']})")
    print(f"error de volumen en clases < 2 ml (mediana / p90): " + "; ".join(
        f"{v} {m['error_relativo_mediano']} / {m['error_relativo_p90']}" for v, m in e["volumen_clases_menores_de_2ml"].items()))
    for grupo in ("todos", "cortes_de_2mm_o_menos"):
        print(f"vuelta ({grupo}):")
        for v, m in s["vuelta"][grupo].items():
            if m:
                print(f"  {v:16s} GTVp {m['gtvp_dsc']}  DSCagg {m['gtvn_dscagg']}  F1 hungaro {m['f1_hungaro']}  F1 grupos {m['f1_grupos']}  (n={m['n']})")
    print("=" * 78)


def _read_jsonl(path: Path) -> Dict[str, dict]:
    done: Dict[str, dict] = {}
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
    ap.add_argument("--out", default="runs/experimento_preproc")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--max-cases", type=int, default=None)
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
    print(f"{len(todo)} casos; por hacer {len(pending)} (ya hechos {len(done)})", flush=True)
    t0 = time.time()
    mod = importlib.import_module("bifusion3d.tools.experimento_preproc")  # por nombre: los procesos hijos lo importan igual
    with open(jsonl, "a", encoding="utf-8") as fj:
        def keep(row):
            fj.write(json.dumps(row, ensure_ascii=False) + "\n")
            fj.flush()
            if "error" not in row:
                done[row["case_id"]] = row
            print(f"  {row['case_id']} ({row['split']}) {row.get('segundos', '-')} s" + (" ERROR" if "error" in row else ""), flush=True)

        if args.workers <= 1:
            for c in pending:
                keep(mod._worker(c, data_cfg, split_of.get(c.case_id, "fuera"), args.threads))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(mod._worker, c, data_cfg, split_of.get(c.case_id, "fuera"), args.threads) for c in pending]
                for fut in as_completed(futs):
                    keep(fut.result())
    rows = [done[c.case_id] for c in todo if c.case_id in done]
    errors = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if "error" in e and e["case_id"] not in done:
            errors.append(e)
    meta = {"config": args.config, "fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "segundos_totales": round(time.time() - t0, 1)}
    s = write_outputs(rows + errors, out, meta)
    print_summary(s)
    for e in errors[:3]:
        print(f"ERROR en {e['case_id']}:\n{e['error']}")
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
