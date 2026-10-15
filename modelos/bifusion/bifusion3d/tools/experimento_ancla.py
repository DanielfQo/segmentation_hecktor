"""Ancla del ROI absoluta (PET > 2) frente a relativa (umbrales en veces el tejido blando del paciente), 680 casos.

    python -m bifusion3d.tools.experimento_ancla --config configs/siam_full.yaml --down 300 --out runs/experimento_ancla

Para cada caso: escala (tejido / 0,82), si hay ancla con cada regla, la altura del ancla, cuanto se mueve el ROI
(mm en cada cara y voxeles) y la cobertura de GTVp y GTVn con cada ROI. Solo CPU; si se corta, al relanzar sigue.
Salida: casos.jsonl, casos.csv y resumen.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..guard import refuse_inside_mamba

import SimpleITK as sitk  # noqa: E402

from mambalite3d.config import load_config, resolve  # noqa: E402
from mambalite3d.data import io as mio  # noqa: E402
from mambalite3d.data.splits import select_subset, split_cases  # noqa: E402

from ..data import preproc  # noqa: E402
from .audit_data import coverage  # noqa: E402
from .verify_preproc import roi_formula  # noqa: E402


def _bbox(arr):
    nz = np.nonzero(arr)
    if not len(nz[0]):
        return None
    return tuple(int(a.min()) for a in nz), tuple(int(a.max()) for a in nz)


def ancla_case(files: mio.CaseFiles, data_cfg: dict, split: str = "") -> dict:
    t0 = time.time()
    spacing = tuple(float(s) for s in data_cfg["target_spacing_xyz"])
    ct, pet = mio.read_lps(files.ct), mio.read_lps(files.pet)
    roi_abs = dict(data_cfg["roi"])
    roi_rel, escala = preproc.roi_cfg_relativo(ct, pet, roi_abs)
    lab = mio.read_lps(files.label) if files.label is not None else None
    lab_arr = sitk.GetArrayFromImage(lab) if lab is not None else None
    bbox = _bbox(lab_arr) if lab is not None else None
    row = {"case_id": files.case_id, "centro": files.center, "split": split, "escala": round(escala, 4)}
    boxes = {}
    for name, cfg in (("absoluta", roi_abs), ("relativa", roi_rel)):
        f = roi_formula(ct, pet, cfg, spacing)
        lo, hi = mio.compute_roi(ct, pet, cfg, spacing)
        size = np.maximum(np.floor((hi - lo) / np.array(spacing)).astype(int) + 1, 1)
        boxes[name] = (lo, hi)
        r = {"ancla": bool(f["ancla"]), "ancla_z_mm": f["ancla_z_mm"], "voxeles": int(np.prod(size)),
             "alto_mm": round(float(hi[2] - lo[2]), 1)}
        if lab is not None:
            cov = coverage(lab, lab_arr, bbox, lo, size, spacing)
            r["cobertura_gtvp"], r["cobertura_gtvn"] = cov["cobertura_gtvp"], cov["cobertura_gtvn"]
        row[name] = r
    d = np.concatenate([boxes["relativa"][0] - boxes["absoluta"][0], boxes["relativa"][1] - boxes["absoluta"][1]])
    row["mueve_mm"] = [round(float(v), 1) for v in d]  # lo x, y, z; hi x, y, z
    row["mueve_max_mm"] = round(float(np.abs(d).max()), 1)
    row["segundos"] = round(time.time() - t0, 1)
    return row


def _worker(files, data_cfg, split, threads):
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(threads))
    except AttributeError:
        pass
    try:
        return ancla_case(files, data_cfg, split)
    except Exception:  # noqa: BLE001
        return {"case_id": files.case_id, "split": split, "error": traceback.format_exc()}


def _cov_loss(r):
    out = 0.0
    for k in ("cobertura_gtvp", "cobertura_gtvn"):
        a, b = r["absoluta"].get(k), r["relativa"].get(k)
        if a is not None and b is not None:
            out = min(out, b - a)
    return out


def summarize(rows) -> dict:
    ok = [r for r in rows if "error" not in r]
    moved = [r for r in ok if r["mueve_max_mm"] > 0]
    esc = np.array([r["escala"] for r in ok]) if ok else np.zeros(0)
    s = {
        "n_casos": len(ok), "errores": [r["case_id"] for r in rows if "error" in r],
        "escala": {f"p{q}": round(float(np.percentile(esc, q)), 3) for q in (1, 5, 50, 95, 99)} if len(esc) else {},
        "sin_ancla": {k: [r["case_id"] for r in ok if not r[k]["ancla"]] for k in ("absoluta", "relativa")},
        "casos_que_se_mueven": len(moved),
        "se_mueven_mas_de": {str(t): len([r for r in ok if r["mueve_max_mm"] > t]) for t in (4, 8, 16, 32)},
        "mueve_max_mm_cuantiles": {f"p{q}": round(float(np.percentile([r["mueve_max_mm"] for r in ok], q)), 1)
                                   for q in (50, 90, 99, 100)} if ok else {},
        "voxeles_relativa_sobre_absoluta": round(sum(r["relativa"]["voxeles"] for r in ok) /
                                                 max(1, sum(r["absoluta"]["voxeles"] for r in ok)), 4),
        "pierden_cobertura": sorted([(r["case_id"], r["split"], round(_cov_loss(r), 4)) for r in ok if _cov_loss(r) < 0],
                                    key=lambda x: x[2]),
        "etiqueta_fuera": {k: [r["case_id"] for r in ok if min(v for v in (r[k].get("cobertura_gtvp"), r[k].get("cobertura_gtvn"), 1.0)
                                                                  if v is not None) < 0.999] for k in ("absoluta", "relativa")},
        "mayores_movimientos": [(r["case_id"], r["split"], r["escala"], r["mueve_max_mm"], r["mueve_mm"])
                                for r in sorted(ok, key=lambda r: -r["mueve_max_mm"])[:15]],
    }
    return s


def _read(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r:
                done[r["case_id"]] = r
    return done


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    refuse_inside_mamba(argv, flags=("--out",))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--down", type=float, default=300.0)
    ap.add_argument("--out", default="runs/experimento_ancla")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-cases", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    data_cfg = preproc.with_options(cfg["data"], down_mm=args.down)
    all_cases = [c for c in mio.find_cases(resolve(data_cfg["root"])) if c.label is not None]
    subset = select_subset(all_cases, data_cfg["cases_per_center"], data_cfg["centers"], cfg["seed"])
    tr, va, te = split_cases(subset, data_cfg["split"], cfg["seed"])
    split_of = {**{c.case_id: "train" for c in tr}, **{c.case_id: "val" for c in va}, **{c.case_id: "test" for c in te}}
    todo = all_cases[: args.max_cases] if args.max_cases else all_cases
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "casos.jsonl"
    done = _read(jsonl)
    if jsonl.exists() and jsonl.stat().st_size and not jsonl.read_bytes().endswith(b"\n"):
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write("\n")
    pending = [c for c in todo if c.case_id not in done]
    print(f"{len(todo)} casos; por hacer {len(pending)} (ROI {args.down} mm)", flush=True)
    errors = []
    with open(jsonl, "a", encoding="utf-8") as fj:
        def keep(r):
            fj.write(json.dumps(r) + "\n")
            fj.flush()
            if "error" in r:
                errors.append(r)
            else:
                done[r["case_id"]] = r
            print(f"  {r['case_id']} escala {r.get('escala', '-')} mueve {r.get('mueve_max_mm', '-')} mm"
                  + (" ERROR" if "error" in r else ""), flush=True)

        if args.workers <= 1:
            for c in pending:
                keep(_worker(c, data_cfg, split_of.get(c.case_id, "fuera"), args.threads))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(_worker, c, data_cfg, split_of.get(c.case_id, "fuera"), args.threads) for c in pending]
                for fut in as_completed(futs):
                    keep(fut.result())
    rows = [done[c.case_id] for c in todo if c.case_id in done] + errors
    s = summarize(rows)
    s["down_mm"] = args.down
    (out / "resumen.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "casos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "split", "escala", "ancla_abs", "ancla_rel", "mueve_max_mm", "cob_gtvn_abs", "cob_gtvn_rel"])
        for r in rows:
            if "error" not in r:
                w.writerow([r["case_id"], r["split"], r["escala"], r["absoluta"]["ancla"], r["relativa"]["ancla"],
                            r["mueve_max_mm"], r["absoluta"].get("cobertura_gtvn"), r["relativa"].get("cobertura_gtvn")])
    print(json.dumps({k: v for k, v in s.items() if k != "mayores_movimientos"}, indent=1, ensure_ascii=False))
    print("mayores movimientos:", *s["mayores_movimientos"], sep="\n  ")
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
