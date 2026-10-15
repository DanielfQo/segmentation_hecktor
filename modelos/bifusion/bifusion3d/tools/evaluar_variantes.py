"""Evalua checkpoints ya entrenados con variantes de inferencia, sin entrenar: profundidad del ROI y vuelta.

    python -m bifusion3d.tools.evaluar_variantes --runs runs/siam_full:final ../mamba/runs/full_base_fold0:final --split val

Para cada run y cada profundidad del ROI (--down, por defecto 270 y 300 mm) infiere una vez cada caso y evalua esa
prediccion con la vuelta cercana (la de siempre) y con la lineal (bifusion3d.data.restore). Metricas oficiales en la
rejilla original: GTVp DSC, GTVn DSCagg y F1 de ganglios, con el emparejamiento hungaro y por grupos. Cada variante
se compara con la de siempre (270 mm, vuelta cercana) en los mismos pacientes: diferencia e IC del 95 % por
bootstrap.

Lee los runs sin escribir en ellos (tambien los de modelos/mamba); todo va a --out/<run>-<checkpoint>/:
casos.jsonl (si se corta, al relanzar sigue), metricas.json y comparacion.csv. Con --thermal 83:72 (por defecto)
pausa entre casos si la GPU se calienta.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .. import thermal
from ..guard import refuse_inside_mamba

REF = (270, "nearest")
METRICS = ("gtvp_dsc", "gtvn_dscagg", "f1_hungaro", "f1_grupos")


def metric_values(rows: Sequence[dict]) -> Dict[str, float]:
    """Las metricas oficiales de un conjunto de casos (filas con los conteos por caso)."""
    if not rows:
        return {}
    a = {k: np.array([r[k] for r in rows], dtype=float) for k in
         ("dice_gtvp", "gtvn_inter_mm3", "gtvn_sum_mm3", "tp", "fp", "fn", "tp_grupos", "fp_grupos", "fn_grupos")}
    return _metrics_from(a, np.arange(len(rows))[None, :])[0] | {"n": len(rows)}


def _metrics_from(a: Dict[str, np.ndarray], idx: np.ndarray) -> List[Dict[str, float]]:
    """Metricas para cada fila de `idx` (B, n): indices de pacientes (remuestreo bootstrap o identidad)."""
    s = {k: v[idx] for k, v in a.items()}
    inter, total = s["gtvn_inter_mm3"].sum(1), s["gtvn_sum_mm3"].sum(1)
    out = {
        "gtvp_dsc": s["dice_gtvp"].mean(1),
        "gtvn_dscagg": np.where(total > 0, 2 * inter / np.maximum(total, 1e-9), 1.0),
    }
    for name, (tp, fp, fn) in (("f1_hungaro", ("tp", "fp", "fn")), ("f1_grupos", ("tp_grupos", "fp_grupos", "fn_grupos"))):
        t, p, n = s[tp].sum(1), s[fp].sum(1), s[fn].sum(1)
        den = 2 * t + p + n
        out[name] = np.where(den > 0, 2 * t / np.maximum(den, 1e-9), 1.0)
    return [{k: float(v[i]) for k, v in out.items()} for i in range(idx.shape[0])]


def paired_bootstrap(ref: Sequence[dict], var: Sequence[dict], n_boot: int = 2000, seed: int = 0) -> Dict[str, dict]:
    """Diferencia (variante - referencia) de cada metrica en los mismos pacientes, con IC del 95 %."""
    by_ref = {r["case_id"]: r for r in ref}
    pairs = [(by_ref[v["case_id"]], v) for v in var if v["case_id"] in by_ref]
    if not pairs:
        return {}
    keys = ("dice_gtvp", "gtvn_inter_mm3", "gtvn_sum_mm3", "tp", "fp", "fn", "tp_grupos", "fp_grupos", "fn_grupos")
    a_ref = {k: np.array([p[0][k] for p in pairs], dtype=float) for k in keys}
    a_var = {k: np.array([p[1][k] for p in pairs], dtype=float) for k in keys}
    n = len(pairs)
    rng = np.random.default_rng(seed)
    idx = np.concatenate([np.arange(n)[None, :], rng.integers(0, n, size=(n_boot, n))])
    m_ref, m_var = _metrics_from(a_ref, idx), _metrics_from(a_var, idx)
    out = {}
    for k in METRICS:
        d = np.array([m_var[i][k] - m_ref[i][k] for i in range(len(idx))])
        lo, hi = np.percentile(d[1:], [2.5, 97.5])
        out[k] = {"diferencia": round(float(d[0]), 4), "ic95": [round(float(lo), 4), round(float(hi), 4)]}
    return out


def case_metrics(case_id: str, pred: np.ndarray, gt: np.ndarray, voxel_mm3: float) -> dict:
    from mambalite3d.eval.metrics import evaluate_case

    from ..eval_groups import match_groups

    r = evaluate_case(case_id, pred, gt, voxel_mm3)
    tp, fp, fn = match_groups(pred == 2, gt == 2)
    return {"dice_gtvp": round(r.dice_gtvp, 5), "gtvp_in_ref": r.gtvp_in_ref, "gtvn_inter_mm3": round(r.gtvn_inter_mm3, 2),
            "gtvn_sum_mm3": round(r.gtvn_sum_mm3, 2), "tp": r.gtvn_tp, "fp": r.gtvn_fp, "fn": r.gtvn_fn,
            "tp_grupos": tp, "fp_grupos": fp, "fn_grupos": fn}


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _dedup(rows: Sequence[dict]) -> List[dict]:
    """Una fila por (caso, profundidad, vuelta): la ultima escrita (un caso cortado se repite entero)."""
    return list({(r["case_id"], r["down"], r["restore"]): r for r in rows}.values())


def _done_keys(rows: Sequence[dict], restores: Sequence[str]) -> set:
    """(caso, profundidad) con todas las vueltas pedidas ya escritas; si un corte cayo entre dos vueltas, se repite."""
    have: Dict[tuple, set] = {}
    for r in rows:
        have.setdefault((r["case_id"], r["down"]), set()).add(r["restore"])
    return {k for k, v in have.items() if set(restores) <= v}


def _end_line(path: Path) -> None:
    """Si un corte dejo la ultima linea a medias, la cierra para que la fila siguiente no se pegue a ella."""
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            last = f.read(1)
        if last != b"\n":
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")


def summarize(rows: Sequence[dict], downs: Sequence[int], restores: Sequence[str], n_boot: int) -> dict:
    groups = {(d, r): [x for x in rows if x["down"] == d and x["restore"] == r] for d in downs for r in restores}
    ref = groups.get(REF, [])
    out = {"referencia": f"{REF[0]} mm, vuelta {REF[1]}", "variantes": {}}
    for (d, r), g in groups.items():
        key = f"{d}mm_{r}"
        out["variantes"][key] = {"metricas": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metric_values(g).items()}}
        if (d, r) != REF and ref:
            out["variantes"][key]["frente_a_referencia"] = paired_bootstrap(ref, g, n_boot)
    return out


def evaluate_run(spec: str, args, device, guard) -> dict:
    import torch  # noqa: F401  (cargado por mambalite3d)
    from mambalite3d.config import resolve
    from mambalite3d.data.io import find_cases
    from mambalite3d.data.sampler import iter_cases
    from mambalite3d.tools import evaluate as base_eval
    from mambalite3d.training.inference import sliding_window_predict

    from ..data.restore import restore_case

    run_s, _, ckpt = spec.rpartition(":")
    if not run_s or ckpt not in ("final", "best", "latest"):
        raise SystemExit(f"run invalido {spec!r}: usa RUN:final, RUN:best o RUN:latest")
    run = resolve(run_s)
    model, cfg = base_eval.load_member(run, ckpt, device)
    ids = set(base_eval._read_split(run).get(args.split, []))
    files = [c for c in find_cases(resolve(cfg["data"]["root"])) if c.case_id in ids]
    if args.max_cases:
        files = files[: args.max_cases]
    out = Path(args.out) / f"{run.name}-{ckpt}"
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "casos.jsonl"
    rows = _dedup(_read_jsonl(jsonl))
    done = _done_keys(rows, args.restore)
    _end_line(jsonl)
    patch_zyx = [int(p) for p in cfg["model"]["patch_size_zyx"]]
    k = cfg["train"]["num_classes"]
    print(f"{run.name}:{ckpt} en {args.split}: {len(files)} casos, ROI {args.down} mm, vuelta {args.restore}", flush=True)
    with open(jsonl, "a", encoding="utf-8") as fj:
        for down in args.down:
            todo = [f for f in files if (f.case_id, down) not in done]
            if not todo:
                continue
            data_cfg = copy.deepcopy(cfg["data"])
            data_cfg["roi"]["down_mm"] = float(down)
            for case in iter_cases(todo, data_cfg, threads=data_cfg.get("load_threads", 2)):
                guard.tick(force=True)
                t0 = time.time()
                probs = sliding_window_predict(model, case.image.astype(np.float32), patch_zyx, k, device,
                                               step_fraction=cfg["infer"]["step_fraction"], amp=cfg["train"]["amp"],
                                               mirror=False, pad_values=case.pad_values,
                                               batch_size=cfg["infer"].get("batch_size", 4))
                for mode in args.restore:
                    pred, gt, vox = restore_case(case, probs, mode)
                    row = {"case_id": case.case_id, "centro": case.center, "down": down, "restore": mode,
                           **case_metrics(case.case_id, pred, gt, vox)}
                    fj.write(json.dumps(row) + "\n")
                    fj.flush()
                    rows.append(row)
                r = rows[-1]
                print(f"  {case.case_id} ROI {down} mm {list(case.image.shape[1:])}: GTVp {r['dice_gtvp']:.3f}, ganglios "
                      f"TP/FP/FN {r['tp']}/{r['fp']}/{r['fn']} ({time.time() - t0:.1f} s)", flush=True)
    s = summarize(_dedup(rows), args.down, args.restore, args.boot)
    s.update(run=str(run), checkpoint=ckpt, split=args.split)
    (out / "metricas.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "comparacion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variante", *METRICS, *[f"dif_{m}" for m in METRICS], *[f"ic95_{m}" for m in METRICS]])
        for key, v in s["variantes"].items():
            m, c = v["metricas"], v.get("frente_a_referencia", {})
            w.writerow([key, *[m.get(x) for x in METRICS], *[c.get(x, {}).get("diferencia") for x in METRICS],
                        *[c.get(x, {}).get("ic95") for x in METRICS]])
    print(f"== {run.name}:{ckpt} ({args.split}) ==")
    for key, v in s["variantes"].items():
        m = v["metricas"]
        line = f"  {key:14s} GTVp {m['gtvp_dsc']:.4f}  DSCagg {m['gtvn_dscagg']:.4f}  F1 hung {m['f1_hungaro']:.4f}  F1 grupos {m['f1_grupos']:.4f}"
        if "frente_a_referencia" in v:
            c = v["frente_a_referencia"]
            line += "  | " + "  ".join(f"{x.split('_')[0] if x != 'f1_grupos' else 'F1g'} {c[x]['diferencia']:+.4f} {c[x]['ic95']}" for x in METRICS)
        print(line)
    return s


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    refuse_inside_mamba(argv, flags=("--out",))
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, metavar="RUN:CHECKPOINT")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--down", nargs="+", type=int, default=[270, 300])
    ap.add_argument("--restore", nargs="+", default=["nearest", "linear"], choices=["nearest", "linear"])
    ap.add_argument("--out", default="runs/variantes")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--thermal", default="83:72", help="pausa si la GPU llega a max_c y sigue al bajar a resume_c")
    args = ap.parse_args(argv)
    if REF[0] not in args.down or REF[1] not in args.restore:
        ap.error(f"la referencia ({REF[0]} mm, vuelta {REF[1]}) tiene que estar entre las variantes")

    import torch

    from .. import patch  # importa torch y las variantes de fusion

    patch.install()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    guard = thermal.ThermalGuard({**thermal.parse_spec(args.thermal), "every": 1}, log=print)
    results = {spec: evaluate_run(spec, args, device, guard) for spec in args.runs}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"resumen_{args.split}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
