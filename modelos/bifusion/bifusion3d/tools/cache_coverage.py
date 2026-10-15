"""Que casos de train vio realmente un entrenamiento con la cache rodante de mambalite3d.

    python -m bifusion3d.tools.cache_coverage --runs runs/siam_full runs/siam_light ../mamba/runs/full_base_fold0
    python -m bifusion3d.tools.cache_coverage --runs runs/siam_full --out runs/verificacion/cache_siam_full.json

El entrenamiento guarda en RAM solo `max_cached_cases` casos de train (60) y cambia `refresh_per_epoch` (10) al final
de cada epoca. Los parches de una epoca salen solo de los casos que estan en la cache durante esa epoca. Esta
herramienta reproduce esa cache con la clase CaseStore de mambalite3d (la misma, con su generador y su semilla) pero
sin leer imagenes, y la recorre sesion por sesion tal como lo hizo el entrenamiento: cada vez que se reanuda, la
cache vuelve a empezar desde la semilla.

De log.txt toma las sesiones (cada linea "experimento ...") y las epocas que termino cada una (lineas "epoca {...}").
Una epoca repetida tras un corte cuenta solo la ultima vez, que es la que quedo en los pesos. La simulacion es exacta
si cada recambio encontro terminadas las cargas de la epoca anterior (una carga tarda ~2 s, una epoca minutos); se
comprueba comparando, epoca por epoca, los casos nuevos simulados con `cache_new_cases` del log.

Solo lee config_used.json, split.json y log.txt del run; no escribe nada dentro de modelos/mamba.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import wait
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_EPOCH_LINE = re.compile(r"\] epoca (\{.*\})\s*$")
_SESSION_LINE = re.compile(r"\] experimento \S+\s+variante")


def parse_sessions(log_text: str) -> List[List[dict]]:
    """Lista de sesiones de entrenamiento; cada una, la lista de registros de epoca que termino, en orden."""
    sessions: List[List[dict]] = []
    current: Optional[List[dict]] = None
    for line in log_text.splitlines():
        if _SESSION_LINE.search(line):
            current = []
            sessions.append(current)
            continue
        m = _EPOCH_LINE.search(line)
        if m and current is not None:
            try:
                current.append(json.loads(m.group(1)))
            except ValueError:
                pass
    return [s for s in sessions if s]


def _dummy_case(files):
    from mambalite3d.data.io import CaseData

    return CaseData(files.case_id, files.center, np.zeros((2, 1, 1, 1), np.float16), None, (1.0, 1.0, 3.0), (0.0, 0.0, 0.0),
                    files=files)


def simulate_store(case_ids: Sequence[str], sessions: Sequence[Sequence[dict]], data_cfg: dict, seed: int):
    """Reproduce CaseStore sesion a sesion. Devuelve, por epoca efectiva, el conjunto de casos en cache, y la validacion.

    Usa mambalite3d.data.sampler.CaseStore sin cambios; solo sustituye load_case por una carga vacia mientras dura.
    """
    import mambalite3d.data.sampler as msamp
    from mambalite3d.data.io import CaseFiles

    files = [CaseFiles(cid, cid.split("-")[0], Path(f"{cid}__CT.nii.gz"), Path(f"{cid}__PT.nii.gz"), Path(f"{cid}.nii.gz"))
             for cid in case_ids]
    real_load = msamp.load_case
    msamp.load_case = lambda f, cfg, with_label=True, seed=0, **kw: _dummy_case(f)
    effective: Dict[int, frozenset] = {}
    checks = {"epocas": 0, "coinciden": 0, "distintas": []}
    try:
        for s_idx, sess in enumerate(sessions):
            store = msamp.CaseStore(files, data_cfg, max_cases=data_cfg["max_cached_cases"],
                                    threads=data_cfg.get("load_threads", 2), seed=seed, log=lambda *a, **k: None)
            store.preload()
            for rec in sess:
                effective[int(rec["epoch"])] = frozenset(store.loaded)  # cache con la que se entreno esta epoca
                wait(list(store.pending.values()))  # en el entrenamiento real, las cargas terminan durante la epoca
                new = store.refresh(data_cfg["refresh_per_epoch"]) if store.max_cases < len(store.all_cases) else 0
                if "cache_new_cases" in rec:
                    checks["epocas"] += 1
                    if int(rec["cache_new_cases"]) == new:
                        checks["coinciden"] += 1
                    else:
                        checks["distintas"].append({"sesion": s_idx, "epoca": int(rec["epoch"]),
                                                    "log": int(rec["cache_new_cases"]), "simulada": new})
            store.close()
    finally:
        msamp.load_case = real_load
    return effective, checks


def window_schedule(case_ids: Sequence[str], epochs: int, cached: int, step: int, seed: int) -> Dict[int, frozenset]:
    """Cache por ventana deslizante sobre una permutacion fija (propuesta de la Fase 2): depende solo de la epoca."""
    rng = np.random.default_rng(seed)
    perm = [case_ids[i] for i in rng.permutation(len(case_ids))]
    n = len(perm)
    return {e: frozenset(perm[(step * e + k) % n] for k in range(min(cached, n))) for e in range(epochs)}


def exposure(case_ids: Sequence[str], schedule: Dict[int, frozenset], patches_per_epoch: int) -> dict:
    """Epocas en cache por caso y parches esperados por caso (los parches de una epoca se reparten entre la cache)."""
    epochs = {cid: 0 for cid in case_ids}
    patches = {cid: 0.0 for cid in case_ids}
    for e, cache in schedule.items():
        share = patches_per_epoch / max(len(cache), 1)
        for cid in cache:
            epochs[cid] += 1
            patches[cid] += share
    vals = np.array([epochs[c] for c in case_ids], dtype=float)
    unseen = sorted(c for c in case_ids if epochs[c] == 0)
    return {
        "casos": len(case_ids),
        "epocas_simuladas": len(schedule),
        "sin_ver": len(unseen),
        "casos_sin_ver": unseen,
        "epocas_en_cache_min_p5_p50_p95_max": [float(np.min(vals)), float(np.percentile(vals, 5)), float(np.median(vals)),
                                               float(np.percentile(vals, 95)), float(np.max(vals))],
        "parches_por_caso_p5_p50_p95": [round(float(np.percentile(list(patches.values()), q)), 1) for q in (5, 50, 95)],
        "epocas_por_caso": epochs,
    }


def analyze_run(run: Path) -> dict:
    cfg = json.loads((run / "config_used.json").read_text(encoding="utf-8"))
    split = json.loads((run / "split.json").read_text(encoding="utf-8"))
    log_text = (run / "log.txt").read_text(encoding="utf-8", errors="replace")
    data_cfg, tr = cfg["data"], cfg["train"]
    train_ids = list(split["train"])  # mismo orden que recibe CaseStore (split_cases devuelve ordenado por id)
    sessions = parse_sessions(log_text)
    out = {"run": str(run), "experimento": cfg.get("experiment"), "semilla": cfg["seed"],
           "cache": {"max_cached_cases": data_cfg.get("max_cached_cases"), "refresh_per_epoch": data_cfg.get("refresh_per_epoch")},
           "sesiones": [{"desde_epoca": int(s[0]["epoch"]), "epocas": len(s)} for s in sessions]}
    if not data_cfg.get("max_cached_cases") or data_cfg["max_cached_cases"] >= len(train_ids):
        out["nota"] = "sin cache rodante: todos los casos de train estan en RAM en cada epoca"
        return out
    effective, checks = simulate_store(train_ids, sessions, data_cfg, int(cfg["seed"]))
    patches = int(tr["iters_per_epoch"]) * int(tr["batch_size"])
    out["validacion"] = {"epocas_con_dato": checks["epocas"], "coinciden": checks["coinciden"],
                         "distintas": checks["distintas"][:20], "exacta": checks["coinciden"] == checks["epocas"]}
    out["real"] = exposure(train_ids, effective, patches)
    out["propuesta_ventana"] = exposure(train_ids, window_schedule(train_ids, int(tr["epochs"]), int(data_cfg["max_cached_cases"]),
                                                                   int(data_cfg["refresh_per_epoch"]), int(cfg["seed"])), patches)
    out["propuesta_ventana"].pop("epocas_por_caso")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="carpetas de runs con config_used.json, split.json y log.txt")
    ap.add_argument("--out", default=None, help="JSON de salida (por defecto solo imprime el resumen)")
    args = ap.parse_args(argv)
    results = []
    for r in args.runs:
        res = analyze_run(Path(r))
        results.append(res)
        if "real" not in res:
            print(f"{r}: {res.get('nota')}")
            continue
        real, prop, val = res["real"], res["propuesta_ventana"], res["validacion"]
        print(f"{r}: {len(res['sesiones'])} sesiones {[(s['desde_epoca'], s['epocas']) for s in res['sesiones']]}")
        print(f"  simulacion exacta: {val['exacta']} ({val['coinciden']}/{val['epocas_con_dato']} epocas coinciden con el log)")
        print(f"  cache real: {real['sin_ver']} de {real['casos']} casos de train nunca entraron; "
              f"epocas en cache min/p5/p50/p95/max {real['epocas_en_cache_min_p5_p50_p95_max']}")
        print(f"  ventana fija (propuesta): {prop['sin_ver']} sin ver; epocas en cache {prop['epocas_en_cache_min_p5_p50_p95_max']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"resultado en {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
