"""Validacion cruzada de una variante de fusion: cada fold es un entrenamiento de bifusion3d.train.

    python -m bifusion3d.cross_validate --config configs/siam_light_cv7.yaml --dry-run
    python -m bifusion3d.cross_validate --config configs/siam_light_cv7.yaml
    python -m bifusion3d.cross_validate --config configs/siam_light_cv7.yaml --folds 0 1 2

Igual que mambalite3d.tools.cross_validate (mismos splits, mismo resumen), pero lanza bifusion3d.train y guarda los
folds en runs/ de esta carpeta. Se puede repetir tras un corte: los folds terminados se saltan y el interrumpido se
reanuda. Argumentos despues de `--` pasan a train.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import List, Sequence, Tuple

from . import PROJECT_DIR
from .guard import refuse_inside_mamba


def fold_command(config: str, fold: int, extra: Sequence[str] = ()) -> Tuple[List[str], str]:
    """Comando y directorio de trabajo con los que se entrena un fold."""
    cmd = [sys.executable, "-m", "bifusion3d.train", "--config", config, "--fold", str(fold), "--resume", *extra]
    return cmd, PROJECT_DIR


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    refuse_inside_mamba(argv, flags=())
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from mambalite3d.config import load_config, resolve
    from mambalite3d.data.io import find_cases
    from mambalite3d.data.splits import select_subset, split_cases
    from mambalite3d.tools.cross_validate import n_folds_of, summarise

    cfg = load_config(args.config)
    n = n_folds_of(cfg)
    folds = args.folds if args.folds else list(range(n))
    out_root = resolve(cfg["output"]["root"])

    if args.dry_run:
        cases = select_subset(find_cases(resolve(cfg["data"]["root"])), cfg["data"]["cases_per_center"],
                              cfg["data"]["centers"], cfg["seed"])
        seen = []
        for k in folds:
            tr, va, te = split_cases(cases, cfg["data"]["split"], cfg["seed"], fold=k)
            centers = sorted({c.center for c in te})
            print(f"fold {k}: train {len(tr)}  val {len(va)}  test {len(te)}  centros test {centers}")
            ids = [{c.case_id for c in x} for x in (tr, va, te)]
            assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "solapamiento entre conjuntos"
            seen += ids[2]
        if folds == list(range(n)):
            ok = sorted(seen) == sorted(c.case_id for c in cases)
            print(f"cada caso esta en test exactamente una vez: {ok}")
            return 0 if ok else 1
        return 0

    for k in folds:
        run = out_root / f"{cfg['experiment']}_fold{k}"
        if (run / "done.json").exists():
            print(f"fold {k}: ya terminado, se omite", flush=True)
            continue
        cmd, cwd = fold_command(args.config, k, extra)
        print(f"fold {k}: {' '.join(cmd[1:])}", flush=True)
        rc = subprocess.call(cmd, cwd=cwd)
        if rc != 0:
            print(f"fold {k} termino con codigo {rc}; vuelve a ejecutar este mismo comando para continuar", flush=True)
            return rc

    summary = summarise(cfg, list(range(n)), out_root)
    path = out_root / f"{cfg['experiment']}_cv_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary.get(k) for k in ("folds_done", "mean_over_folds", "std_over_folds", "pooled_test_cases")},
                     indent=2))
    print(f"resumen en {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
