"""Runs every fold of a cross-validation and summarises the test metrics.

    python -m mambalite3d.tools.cross_validate --config configs/full_cv7.yaml
    python -m mambalite3d.tools.cross_validate --config configs/full_cv7.yaml --folds 0 1 2   # only some folds
    python -m mambalite3d.tools.cross_validate --config configs/full_cv7.yaml --dry-run       # only print the splits

Each fold is a normal training run in runs/<experiment>_fold<k>/ launched with --resume, so this command
can be repeated after a crash or a power cut: finished folds (done.json) are skipped and an
interrupted fold continues from its last checkpoint. Extra arguments after `--` go to train
(e.g. `-- --epochs 2 --iters 5`).

The summary (runs/<experiment>_cv_summary.json) gives per-fold metrics, mean and standard deviation over
folds, and the pooled metrics over all test cases (each case is tested exactly once over the folds).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..config import load_config, resolve
from ..data.io import find_cases
from ..data.splits import select_subset, split_cases
from ..eval.metrics import CaseResult, aggregate


def n_folds_of(cfg) -> int:
    split = cfg["data"]["split"]
    if not split.get("n_folds"):
        raise SystemExit("el config no define data.split.n_folds (no es una validacion cruzada)")
    if split.get("mode") == "loco":
        cases = find_cases(resolve(cfg["data"]["root"]))
        return len({c.center for c in cases})
    return int(split["n_folds"])


def _read_cases(csv_path: Path):
    out = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(
                CaseResult(
                    case_id=row["case_id"],
                    dice_gtvp=float(row["dice_gtvp"]),
                    gtvp_in_ref=row["gtvp_in_ref"] == "True",
                    gtvn_inter_mm3=float(row["gtvn_inter_mm3"]),
                    gtvn_sum_mm3=float(row["gtvn_sum_mm3"]),
                    gtvn_tp=int(row["gtvn_tp"]),
                    gtvn_fp=int(row["gtvn_fp"]),
                    gtvn_fn=int(row["gtvn_fn"]),
                    dice_gtvn_case=float(row["dice_gtvn_case"]),
                    gtvp_inter_mm3=float(row.get("gtvp_inter_mm3", "nan")),
                    gtvp_sum_mm3=float(row.get("gtvp_sum_mm3", "nan")),
                )
            )
    return out


def summarise(cfg, folds, out_root: Path) -> dict:
    exp = cfg["experiment"]
    per_fold, pooled = {}, []
    for k in folds:
        run = out_root / f"{exp}_fold{k}"
        m = run / "metrics_test_final.json"
        if not m.exists():
            continue
        with open(m, encoding="utf-8") as f:
            per_fold[k] = json.load(f)["original_space"]
        if (run / "per_case_test_final.csv").exists():
            pooled += _read_cases(run / "per_case_test_final.csv")
    keys = ["gtvp_dice_mean", "gtvn_dice_agg", "gtvn_f1_agg", "gtvp_dice_agg", "dsc_agg_mean"]
    summary = {"experiment": exp, "folds_done": sorted(per_fold), "per_fold": per_fold}
    if per_fold:
        nan = float("nan")
        summary["mean_over_folds"] = {k: float(np.nanmean([v.get(k, nan) for v in per_fold.values()])) for k in keys}
        summary["std_over_folds"] = {k: float(np.nanstd([v.get(k, nan) for v in per_fold.values()])) for k in keys}
    if pooled:
        summary["pooled_test_cases"] = aggregate(pooled)
    return summary


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

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
        cmd = [sys.executable, "-m", "mambalite3d.training.train", "--config", args.config, "--fold", str(k), "--resume", *extra]
        print(f"fold {k}: {' '.join(cmd[1:])}", flush=True)
        rc = subprocess.call(cmd, cwd=str(Path(__file__).resolve().parents[2]))
        if rc != 0:
            print(f"fold {k} termino con codigo {rc}; vuelve a ejecutar este mismo comando para continuar", flush=True)
            return rc

    summary = summarise(cfg, list(range(n)), out_root)
    path = out_root / f"{cfg['experiment']}_cv_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary.get(k) for k in ("folds_done", "mean_over_folds", "std_over_folds", "pooled_test_cases")}, indent=2))
    print(f"resumen en {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
