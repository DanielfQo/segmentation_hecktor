"""Paired comparison of two evaluations of the same cases (per_case_*.csv written by evaluate).

    python -m mambalite3d.tools.compare_runs --a runs/full_base_fold0:best --b runs/exp_wide:final --split test

Each side is RUN:TAG and reads RUN/per_case_<split>_<TAG>.csv (TAG can be best, final, best_tta, ens_..., etc.).
For every metric it reports A, B, the difference B - A, a 95% paired bootstrap interval and a two-sided paired
permutation p-value (the two results of each case are swapped at random). For the per-patient GTVp Dice it also
reports the Wilcoxon signed-rank test. Metrics: HECKTOR 2025 (GTVp mean per patient, GTVn aggregated DSC, GTVn F1,
score) and, when the CSV files have GTVp volumes, HECKTOR 2022 (GTVp aggregated DSC, mean of both aggregated DSC).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from ..config import resolve

FIELDS = {"p": "dice_gtvp", "ni": "gtvn_inter_mm3", "ns": "gtvn_sum_mm3", "tp": "gtvn_tp", "fp": "gtvn_fp",
          "fn": "gtvn_fn", "pi": "gtvp_inter_mm3", "ps": "gtvp_sum_mm3"}


def read_side(spec: str, split: str):
    run, _, tag = spec.rpartition(":")
    if not run:
        raise SystemExit(f"usa RUN:TAG, por ejemplo runs/exp_wide:final (recibido {spec!r})")
    run_path = resolve(run) if not Path(run).is_absolute() else Path(run)
    path = run_path / f"per_case_{split}_{tag}.csv"
    if not path.exists():
        raise SystemExit(f"no existe {path}")
    with open(path, encoding="utf-8") as f:
        rows = {r["case_id"]: r for r in csv.DictReader(f)}
    return f"{run_path.name}:{tag}", rows


def to_arrays(rows: dict, ids) -> dict:
    out = {}
    for key, col in FIELDS.items():
        out[key] = np.array([float(rows[i].get(col, "nan") or "nan") for i in ids])
    return out


def _agg(inter, total):
    return np.where(total > 0, 2 * inter / np.where(total > 0, total, 1), 1.0)


def metrics(arr: dict, idx=None) -> dict:
    """Metrics as arrays of shape (R,): for one set of cases (1-D arrays, idx None), for R resamples of it
    (idx: (R, n) case indices), or for R ready-made rows (2-D arrays of shape (R, n))."""

    def take(a):
        if a.ndim == 2:
            return a
        return a[None, :] if idx is None else a[idx]

    gtvp = take(arr["p"]).mean(axis=1)
    gtvn = _agg(take(arr["ni"]).sum(axis=1), take(arr["ns"]).sum(axis=1))
    tp, fp, fn = (take(arr[k]).sum(axis=1) for k in ("tp", "fp", "fn"))
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 1.0)
    out = {"gtvp_mean": gtvp, "gtvn_agg": gtvn, "f1_ganglios": f1, "score_2025": 0.5 * (gtvp + gtvn)}
    if np.all(np.isfinite(arr["pi"])) and np.all(np.isfinite(arr["ps"])):
        gtvp_agg = _agg(take(arr["pi"]).sum(axis=1), take(arr["ps"]).sum(axis=1))
        out["gtvp_agg"] = gtvp_agg
        out["dsc_agg_mean_2022"] = 0.5 * (gtvp_agg + gtvn)
    return out


def compare(a: dict, b: dict, n_boot: int = 10000, n_perm: int = 10000, seed: int = 0) -> dict:
    n = len(a["p"])
    rng = np.random.default_rng(seed)
    ma, mb = metrics(a), metrics(b)
    keys = [k for k in ma if k in mb]
    idx = rng.integers(0, n, size=(n_boot, n))
    ba, bb = metrics(a, idx), metrics(b, idx)
    swap = rng.random((n_perm, n)) < 0.5
    pa = {k: np.where(swap, b[k], a[k]) for k in a}
    pb = {k: np.where(swap, a[k], b[k]) for k in a}
    qa, qb = metrics(pa), metrics(pb)
    res = {}
    for k in keys:
        d = float(mb[k][0] - ma[k][0])
        lo, hi = np.percentile(bb[k] - ba[k], [2.5, 97.5])
        perm = qb[k] - qa[k]
        p = (np.sum(np.abs(perm) >= abs(d) - 1e-12) + 1) / (n_perm + 1)
        res[k] = {"a": float(ma[k][0]), "b": float(mb[k][0]), "diff": d, "ci95": [float(lo), float(hi)], "p_perm": float(p)}
    diff_p = b["p"] - a["p"]
    if np.any(np.abs(diff_p) > 1e-12):
        w = wilcoxon(b["p"], a["p"], zero_method="pratt")
        res["gtvp_wilcoxon"] = {"p": float(w.pvalue), "b_better": int(np.sum(diff_p > 1e-9)), "b_worse": int(np.sum(diff_p < -1e-9))}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="RUN:TAG, por ejemplo runs/full_base_fold0:best")
    ap.add_argument("--b", required=True, help="RUN:TAG, por ejemplo runs/exp_wide:final")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--out", default=None, help="JSON de salida (por defecto runs/comparaciones/<a>_vs_<b>_<split>.json)")
    args = ap.parse_args(argv)

    name_a, rows_a = read_side(args.a, args.split)
    name_b, rows_b = read_side(args.b, args.split)
    ids = sorted(set(rows_a) & set(rows_b))
    only = sorted(set(rows_a) ^ set(rows_b))
    if not ids:
        raise SystemExit("no hay casos en comun")
    if only:
        print(f"aviso: {len(only)} casos estan solo en uno de los dos archivos y se ignoran: {only[:5]}...")
    res = compare(to_arrays(rows_a, ids), to_arrays(rows_b, ids), args.n_boot, args.n_perm)
    print(f"A = {name_a}   B = {name_b}   ({args.split}, {len(ids)} casos pareados)")
    print(f"{'metrica':18s} {'A':>7s} {'B':>7s} {'B-A':>8s}   IC95% bootstrap      p (permutacion)")
    for k, v in res.items():
        if k == "gtvp_wilcoxon":
            continue
        lo, hi = v["ci95"]
        print(f"{k:18s} {v['a']:7.4f} {v['b']:7.4f} {v['diff']:+8.4f}   [{lo:+.4f}, {hi:+.4f}]   {v['p_perm']:.3f}")
    if "gtvp_wilcoxon" in res:
        w = res["gtvp_wilcoxon"]
        print(f"GTVp por paciente, Wilcoxon p = {w['p']:.3f} (B mejor en {w['b_better']} casos, peor en {w['b_worse']})")
    out = Path(args.out) if args.out else resolve("runs") / "comparaciones" / (
        f"{name_a.replace(':', '-')}_vs_{name_b.replace(':', '-')}_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"a": name_a, "b": name_b, "split": args.split, "n_cases": len(ids), "results": res},
                              indent=2), encoding="utf-8")
    print(f"guardado {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
