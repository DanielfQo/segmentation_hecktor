"""Evaluates a trained run (or an ensemble of runs) on its validation or test split (split.json of the run).

    python -m mambalite3d.tools.evaluate --run runs/full_base_fold0 --split test --checkpoint best
    python -m mambalite3d.tools.evaluate --run runs/exp_wide --split val --checkpoint final --mirror
    python -m mambalite3d.tools.evaluate --ensemble runs/full_base_fold0:best runs/exp_wide:final --name base_wide --split val

Writes metrics_<split>_<checkpoint>.json, per_case_<split>_<checkpoint>.csv and QC images into the run folder
(tag suffixes: _tta with --mirror, _pp with --postprocess). An ensemble averages the softmax probabilities of its
members; its results go to runs/ens_<name>/ with tag <split>_ens[_tta]. All members must share the same split and
the same preprocessing and patch size (the same input is fed to every member).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from ..config import resolve
from ..data.io import find_cases
from ..data.sampler import iter_cases
from ..models.ensemble import EnsembleModel
from ..models.network import build_model, model_overrides
from ..training.train import Logger, evaluate, save_json

# data settings that change what the network sees; the rest (cache size, threads, split) do not
_INPUT_KEYS = ("root", "target_spacing_xyz", "ct_window", "pet_norm", "suv_clip", "roi")


def _run_path(run: str) -> Path:
    return resolve(run) if not Path(run).is_absolute() else Path(run)


def load_member(run: Path, checkpoint: str, device):
    ck_path = run / f"checkpoint_{checkpoint}.pth"
    if not ck_path.exists():
        raise SystemExit(f"no existe {ck_path}")
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["model"]["variant"], ck["strides"], ck["kernels"], in_channels=2,
                        num_classes=cfg["train"]["num_classes"], **model_overrides(cfg["model"])).to(device)
    model.load_state_dict(ck["model"])
    return model, cfg


def _read_split(run: Path) -> dict:
    with open(run / "split.json", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run folder, e.g. runs/full_base_fold0")
    ap.add_argument("--checkpoint", default="final", choices=["final", "best", "latest"])
    ap.add_argument("--ensemble", nargs="+", metavar="RUN:CHECKPOINT",
                    help="evaluate the average of several runs, e.g. runs/full_base_fold0:best runs/exp_wide:final")
    ap.add_argument("--name", default=None, help="ensemble name (output folder runs/ens_<name>)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--mirror", action="store_true", help="test-time augmentation with flips (slower)")
    ap.add_argument("--postprocess", default=None,
                    help="JSON with post-processing rules (or the postprocess_*.json written by tune_postprocess)")
    ap.add_argument("--max-cases", type=int, default=None,
                    help="quick check on the first N cases; the tag gets the suffix _primerosN")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if bool(args.run) == bool(args.ensemble):
        ap.error("usa --run o --ensemble (uno de los dos)")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.ensemble:
        members, cfgs, splits = [], [], []
        for spec in args.ensemble:
            run_s, _, ckpt = spec.rpartition(":")
            if not run_s or ckpt not in ("final", "best", "latest"):
                ap.error(f"miembro invalido {spec!r}: usa RUN:final, RUN:best o RUN:latest")
            run_m = _run_path(run_s)
            model, cfg_m = load_member(run_m, ckpt, device)
            members.append(model)
            cfgs.append(cfg_m)
            splits.append(_read_split(run_m))
        for i in range(1, len(cfgs)):
            if any(splits[i].get(k) != splits[0].get(k) for k in ("train", "val", "test")):
                raise SystemExit(f"{args.ensemble[i]} tiene otro split: no se puede ensamblar sin mezclar casos de test")
            diff = [k for k in _INPUT_KEYS if cfgs[i]["data"].get(k) != cfgs[0]["data"].get(k)]
            if diff or cfgs[i]["model"]["patch_size_zyx"] != cfgs[0]["model"]["patch_size_zyx"]:
                raise SystemExit(f"{args.ensemble[i]} usa otro preprocesamiento o patch ({diff or 'patch_size_zyx'})")
        name = args.name or "_".join(f"{Path(s.rpartition(':')[0]).name}-{s.rpartition(':')[2]}" for s in args.ensemble)
        run = resolve(cfgs[0]["output"]["root"]) / f"ens_{name}"
        run.mkdir(parents=True, exist_ok=True)
        save_json(splits[0], run / "split.json")
        save_json({"members": args.ensemble}, run / "ensemble.json")
        model, cfg, ids_all = EnsembleModel(members).to(device), cfgs[0], splits[0]
        tag_base, what = f"{args.split}_ens", f"ensamble {' + '.join(args.ensemble)}"
    else:
        run = _run_path(args.run)
        model, cfg = load_member(run, args.checkpoint, device)
        ids_all = _read_split(run)
        tag_base, what = f"{args.split}_{args.checkpoint}", f"{run.name} checkpoint {args.checkpoint}"

    if args.mirror:
        cfg["infer"]["mirror"] = True
    ids = set(ids_all.get(args.split, []))
    if not ids:
        print(f"el split '{args.split}' esta vacio en {run / 'split.json'}")
        return 3
    files = [c for c in find_cases(resolve(cfg["data"]["root"])) if c.case_id in ids]
    if args.max_cases:
        files = files[: args.max_cases]
    log = Logger(run / "log.txt")
    rules = None
    if args.postprocess:
        rules = json.loads(Path(args.postprocess).read_text(encoding="utf-8"))
        rules = rules.get("rules_best_dice", rules)
    tag = tag_base + ("_tta" if args.mirror else "") + ("_pp" if rules else "")
    if args.max_cases:
        tag += f"_primeros{args.max_cases}"
    log(f"evaluando {what} en {args.split} ({len(files)} casos){' con TTA' if args.mirror else ''}")
    patch = [int(p) for p in cfg["model"]["patch_size_zyx"]]
    evaluate(model, iter_cases(files, cfg["data"], threads=cfg["data"]["load_threads"]), cfg, patch, device, run, tag, log,
             per_center=True, postprocess=rules)
    return 0


if __name__ == "__main__":
    sys.exit(main())
