"""Training entry point.

    python -m mambalite3d.training.train --config configs/smoke_subset.yaml
    python -m mambalite3d.training.train --config configs/smoke_subset.yaml --overfit 2 --epochs 6

Data are read from the original HECKTOR files, cropped to the ROI and kept in RAM; patches are
augmented on the GPU on the fly. Only logs, metrics and checkpoints are written to disk.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..config import load_config, resolve
from ..data.augment import augment_batch, context_size
from ..data.io import find_cases, label_coverage, restore_to_original
from ..data.sampler import CaseStore, PatchSampler, iter_cases
from ..data.splits import pick_stratified, select_subset, split_cases
from ..eval.components import postprocess_prediction
from ..eval.hecktor import hecktor2022, hecktor2025
from ..eval.metrics import aggregate, evaluate_case
from ..models.network import build_model, model_overrides
from ..models.plan import check_patch, feature_map_sizes, plan_network
from ..models.ssm import HAS_MAMBA_SSM
from .inference import sliding_window_predict
from .loss import DeepSupervisionLoss, DiceCELoss


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str = "") -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        for _ in range(5):
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                break
            except OSError:
                time.sleep(0.5)


_CRASH_AT = os.environ.get("MAMBALITE_TEST_CRASH", "")  # only for the crash/resume tests


def maybe_crash(point: str, log=None) -> None:
    """Simulates an abrupt termination (like a power cut or a killed process) at a named point."""
    if _CRASH_AT and _CRASH_AT == point:
        if log is not None:
            log(f"[prueba] terminacion abrupta simulada en {point}")
        os._exit(99)


def save_json(obj, path: Path) -> None:
    """Atomic JSON write: a crash never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except OSError:  # OneDrive/antivirus may hold the file for a moment
            time.sleep(0.5 + attempt)
    raise OSError(f"could not write {path}")


def load_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def safe_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(8):
        try:
            torch.save(obj, tmp)
            maybe_crash("save")
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(1.0 + attempt)
    raise OSError(f"could not save {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def gb(x: float) -> float:
    return round(x / 1024**3, 3)


def save_qc_png(case, pred, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    lab = case.label
    zs = np.nonzero(lab.reshape(lab.shape[0], -1).any(axis=1))[0] if lab is not None else []
    z = int(zs[len(zs) // 2]) if len(zs) else lab.shape[0] // 2
    ct = case.image[0, z].astype(np.float32)
    pet = case.image[1, z].astype(np.float32)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(ct, cmap="gray")
    axes[0].set_title(f"{case.case_id} CT z={z}")
    axes[1].imshow(pet, cmap="hot")
    axes[1].set_title("PET (normalizado)")
    for ax, arr, title in ((axes[2], lab[z], "referencia"), (axes[3], pred[z], "prediccion")):
        ax.imshow(ct, cmap="gray")
        for cls, color in ((1, "red"), (2, "cyan")):
            if (arr == cls).any():
                ax.contour(arr == cls, levels=[0.5], colors=color, linewidths=1.2)
        ax.set_title(f"{title} (rojo GTVp, cian GTVn)")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def hecktor_score(metrics: dict) -> float:
    """Model-selection score (HECKTOR 2025 style): mean of GTVp mean DSC and GTVn aggregated DSC."""
    return 0.5 * (float(metrics["gtvp_dice_mean"]) + float(metrics["gtvn_dice_agg"]))


def hecktor2022_score(metrics: dict) -> float:
    """HECKTOR 2022 ranking metric: mean of the GTVp and GTVn aggregated DSC (NaN if GTVp volumes are missing)."""
    return float(metrics.get("dsc_agg_mean", float("nan")))


def evaluate(model, cases, cfg, patch, device, out_dir: Path, tag: str, log, original_space: bool = True, qc: int = 2,
             per_center: bool = False, max_cases: Optional[int] = None, postprocess: Optional[dict] = None):
    k = cfg["train"]["num_classes"]
    res_roi, res_orig = [], []
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, case in enumerate(cases):
        if max_cases is not None and i >= max_cases:
            break
        t_case = time.time()
        probs = sliding_window_predict(
            model,
            case.image.astype(np.float32),
            patch,
            k,
            device,
            step_fraction=cfg["infer"]["step_fraction"],
            amp=cfg["train"]["amp"],
            mirror=cfg["infer"]["mirror"],
            pad_values=case.pad_values,
            batch_size=cfg["infer"].get("batch_size", 4),
        )
        pred = probs.argmax(0).astype(np.uint8)
        vox_roi = float(np.prod(case.spacing_xyz))
        if postprocess:
            pred = postprocess_prediction(pred, probs, postprocess, vox_roi)
        res_roi.append(evaluate_case(case.case_id, pred, case.label, vox_roi))
        if original_space:
            pred_o, gt_o, vox_o = restore_to_original(case, pred)
            res_orig.append(evaluate_case(case.case_id, pred_o, gt_o, vox_o))
        if i < qc:
            save_qc_png(case, pred, out_dir / f"qc_{tag}_{case.case_id}.png")
        maybe_crash(f"eval:{tag}", log)
        r = res_orig[-1] if original_space else res_roi[-1]
        log(
            f"  {tag} {case.case_id} ROI {list(case.image.shape[1:])} DSC GTVp {r.dice_gtvp:.3f} GTVn {r.dice_gtvn_case:.3f} "
            f"TP/FP/FN nodos {r.gtvn_tp}/{r.gtvn_fp}/{r.gtvn_fn}  {time.time() - t_case:.1f} s"
        )
    agg = {"roi_space": aggregate(res_roi), "seconds": round(time.time() - t0, 1)}
    if original_space:
        agg["original_space"] = aggregate(res_orig)
    if per_center:
        res = res_orig if original_space else res_roi
        centers = sorted({r.case_id.split("-")[0] for r in res})
        agg["per_center"] = {c: aggregate([r for r in res if r.case_id.split("-")[0] == c]) for c in centers}
    main_space = agg["original_space"] if original_space else agg["roi_space"]
    agg["score"] = hecktor_score(main_space)
    agg["score_2022"] = hecktor2022_score(main_space)
    agg["hecktor2025"] = hecktor2025(main_space)  # official metrics of each edition, named as in the challenge
    agg["hecktor2022"] = hecktor2022(main_space)
    if original_space:
        with open(out_dir / f"per_case_{tag}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(res_orig[0]).keys()))
            writer.writeheader()
            for r in res_orig:
                writer.writerow(asdict(r))
    save_json(agg, out_dir / f"metrics_{tag}.json")
    log(f"evaluacion [{tag}] {json.dumps(agg)}")
    h25, h22 = agg["hecktor2025"], agg["hecktor2022"]
    log(f"resumen [{tag}] {main_space['n_cases']} casos | HECKTOR 2025: GTVp DSC {h25['GTVp_DSC']:.4f}, "
        f"GTVn DSCagg {h25['GTVn_DSCagg']:.4f}, GTVn F1agg {h25['GTVn_F1agg']:.4f} | HECKTOR 2022: "
        f"GTVp DSCagg {h22['GTVp_DSCagg']:.4f}, GTVn DSCagg {h22['GTVn_DSCagg']:.4f}, media {h22['DSCagg_media']:.4f}")
    if device.type == "cuda":
        torch.cuda.empty_cache()  # give back the inference buffers before training continues
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--patch", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--overfit", type=int, default=0, help="train and evaluate on the first N training cases, no augmentation")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoint_latest.pth if it exists (safe to use always)")
    ap.add_argument("--fold", type=int, default=None, help="cross-validation fold (needs data.split.n_folds)")
    ap.add_argument("--eval-max-cases", type=int, default=None, help="evaluate only the first N val/test cases (quick tests)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.iters is not None:
        cfg["train"]["iters_per_epoch"] = args.iters
    if args.variant is not None:
        cfg["model"]["variant"] = args.variant
    if args.patch is not None:
        cfg["model"]["patch_size_zyx"] = args.patch
    if args.batch is not None:
        cfg["train"]["batch_size"] = args.batch
    if args.no_aug or args.overfit:
        cfg["augment"]["enabled"] = False
    if args.fold is not None:
        cfg["data"]["split"]["fold"] = args.fold
    fold = cfg["data"]["split"].get("fold")
    if cfg["data"]["split"].get("n_folds") and fold is None:
        fold = cfg["data"]["split"]["fold"] = 0
    name = args.name or cfg["experiment"]
    if fold is not None and cfg["data"]["split"].get("n_folds") and not name.endswith(f"_fold{fold}"):
        name = f"{name}_fold{fold}"
    out_dir = resolve(cfg["output"]["root"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "log.txt")
    set_seed(cfg["seed"])
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.backends.cudnn.benchmark = True

    log("=" * 70)
    log(f"experimento {name}  variante {cfg['model']['variant']}  device {device}  mamba_ssm={HAS_MAMBA_SSM}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        log(f"GPU {torch.cuda.get_device_name(0)}  libre {gb(free)} GB de {gb(total)} GB")

    # ---------------- data ----------------
    data_cfg = cfg["data"]
    root = resolve(data_cfg["root"])
    all_cases = find_cases(root)
    subset = select_subset(all_cases, data_cfg["cases_per_center"], data_cfg["centers"], cfg["seed"])
    train_files, val_files, test_files = split_cases(subset, data_cfg["split"], cfg["seed"])
    if data_cfg["split"].get("n_folds"):
        log(f"validacion cruzada: fold {data_cfg['split']['fold']} de {data_cfg['split']['n_folds']} ({data_cfg['split']['mode']})")
    if args.overfit:
        train_files = train_files[: args.overfit]
        val_files = list(train_files)
        test_files = []
        cfg["train"]["fg_fraction"] = 0.5
    log(
        f"casos encontrados {len(all_cases)}; subconjunto {len(subset)}; split {data_cfg['split']['mode']}: "
        f"train {len(train_files)}; val {len(val_files)}; test {len(test_files)}"
    )
    for tag, files in (("train", train_files), ("val", val_files), ("test", test_files)):
        per_center = {}
        for c in files:
            per_center[c.center] = per_center.get(c.center, 0) + 1
        log(f"  {tag} por centro {dict(sorted(per_center.items()))}")
    split_now = {"mode": data_cfg["split"]["mode"], "fold": data_cfg["split"].get("fold"),
                 "train": [c.case_id for c in train_files], "val": [c.case_id for c in val_files],
                 "test": [c.case_id for c in test_files]}
    split_prev = load_json(out_dir / "split.json", None) if args.resume else None
    if split_prev is not None and any(split_prev.get(k) != split_now[k] for k in ("train", "val", "test")):
        log("ERROR: el split actual no coincide con split.json de este experimento; no se reanuda para no mezclar "
            "casos de test con entrenamiento. Usa el mismo config/semilla/fold o cambia --name.")
        return 4
    save_json(split_now, out_dir / "split.json")
    save_json(cfg, out_dir / "config_used.json")

    t0 = time.time()
    store = CaseStore(
        train_files, data_cfg, max_cases=data_cfg["max_cached_cases"], threads=data_cfg["load_threads"], seed=cfg["seed"], log=log
    )
    store.preload()
    ram = sum(c.image.nbytes + (c.label.nbytes if c.label is not None else 0) for c in store.cases())
    shapes = [c.image.shape[1:] for c in store.cases()]
    log(
        f"train en RAM: {len(store.cases())} casos, {gb(ram)} GB, carga {time.time() - t0:.1f} s "
        f"({np.mean(store.load_seconds):.1f} s/caso); ROI zyx min {np.min(shapes, 0).tolist()} max {np.max(shapes, 0).tolist()}"
    )
    cov = [label_coverage(c) for c in store.cases()]
    log(f"cobertura de la etiqueta dentro del ROI: min {min(cov):.4f}, media {np.mean(cov):.4f}")

    # ---------------- model ----------------
    patch = [int(p) for p in cfg["model"]["patch_size_zyx"]]
    spacing_zyx = list(reversed([float(s) for s in data_cfg["target_spacing_xyz"]]))
    strides, kernels = plan_network(patch, spacing_zyx, cfg["model"]["max_stages"], cfg["model"]["min_feature_size"])
    check_patch(patch, strides)
    overrides = model_overrides(cfg["model"])
    model = build_model(
        cfg["model"]["variant"], strides, kernels, in_channels=2, num_classes=cfg["train"]["num_classes"], **overrides
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    sizes = feature_map_sizes(patch, strides)
    log(f"patch zyx {patch}  strides {strides}  kernels {kernels}")
    log(f"mapas por nivel {sizes}  canales {model.feats}  niveles Mamba {model.mamba_levels}  "
        f"niveles con atencion global {model.lgfm_levels}  niveles CGA {model.cga_levels}")
    if model.cfg.dual_encoder:
        log(f"doble codificador, fusion '{model.cfg.dual_fusion}'")
    log(f"parametros {n_params / 1e6:.3f} M")

    loss_fn = DeepSupervisionLoss(DiceCELoss())
    tr = cfg["train"]
    opt = torch.optim.SGD(model.parameters(), lr=tr["lr"], momentum=tr["momentum"], nesterov=True, weight_decay=tr["weight_decay"])
    use_amp = bool(tr["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    history = []
    best = {"score": -1.0, "epoch": None}
    ckpt_latest = out_dir / "checkpoint_latest.pth"
    if args.resume and ckpt_latest.exists():
        try:
            ck = torch.load(ckpt_latest, map_location=device, weights_only=False)
        except Exception as e:  # noqa: BLE001
            log(f"ERROR: no se pudo leer {ckpt_latest.name} ({e!r}); borralo para empezar de cero")
            return 5
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best = load_json(out_dir / "best.json", best)
        history = [h for h in load_json(out_dir / "history.json", []) if h["epoch"] < start_epoch]
        log(f"reanudando desde la epoca {start_epoch} (de {tr['epochs']})")
    elif args.resume:
        log("no hay checkpoint_latest.pth: se empieza desde la epoca 0")

    ext = context_size(patch, cfg["augment"]["context_margin"])
    sampler = PatchSampler(store, ext, fg_fraction=tr["fg_fraction"], seed=cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    val_store = None

    # ---------------- training ----------------
    epochs = tr["epochs"]
    for epoch in range(start_epoch, epochs):
        lr = tr["lr"] * (1 - epoch / epochs) ** 0.9
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        losses, t_ep = [], time.time()
        t_data = t_aug = 0.0
        for it in range(tr["iters_per_epoch"]):
            maybe_crash(f"train:{epoch}:{it}", log)
            t1 = time.time()
            imgs, labs = sampler.sample(tr["batch_size"])
            x = torch.from_numpy(imgs).to(device, non_blocking=True)
            y = torch.from_numpy(labs).to(device, non_blocking=True)
            t2 = time.time()
            x, y = augment_batch(x, y, patch, spacing_zyx, cfg["augment"], gen, enabled=cfg["augment"]["enabled"])
            t3 = time.time()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = loss_fn(out, y)
            if not torch.isfinite(loss):
                log(f"ERROR: perdida no finita en la epoca {epoch} iteracion {it}")
                return 3
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            t_data += t2 - t1
            t_aug += t3 - t2
        dt = time.time() - t_ep
        rec = {
            "epoch": epoch,
            "lr": round(lr, 6),
            "loss": round(float(np.mean(losses)), 4),
            "seconds": round(dt, 1),
            "s_per_iter": round(dt / tr["iters_per_epoch"], 3),
            "data_s": round(t_data, 1),
            "aug_s": round(t_aug, 1),
            "eta_h": round(dt * (epochs - epoch - 1) / 3600, 2),
        }
        if device.type == "cuda":
            rec["peak_alloc_gb"] = gb(torch.cuda.max_memory_allocated())
            rec["peak_reserved_gb"] = gb(torch.cuda.max_memory_reserved())
        if store.max_cases < len(store.all_cases):
            rec["cache_new_cases"] = store.refresh(data_cfg["refresh_per_epoch"])
        history.append(rec)
        log(f"epoca {json.dumps(rec)}")
        save_json(history, out_dir / "history.json")
        if (epoch + 1) % tr["save_every"] == 0 or epoch == epochs - 1:
            safe_save(
                {"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                 "config": cfg, "strides": strides, "kernels": kernels},
                ckpt_latest,
            )
        if tr["val_every"] and (epoch + 1) % tr["val_every"] == 0 and epoch != epochs - 1 and val_files:
            subset_val = pick_stratified(val_files, tr["val_cases_during_training"], cfg["seed"])
            if len(subset_val) > 30:  # many cases: stream them instead of keeping them all in RAM
                val_cases = iter_cases(subset_val, data_cfg, threads=data_cfg["load_threads"])
            else:
                if val_store is None:
                    val_store = CaseStore(subset_val, data_cfg, threads=data_cfg["load_threads"], log=log)
                    val_store.preload()
                val_cases = val_store.cases()
            agg = evaluate(model, val_cases, cfg, patch, device, out_dir, f"ep{epoch + 1}", log, original_space=False, qc=0)
            score = hecktor_score(agg["roi_space"])
            if score > best["score"]:
                best = {"score": score, "epoch": epoch + 1}
                safe_save({"model": model.state_dict(), "config": cfg, "strides": strides, "kernels": kernels, "epoch": epoch},
                          out_dir / "checkpoint_best.pth")
                save_json(best, out_dir / "best.json")
                log(f"nuevo mejor modelo en validacion: score {score:.4f} (epoca {epoch + 1})")

    safe_save({"model": model.state_dict(), "config": cfg, "strides": strides, "kernels": kernels}, out_dir / "checkpoint_final.pth")

    # ---------------- evaluation ----------------
    # final weights on the whole validation set and on the test set (never used during training)
    if args.overfit:
        evaluate(model, store.cases(), cfg, patch, device, out_dir, "final", log)
    else:
        store.close()
        store.loaded.clear()  # free the training cache; evaluation streams cases (bounded RAM)
        if val_store is not None:
            val_store.close()
            val_store = None
        for tag, files in (("val", val_files), ("test", test_files)):
            if files:
                evaluate(model, iter_cases(files, data_cfg, threads=data_cfg["load_threads"]), cfg, patch, device, out_dir,
                         f"{tag}_final", log, per_center=True, max_cases=args.eval_max_cases)
    save_json({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "epochs": epochs, "fold": fold}, out_dir / "done.json")
    store.close()
    log("listo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
