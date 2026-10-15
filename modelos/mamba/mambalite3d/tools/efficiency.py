"""Efficiency table: parameters, operations per patch, peak GPU memory and time (training and inference).

    python -m mambalite3d.tools.efficiency --models runs/full_base_fold0:best runs/exp_wide:final runs/full_unet:best
    python -m mambalite3d.tools.efficiency --config configs/full.yaml --variants base wide unet

Operations are counted for one forward pass of one patch (batch 1, no deep-supervision heads):
- convolutions, linear layers, einsum/matmul: torch.utils.flop_counter.FlopCounterMode (2 FLOPs per multiply-add);
- Mamba selective scan (elementwise recurrence that FlopCounterMode does not see): 9*B*L*D*N multiply-adds
  (+B*D*L for the D skip, +B*D*L for the z gate), the formula of VMamba's flops_selective_scan_fn;
- attention (F.scaled_dot_product_attention): B*H*Lq*Lk*(dk + dv) multiply-adds (QK^T and AV).
Normalisations, activations and other elementwise operations are not counted (usual convention).
"GMACs" is what fvcore/thop and most papers report as "GFLOPs"; the "GFLOPs" column here is 2 x GMACs.
Times: training = one optimiser step (batch 2, AMP); inference = network time of the sliding-window prediction of
one ROI of the given size (no disk reading or resampling). Only valid on Windows/CPU: with the mamba_ssm CUDA
kernels (Linux) the scan is not intercepted and its operations would be missing.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import load_config, resolve
from ..models import ssm
from ..models.network import build_model, model_overrides
from ..models.plan import check_patch, plan_network
from ..training.inference import sliding_window_predict
from ..training.loss import DeepSupervisionLoss, DiceCELoss


@contextlib.contextmanager
def _analytic_ops(acc: dict):
    """Replaces the selective scan and SDPA by stubs that add their multiply-adds to acc and return zeros."""
    orig_scan, orig_sdpa = ssm.selective_scan, F.scaled_dot_product_attention

    def scan_stub(u, delta, A, B, C, D=None, z=None, **kwargs):
        b, d, length = u.shape
        n = A.shape[-1]
        acc["scan"] += 9 * b * length * d * n + (b * d * length if D is not None else 0) + (b * d * length if z is not None else 0)
        return torch.zeros_like(u)

    def sdpa_stub(q, k, v, *args, **kwargs):
        *batch, lq, dk = q.shape
        lk, dv = k.shape[-2], v.shape[-1]
        nb = math.prod(batch)
        acc["attn"] += nb * lq * lk * (dk + dv)
        return torch.zeros(*batch, lq, dv, dtype=v.dtype, device=v.device)

    ssm.selective_scan = scan_stub
    F.scaled_dot_product_attention = sdpa_stub
    try:
        yield
    finally:
        ssm.selective_scan = orig_scan
        F.scaled_dot_product_attention = orig_sdpa


def _set_deep_supervision(model, value):
    dec = getattr(model, "decoder", None)
    old = getattr(dec, "deep_supervision", None)
    if old is not None:
        dec.deep_supervision = value
    return old


def count_ops(model: torch.nn.Module, patch, device) -> dict:
    from torch.utils.flop_counter import FlopCounterMode

    acc = {"scan": 0, "attn": 0}
    old = _set_deep_supervision(model, False)
    model.eval()
    x = torch.zeros(1, 2, *patch, device=device)
    with torch.no_grad(), _analytic_ops(acc), FlopCounterMode(display=False) as fc:
        model(x)
    _set_deep_supervision(model, old)
    dense = fc.get_total_flops() / 2  # FlopCounterMode counts 2 per multiply-add
    total = dense + acc["scan"] + acc["attn"]
    return {"gmacs": total / 1e9, "gflops": 2 * total / 1e9, "gmacs_conv_lineal": dense / 1e9,
            "gmacs_barrido_mamba": acc["scan"] / 1e9, "gmacs_atencion": acc["attn"] / 1e9}


def train_step_stats(model, patch, batch, device, steps=6) -> dict:
    model.train()
    _set_deep_supervision(model, True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, nesterov=True)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = DeepSupervisionLoss(DiceCELoss())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        x = torch.randn(batch, 2, *patch, device=device)
        y = torch.randint(0, 3, (batch, 1, *patch), device=device)
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    del opt
    return {"vram_entrenamiento_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "s_por_iteracion": float(np.mean(times[2:]))}


def inference_stats(model, patch, roi, device, batch_size=4, mirror=False) -> dict:
    image = np.random.default_rng(0).normal(size=(2, *roi)).astype(np.float32)
    torch.cuda.empty_cache()
    sliding_window_predict(model, image[:, : patch[0], : patch[1], : patch[2]], patch, 3, device, batch_size=batch_size)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    sliding_window_predict(model, image, patch, 3, device, batch_size=batch_size, mirror=mirror)
    torch.cuda.synchronize()
    key = "_tta" if mirror else ""
    return {f"vram_inferencia{key}_gb": torch.cuda.max_memory_allocated() / 1024**3, f"s_por_caso{key}": time.time() - t0}


def _from_checkpoint(spec: str, device):
    run_s, _, ckpt = spec.rpartition(":")
    run = resolve(run_s) if not Path(run_s).is_absolute() else Path(run_s)
    ck = torch.load(run / f"checkpoint_{ckpt}.pth", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["model"]["variant"], ck["strides"], ck["kernels"], in_channels=2,
                        num_classes=cfg["train"]["num_classes"], **model_overrides(cfg["model"]))
    model.load_state_dict(ck["model"])
    return f"{run.name}:{ckpt}", cfg["model"]["variant"], model.to(device), [int(p) for p in cfg["model"]["patch_size_zyx"]]


def _from_variant(variant: str, cfg: dict, device):
    patch = [int(p) for p in cfg["model"]["patch_size_zyx"]]
    spacing_zyx = list(reversed([float(s) for s in cfg["data"]["target_spacing_xyz"]]))
    strides, kernels = plan_network(patch, spacing_zyx, cfg["model"]["max_stages"], cfg["model"]["min_feature_size"])
    check_patch(patch, strides)
    model = build_model(variant, strides, kernels, in_channels=2, num_classes=cfg["train"]["num_classes"],
                        **model_overrides(cfg["model"]))
    return variant, variant, model.to(device), patch


def profile_one(name, variant, model, patch, roi, device, batch, tta) -> dict:
    row = {"modelo": name, "variante": variant, "patch_zyx": patch,
           "parametros_M": sum(p.numel() for p in model.parameters()) / 1e6}
    row.update(count_ops(model, patch, device))
    if device.type == "cuda":
        row.update(train_step_stats(model, patch, batch, device))
        row.update(inference_stats(model, patch, roi, device))
        if tta:
            row.update(inference_stats(model, patch, roi, device, mirror=True))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], metavar="RUN:CHECKPOINT")
    ap.add_argument("--variants", nargs="*", default=[], help="variants built from --config (no checkpoint needed)")
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--roi", type=int, nargs=3, default=[104, 288, 448], metavar=("Z", "Y", "X"),
                    help="ROI size for the inference time (typical ROI of HECKTOR at 1x1x3 mm)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--tta", action="store_true", help="also time inference with flips")
    ap.add_argument("--out", default="runs/eficiencia.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not args.models and not args.variants:
        ap.error("indica --models y/o --variants")
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    cfg = load_config(args.config)
    rows = []
    jobs = [("ck", m) for m in args.models] + [("var", v) for v in args.variants]
    for kind, item in jobs:
        name, variant, model, patch = _from_checkpoint(item, device) if kind == "ck" else _from_variant(item, cfg, device)
        try:
            row = profile_one(name, variant, model, patch, args.roi, device, args.batch, args.tta)
        except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
            row = {"modelo": name, "variante": variant, "error": "OOM", "detalle": str(e)[:200]}
        print(json.dumps(row), flush=True)
        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"roi_zyx": args.roi, "batch_entrenamiento": args.batch, "filas": rows}, indent=2),
                   encoding="utf-8")
    cols = [("parametros_M", "Param. (M)", 2), ("gmacs", "GMACs", 1), ("gflops", "GFLOPs", 1),
            ("vram_entrenamiento_gb", "VRAM entr. (GB)", 2), ("s_por_iteracion", "s/it", 2),
            ("vram_inferencia_gb", "VRAM inf. (GB)", 2), ("s_por_caso", "s/caso", 1)]
    print("\n| Modelo | " + " | ".join(c[1] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for r in rows:
        vals = [f"{r[k]:.{d}f}" if isinstance(r.get(k), (int, float)) else "-" for k, _, d in cols]
        print(f"| {r['modelo']} | " + " | ".join(vals) + " |")
    print(f"\nguardado {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
