"""Sliding-window inference with Gaussian weighting (nnU-Net style)."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch


def _gaussian(patch_zyx: Sequence[int], device, sigma_scale: float = 1 / 8) -> torch.Tensor:
    grids = []
    for p in patch_zyx:
        x = torch.arange(p, device=device, dtype=torch.float32) - (p - 1) / 2
        sigma = p * sigma_scale
        grids.append(torch.exp(-(x**2) / (2 * sigma**2)))
    g = grids[0][:, None, None] * grids[1][None, :, None] * grids[2][None, None, :]
    g = g / g.max()
    return g.clamp_min(1e-4)


def _starts(size: int, patch: int, step_fraction: float):
    if size <= patch:
        return [0]
    n = int(math.ceil((size - patch) / (patch * step_fraction))) + 1
    return [int(round(i * (size - patch) / (n - 1))) for i in range(n)]


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_zyx: Sequence[int],
    num_classes: int,
    device: torch.device,
    step_fraction: float = 0.5,
    amp: bool = True,
    mirror: bool = False,
    pad_values=None,
    batch_size: int = 4,
) -> np.ndarray:
    """image: (C, Z, Y, X) numpy. Returns softmax probabilities (K, Z, Y, X) float32 numpy."""
    was_training = model.training
    model.eval()
    ds = getattr(getattr(model, "decoder", None), "deep_supervision", None)
    if ds is not None:
        model.decoder.deep_supervision = False
    c, *shape = image.shape
    padded = [max(s, p) for s, p in zip(shape, patch_zyx)]
    vol = np.empty((c, *padded), dtype=np.float32)
    for ch in range(c):
        vol[ch] = pad_values[ch] if pad_values is not None else float(image[ch].min())
    vol[:, : shape[0], : shape[1], : shape[2]] = image
    x_full = torch.from_numpy(vol).to(device)
    gauss = _gaussian(patch_zyx, device)
    acc = torch.zeros((num_classes, *padded), device=device, dtype=torch.float32)
    weight = torch.zeros(padded, device=device, dtype=torch.float32)
    pz, py, px = patch_zyx
    starts = [
        (z, y, x)
        for z in _starts(padded[0], pz, step_fraction)
        for y in _starts(padded[1], py, step_fraction)
        for x in _starts(padded[2], px, step_fraction)
    ]
    for i in range(0, len(starts), max(1, batch_size)):
        group = starts[i : i + max(1, batch_size)]
        patch = torch.stack([x_full[:, z : z + pz, y : y + py, x : x + px] for z, y, x in group])
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(patch)
            if mirror:
                for dims in ([2], [3], [4]):
                    logits = logits + torch.flip(model(torch.flip(patch, dims)), dims)
        probs = torch.softmax(logits.float(), dim=1)
        for (z, y, x), p_ in zip(group, probs):
            acc[:, z : z + pz, y : y + py, x : x + px] += p_ * gauss
            weight[z : z + pz, y : y + py, x : x + px] += gauss
    probs = (acc / weight.clamp_min(1e-8))[:, : shape[0], : shape[1], : shape[2]]
    if ds is not None:
        model.decoder.deep_supervision = ds
    model.train(was_training)
    return probs.cpu().numpy()
