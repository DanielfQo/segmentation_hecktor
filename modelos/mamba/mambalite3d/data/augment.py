"""On-the-fly data augmentation on the GPU (nothing is precomputed or stored).

The sampler returns a context patch slightly larger than the training patch. Here, per sample:
- spatial: rotation (in-plane, plus small out-of-plane rotations for isotropic spacing) and
  scaling in physical millimetres, applied with one trilinear resampling (nearest for labels)
  that also crops the context patch to the training patch;
- mirroring along each axis;
- intensity: multiplicative brightness, contrast, gamma, Gaussian noise and Gaussian blur.
The probabilities follow nnU-Net's defaults.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


def _uniform(gen: torch.Generator, lo: float, hi: float) -> float:
    return lo + (hi - lo) * torch.rand(1, generator=gen).item()


def _coin(gen: torch.Generator, p: float) -> bool:
    return p > 0 and torch.rand(1, generator=gen).item() < p


def build_theta(batch: int, out_zyx, in_zyx, spacing_zyx, cfg: dict, gen: torch.Generator, enabled: bool) -> torch.Tensor:
    """Affine matrices (B, 3, 4) for F.affine_grid, which works in (x, y, z) = (W, H, D) order."""
    sz, sy, sx = spacing_zyx
    e_out = torch.tensor([out_zyx[2] * sx, out_zyx[1] * sy, out_zyx[0] * sz], dtype=torch.float32) / 2
    e_in = torch.tensor([in_zyx[2] * sx, in_zyx[1] * sy, in_zyx[0] * sz], dtype=torch.float32) / 2
    anisotropic = max(spacing_zyx) / min(spacing_zyx) > 2.0
    thetas = []
    for _ in range(batch):
        a = torch.eye(3)
        if enabled and _coin(gen, cfg["p_rotation"]):
            r = math.radians(cfg["rotation_deg"])
            a = _rot_z(_uniform(gen, -r, r)) @ a  # in-plane (axial) rotation
            if not anisotropic:
                r2 = math.radians(cfg.get("rotation_oop_deg", 15))
                a = _rot_x(_uniform(gen, -r2, r2)) @ _rot_y(_uniform(gen, -r2, r2)) @ a
        if enabled and _coin(gen, cfg["p_scale"]):
            a = a * _uniform(gen, *cfg["scale"])
        theta = torch.zeros(3, 4)
        theta[:, :3] = torch.diag(1.0 / e_in) @ a @ torch.diag(e_out)
        thetas.append(theta)
    return torch.stack(thetas)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable blur of one (1, 1, D, H, W) volume."""
    radius = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(t**2) / (2 * sigma**2))
    k = k / k.sum()
    x = F.conv3d(F.pad(x, (0, 0, 0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1, 1))
    x = F.conv3d(F.pad(x, (0, 0, radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1, 1))
    x = F.conv3d(F.pad(x, (radius, radius, 0, 0, 0, 0), mode="replicate"), k.view(1, 1, 1, 1, -1))
    return x


def _intensity(x: torch.Tensor, cfg: dict, gen: torch.Generator) -> torch.Tensor:
    """x: (C, D, H, W) float32, augmented per channel."""
    for c in range(x.shape[0]):
        v = x[c]
        if _coin(gen, cfg["p_brightness"]):
            v = v * _uniform(gen, *cfg["brightness"])
        if _coin(gen, cfg["p_contrast"]):
            mn, mx, mean = v.min(), v.max(), v.mean()
            v = ((v - mean) * _uniform(gen, *cfg["contrast"]) + mean).clamp(mn, mx)
        if _coin(gen, cfg["p_gamma"]):
            mn, mx = v.min(), v.max()
            rng = (mx - mn).clamp_min(1e-7)
            mean, std = v.mean(), v.std()
            v = ((v - mn) / rng).clamp(0, 1) ** _uniform(gen, *cfg["gamma"]) * rng + mn
            v = (v - v.mean()) / v.std().clamp_min(1e-7) * std + mean  # retain statistics
        if _coin(gen, cfg["p_noise"]):
            std = _uniform(gen, *cfg["noise_std"])
            v = v + torch.randn(v.shape, device=v.device, generator=None) * std
        if _coin(gen, cfg["p_blur"]):
            v = _gaussian_blur(v[None, None], _uniform(gen, *cfg["blur_sigma"]))[0, 0]
        x[c] = v
    return x


def augment_batch(
    img: torch.Tensor,
    lab: torch.Tensor,
    out_zyx: Sequence[int],
    spacing_zyx: Sequence[float],
    cfg: dict,
    gen: torch.Generator,
    enabled: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """img: (B, C, *ext) float, lab: (B, 1, *ext) on the GPU. Returns patches of size out_zyx.

    With enabled=False this is an exact centre crop (the context margin is even per axis).
    """
    b = img.shape[0]
    in_zyx = img.shape[2:]
    theta = build_theta(b, out_zyx, in_zyx, spacing_zyx, cfg, gen, enabled).to(img.device)
    grid = F.affine_grid(theta, (b, img.shape[1], *out_zyx), align_corners=False)
    img_o = F.grid_sample(img.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    lab_o = F.grid_sample(lab.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    if enabled:
        for i in range(b):
            dims = [2 + ax for ax in cfg["mirror_axes"] if _coin(gen, 0.5)]
            if dims:
                img_o[i : i + 1] = torch.flip(img_o[i : i + 1], dims)
                lab_o[i : i + 1] = torch.flip(lab_o[i : i + 1], dims)
            img_o[i] = _intensity(img_o[i], cfg, gen)
    return img_o, lab_o.round().long()


def context_size(patch_zyx: Sequence[int], margin: float) -> Tuple[int, int, int]:
    """Context patch = patch + an even number of voxels (~margin fraction) per axis."""
    return tuple(int(p + 2 * math.ceil(p * margin / 2)) for p in patch_zyx)
