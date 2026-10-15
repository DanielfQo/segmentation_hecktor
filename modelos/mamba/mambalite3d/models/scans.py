"""Scan orders that turn a 3D feature map into a token sequence and back.

Tensors are (B, C, D, H, W) with D, H, W the spatial axes in array order (z, y, x for HECKTOR).

    fwd   : (D, H, W) with W varying fastest
    rev   : fwd reversed along the sequence
    slice : (H, W, D) with D varying fastest (SegMamba's inter-slice scan)
    ap    : (D, W, H) with H varying fastest (antero-posterior axis)
"""
from __future__ import annotations

import torch

ORDERS = ("fwd", "rev", "slice", "ap")

_PERM = {
    "fwd": (0, 2, 3, 4, 1),
    "rev": (0, 2, 3, 4, 1),
    "slice": (0, 3, 4, 2, 1),
    "ap": (0, 2, 4, 3, 1),
}


def _inverse(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


_INV = {k: _inverse(v) for k, v in _PERM.items()}


def flatten(x: torch.Tensor, order: str) -> torch.Tensor:
    """(B, C, D, H, W) -> (B, L, C)."""
    if order not in _PERM:
        raise ValueError(f"unknown scan order {order}")
    b, c = x.shape[:2]
    t = x.permute(*_PERM[order]).reshape(b, -1, c)
    if order == "rev":
        t = t.flip(1)
    return t.contiguous()


def unflatten(t: torch.Tensor, order: str, spatial_shape) -> torch.Tensor:
    """(B, L, C) -> (B, C, D, H, W), inverse of `flatten`."""
    b, _, c = t.shape
    d, h, w = spatial_shape
    if order == "rev":
        t = t.flip(1)
    sizes = {2: d, 3: h, 4: w}
    perm = _PERM[order]
    permuted_shape = (b,) + tuple(sizes[p] for p in perm[1:4]) + (c,)
    return t.reshape(permuted_shape).permute(*_INV[order]).contiguous()
