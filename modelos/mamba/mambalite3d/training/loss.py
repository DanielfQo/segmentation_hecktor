"""Dice + cross-entropy with deep supervision, as in nnU-Net."""
from __future__ import annotations

from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, batch_dice: bool = True, do_bg: bool = False):
        super().__init__()
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.do_bg = do_bg

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits (B, K, ...), target (B, 1, ...) long."""
        logits = logits.float()
        ce = F.cross_entropy(logits, target[:, 0])
        probs = torch.softmax(logits, dim=1)
        onehot = torch.zeros_like(probs).scatter_(1, target, 1.0)
        axes = [0] + list(range(2, probs.ndim)) if self.batch_dice else list(range(2, probs.ndim))
        tp = (probs * onehot).sum(axes)
        denom = probs.sum(axes) + onehot.sum(axes)
        dc = (2 * tp + self.smooth) / torch.clamp(denom + self.smooth, min=1e-8)
        if not self.do_bg:
            dc = dc[1:] if self.batch_dice else dc[:, 1:]
        return ce - dc.mean()


def ds_weights(n: int) -> List[float]:
    w = [1 / (2**i) for i in range(n)]
    if n > 1:
        w[-1] = 0.0
    s = sum(w)
    return [x / s for x in w]


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, outputs: Union[torch.Tensor, Sequence[torch.Tensor]], target: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self.base(outputs, target)
        weights = ds_weights(len(outputs))
        total = outputs[0].new_zeros((), dtype=torch.float32)
        for w, out in zip(weights, outputs):
            if w == 0:
                continue
            t = target if out.shape[2:] == target.shape[2:] else F.interpolate(target.float(), size=out.shape[2:], mode="nearest").long()
            total = total + w * self.base(out, t)
        return total
