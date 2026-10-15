"""Ensemble of trained networks for inference: averages their softmax probabilities.

`forward` returns log(mean_i softmax(model_i(x))), so the softmax applied by sliding_window_predict gives back
exactly the averaged probabilities. With flip test-time augmentation, sliding_window_predict sums these
log-probabilities over the flips (as it sums logits for a single model) before the softmax.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class EnsembleModel(nn.Module):
    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        if not models:
            raise ValueError("EnsembleModel needs at least one model")
        self.models = nn.ModuleList(models)
        for m in self.models:  # inference only: a single full-resolution output per model
            dec = getattr(m, "decoder", None)
            if dec is not None and hasattr(dec, "deep_supervision"):
                dec.deep_supervision = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = None
        for m in self.models:
            out = m(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            p = torch.softmax(out.float(), dim=1)
            probs = p if probs is None else probs + p
        return torch.log((probs / len(self.models)).clamp_min(1e-8))
