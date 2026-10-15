"""Convolutional building blocks, SegMamba's GSC and FUE."""
from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn as nn

Kernel = Union[int, Sequence[int]]


def _pad(kernel: Kernel):
    if isinstance(kernel, int):
        return kernel // 2
    return [k // 2 for k in kernel]


def norm3d(channels: int) -> nn.Module:
    return nn.InstanceNorm3d(channels, eps=1e-5, affine=True)


class ConvIN(nn.Sequential):
    """Conv3d -> InstanceNorm -> LeakyReLU (nnU-Net convention)."""

    def __init__(self, cin: int, cout: int, kernel: Kernel = 3, stride: Kernel = 1):
        super().__init__(
            nn.Conv3d(cin, cout, kernel, stride=stride, padding=_pad(kernel)),
            norm3d(cout),
            nn.LeakyReLU(0.01, inplace=True),
        )


class BasicResBlock(nn.Module):
    def __init__(self, channels: int, kernel: Kernel = 3):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel, padding=_pad(kernel))
        self.norm1 = norm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel, padding=_pad(kernel))
        self.norm2 = norm3d(channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(y + x)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a (B, C, D, H, W) tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()


class ConvNormReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int):
        super().__init__(nn.Conv3d(cin, cout, kernel, padding=kernel // 2), norm3d(cout), nn.ReLU(inplace=True))


class GSC3D(nn.Module):
    """SegMamba eq. 2: GSC(z) = z + C3(C3(z) * C1(z))."""

    def __init__(self, channels: int):
        super().__init__()
        self.branch3 = ConvNormReLU(channels, channels, 3)
        self.branch1 = ConvNormReLU(channels, channels, 1)
        self.fuse = ConvNormReLU(channels, channels, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.fuse(self.branch3(z) * self.branch1(z))


class FUE(nn.Module):
    """SegMamba eq. 4 (feature-level uncertainty estimation), no parameters."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zf = z.float()
        zbar = torch.sigmoid(zf.mean(dim=1, keepdim=True))
        u = -zbar * torch.log(zbar + 1e-6)
        return (zf + zf * (1.0 - u)).to(z.dtype)
