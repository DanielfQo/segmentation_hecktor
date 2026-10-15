"""nnU-Net-style planning of pooling strides and kernel sizes from spacing and patch size."""
from __future__ import annotations

from typing import List, Sequence, Tuple


def plan_network(
    patch_zyx: Sequence[int], spacing_zyx: Sequence[float], max_stages: int = 6, min_size: int = 4
) -> Tuple[List[List[int]], List[List[int]]]:
    """Returns (strides, kernels), one entry per stage, in (z, y, x) order.

    An axis is pooled when its current spacing is at most twice the smallest current spacing and
    the feature map along it stays >= min_size (and even). Kernels are 3 along axes whose spacing is
    at most twice the smallest spacing at that stage, else 1 (anisotropic first stages).
    """
    size = [int(s) for s in patch_zyx]
    spacing = [float(s) for s in spacing_zyx]
    strides: List[List[int]] = [[1, 1, 1]]
    stage_spacings = [list(spacing)]
    for _ in range(max_stages - 1):
        smallest = min(spacing)
        s = [
            2 if (spacing[i] <= 2 * smallest and size[i] % 2 == 0 and size[i] // 2 >= min_size) else 1
            for i in range(3)
        ]
        if max(s) == 1:
            break
        strides.append(s)
        size = [size[i] // s[i] for i in range(3)]
        spacing = [spacing[i] * s[i] for i in range(3)]
        stage_spacings.append(list(spacing))
    kernels = []
    for sp in stage_spacings:
        smallest = min(sp)
        kernels.append([3 if sp[i] <= 2 * smallest else 1 for i in range(3)])
    return strides, kernels


def feature_map_sizes(patch_zyx: Sequence[int], strides: Sequence[Sequence[int]]) -> List[List[int]]:
    sizes, cur = [], [int(s) for s in patch_zyx]
    for s in strides:
        cur = [cur[i] // s[i] for i in range(3)]
        sizes.append(list(cur))
    return sizes


def check_patch(patch_zyx: Sequence[int], strides: Sequence[Sequence[int]]) -> None:
    total = [1, 1, 1]
    for s in strides:
        total = [total[i] * s[i] for i in range(3)]
    for i in range(3):
        if patch_zyx[i] % total[i]:
            raise ValueError(f"patch {list(patch_zyx)} is not divisible by the total stride {total}")
