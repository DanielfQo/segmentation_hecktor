"""Entrenamiento con etiquetas suaves (data.etiquetas: suave), sobre el mismo bucle de mambalite3d.

`install()` sustituye, solo en memoria:
- PatchSampler.sample: ademas de la etiqueta dura, extrae las fracciones de GTVp y GTVn (case.soft) del mismo parche.
  Devuelve la etiqueta como (B, 3, ...) uint8: canal 0 la dura, canales 1-2 las fracciones (0-255). Usa el mismo
  generador y en el mismo orden, asi que los parches elegidos son los mismos que sin etiquetas suaves.
- augment_batch: la misma transformacion espacial; las fracciones se interpolan con bilineal (no con vecino mas
  cercano) y el objetivo sale como probabilidades (B, 3, ...) float: fondo, GTVp, GTVn.
- DiceCELoss y DeepSupervisionLoss: con un objetivo de probabilidades, entropia cruzada suave y Dice suave; las
  salidas de baja resolucion reciben el objetivo promediado por area. Con un objetivo duro, exactamente lo de antes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mambalite3d.data import augment as maug
from mambalite3d.data import sampler as msamp
from mambalite3d.training import loss as mloss

from .data.preproc import soft_patch

_orig_augment = maug.augment_batch


def sample_soft(self, batch_size: int):
    cases = self.store.cases()
    n_fg = int(round(batch_size * self.fg_fraction)) if self.fg_fraction > 0 else 0
    imgs = np.empty((batch_size, 2, *self.ext), dtype=np.float16)
    labs = np.zeros((batch_size, 3, *self.ext), dtype=np.uint8)
    for b in range(batch_size):
        case = cases[self.rng.integers(len(cases))]
        shape = case.image.shape[1:]
        force_fg = b >= batch_size - n_fg
        classes = [c for c in (1, 2) if c in case.fg_coords]
        if force_fg and classes:
            coords = case.fg_coords[classes[self.rng.integers(len(classes))]]
            center = coords[self.rng.integers(len(coords))]
        else:
            center = [self.rng.integers(0, s) for s in shape]
        imgs[b] = msamp._extract(case.image, center, self.ext, case.pad_values)
        if case.label is not None:
            labs[b, 0] = msamp._extract(case.label, center, self.ext, 0)
            labs[b, 1:] = soft_patch(case, center, self.ext)
    return imgs, labs


def augment_soft(img, lab, out_zyx, spacing_zyx, cfg, gen, enabled=True):
    """Como augment_batch; si `lab` trae fracciones (3 canales), devuelve probabilidades (B, 3, ...) float."""
    if lab.shape[1] == 1:
        return _orig_augment(img, lab, out_zyx, spacing_zyx, cfg, gen, enabled)
    b = img.shape[0]
    in_zyx = img.shape[2:]
    theta = maug.build_theta(b, out_zyx, in_zyx, spacing_zyx, cfg, gen, enabled).to(img.device)
    grid = F.affine_grid(theta, (b, img.shape[1], *out_zyx), align_corners=False)
    img_o = F.grid_sample(img.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    frac = F.grid_sample(lab[:, 1:].float() / 255.0, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if enabled:
        for i in range(b):
            dims = [2 + ax for ax in cfg["mirror_axes"] if maug._coin(gen, 0.5)]
            if dims:
                img_o[i: i + 1] = torch.flip(img_o[i: i + 1], dims)
                frac[i: i + 1] = torch.flip(frac[i: i + 1], dims)
            img_o[i] = maug._intensity(img_o[i], cfg, gen)
    frac = frac.clamp(0, 1)
    s = frac.sum(1, keepdim=True)
    frac = torch.where(s > 1, frac / s.clamp_min(1e-6), frac)
    return img_o, torch.cat([1 - frac.sum(1, keepdim=True), frac], dim=1)


class SoftDiceCELoss(mloss.DiceCELoss):
    def forward(self, logits, target):
        if not target.is_floating_point():
            return super().forward(logits, target)
        logits = logits.float()
        t = target.float()
        ce = -(t * F.log_softmax(logits, dim=1)).sum(1).mean()
        probs = torch.softmax(logits, dim=1)
        axes = [0] + list(range(2, probs.ndim)) if self.batch_dice else list(range(2, probs.ndim))
        tp = (probs * t).sum(axes)
        denom = probs.sum(axes) + t.sum(axes)
        dc = (2 * tp + self.smooth) / torch.clamp(denom + self.smooth, min=1e-8)
        if not self.do_bg:
            dc = dc[1:] if self.batch_dice else dc[:, 1:]
        return ce - dc.mean()


class SoftDeepSupervisionLoss(mloss.DeepSupervisionLoss):
    def forward(self, outputs, target):
        if not target.is_floating_point():
            return super().forward(outputs, target)
        if isinstance(outputs, torch.Tensor):
            return self.base(outputs, target)
        weights = mloss.ds_weights(len(outputs))
        total = outputs[0].new_zeros((), dtype=torch.float32)
        for w, out in zip(weights, outputs):
            if w == 0:
                continue
            t = target if out.shape[2:] == target.shape[2:] else F.interpolate(target, size=out.shape[2:], mode="area")
            total = total + w * self.base(out, t)
        return total


def install() -> None:
    import mambalite3d.training.train as mtrain

    msamp.PatchSampler.sample = sample_soft
    mtrain.augment_batch = augment_soft
    mtrain.DiceCELoss = SoftDiceCELoss
    mtrain.DeepSupervisionLoss = SoftDeepSupervisionLoss
