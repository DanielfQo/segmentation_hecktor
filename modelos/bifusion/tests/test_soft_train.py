"""Etiquetas suaves en el bucle de entrenamiento (necesita torch: corre en el PC)."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bifusion3d import soft_train  # noqa: E402
from bifusion3d.data import preproc  # noqa: E402
from mambalite3d.config import load_config  # noqa: E402
from mambalite3d.data import augment as maug  # noqa: E402
from mambalite3d.data import sampler as msamp  # noqa: E402
from mambalite3d.training import loss as mloss  # noqa: E402
from test_audit import make_case  # noqa: E402


class _Store:
    def __init__(self, cases):
        self._cases = cases

    def cases(self):
        return self._cases


def test_mismos_parches_y_fracciones(tmp_path):
    cfg = preproc.with_options(load_config(None)["data"], etiquetas="suave")
    case = preproc.load_case_v2(make_case(tmp_path, "SYN-001"), cfg)
    ext = (16, 48, 48)
    a = msamp.PatchSampler(_Store([case]), ext, fg_fraction=0.5, seed=3)
    b = msamp.PatchSampler(_Store([case]), ext, fg_fraction=0.5, seed=3)
    imgs, labs = msamp.PatchSampler.sample(a, 4)
    imgs2, labs2 = soft_train.sample_soft(b, 4)
    assert np.array_equal(imgs, imgs2) and np.array_equal(labs[:, 0], labs2[:, 0])
    assert labs2.shape == (4, 3, *ext) and labs2[:, 1:].any()


def test_aumentado_sin_cambios_es_recorte_y_suma_uno():
    gen = torch.Generator().manual_seed(0)
    img = torch.randn(2, 2, 8, 16, 16)
    lab = torch.zeros(2, 3, 8, 16, 16, dtype=torch.uint8)
    lab[:, 2, 2:6, 4:12, 4:12] = 255
    lab[:, 1, 3:5, 6:10, 6:10] = 128
    aug_cfg = load_config(None)["augment"]
    x, y = soft_train.augment_soft(img, lab, (4, 8, 8), (3.0, 1.0, 1.0), aug_cfg, gen, enabled=False)
    assert y.shape == (2, 3, 4, 8, 8) and y.dtype == torch.float32
    assert torch.allclose(y.sum(1), torch.ones(2, 4, 8, 8), atol=1e-5) and (y >= 0).all()
    x0, _ = maug.augment_batch(img, lab[:, :1], (4, 8, 8), (3.0, 1.0, 1.0), aug_cfg, torch.Generator().manual_seed(0), enabled=False)
    assert torch.allclose(x, x0)
    yy = soft_train.augment_soft(img, lab, (4, 8, 8), (3.0, 1.0, 1.0), copy.deepcopy(aug_cfg), gen, enabled=True)[1]
    assert torch.allclose(yy.sum(1), torch.ones_like(yy[:, 0]), atol=1e-5)


def test_perdida_suave_con_objetivo_duro_es_la_de_siempre():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 8, 8)
    hard = torch.randint(0, 3, (2, 1, 4, 8, 8))
    onehot = torch.zeros(2, 3, 4, 8, 8).scatter_(1, hard, 1.0)
    base = mloss.DiceCELoss()(logits, hard)
    soft = soft_train.SoftDiceCELoss()(logits, onehot)
    assert torch.allclose(base, soft, atol=1e-5)
    assert torch.allclose(soft_train.SoftDiceCELoss()(logits, hard), base)
    outs = [logits, torch.randn(2, 3, 2, 4, 4), torch.randn(2, 3, 1, 2, 2)]
    ds = soft_train.SoftDeepSupervisionLoss(soft_train.SoftDiceCELoss())(outs, onehot)
    assert torch.isfinite(ds)
