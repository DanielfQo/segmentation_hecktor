import numpy as np
import pytest
import torch

from mambalite3d.training.inference import sliding_window_predict


class _Pointwise(torch.nn.Module):
    """1x1x1 conv: the sliding-window result must equal the whole-volume result exactly."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv3d(2, 3, 1)

    def forward(self, x):
        return self.conv(x)


@pytest.mark.parametrize("batch_size", [1, 3, 8])
@pytest.mark.parametrize("shape", [(20, 50, 45), (8, 20, 20)])
def test_sliding_window_matches_full_volume(batch_size, shape):
    model = _Pointwise()
    rng = np.random.default_rng(0)
    image = rng.normal(size=(2, *shape)).astype(np.float32)
    probs = sliding_window_predict(
        model, image, (10, 32, 32), 3, torch.device("cpu"), step_fraction=0.5, amp=False, batch_size=batch_size
    )
    with torch.no_grad():
        ref = torch.softmax(model(torch.from_numpy(image)[None]), dim=1)[0].numpy()
    assert probs.shape == (3, *shape)
    assert np.allclose(probs, ref, atol=1e-5)


class _Seeded(_Pointwise):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = torch.nn.Conv3d(2, 3, 1)


def test_ensemble_averages_member_probabilities():
    from mambalite3d.models.ensemble import EnsembleModel

    a, b = _Seeded(1), _Seeded(2)
    image = np.random.default_rng(2).normal(size=(2, 12, 40, 40)).astype(np.float32)
    dev = torch.device("cpu")
    pa = sliding_window_predict(a, image, (10, 32, 32), 3, dev, amp=False, batch_size=2)
    pb = sliding_window_predict(b, image, (10, 32, 32), 3, dev, amp=False, batch_size=2)
    pe = sliding_window_predict(EnsembleModel([a, b]), image, (10, 32, 32), 3, dev, amp=False, batch_size=2)
    assert np.allclose(pe, 0.5 * (pa + pb), atol=1e-5)
    # an ensemble of one model is that model
    p1 = sliding_window_predict(EnsembleModel([a]), image, (10, 32, 32), 3, dev, amp=False, batch_size=2)
    assert np.allclose(p1, pa, atol=1e-5)


def test_ensemble_disables_deep_supervision():
    from mambalite3d.models.ensemble import EnsembleModel

    class _DS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.deep_supervision = True
            self.conv = torch.nn.Conv3d(2, 3, 1)

        def forward(self, x):
            out = self.conv(x)
            return [out, out[:, :, ::2, ::2, ::2]] if self.decoder.deep_supervision else out

    m = _DS()
    ens = EnsembleModel([m])
    assert m.decoder.deep_supervision is False
    x = torch.randn(1, 2, 4, 8, 8)
    assert ens(x).shape == (1, 3, 4, 8, 8)


def test_sliding_window_mirror_keeps_pointwise_result():
    model = _Pointwise()
    image = np.random.default_rng(1).normal(size=(2, 12, 40, 40)).astype(np.float32)
    a = sliding_window_predict(model, image, (10, 32, 32), 3, torch.device("cpu"), amp=False, mirror=True, batch_size=2)
    b = sliding_window_predict(model, image, (10, 32, 32), 3, torch.device("cpu"), amp=False, mirror=False, batch_size=2)
    # a pointwise model is flip-equivariant, so mirroring only rescales the logits (x4)
    with torch.no_grad():
        ref = torch.softmax(4 * model(torch.from_numpy(image)[None]), dim=1)[0].numpy()
    assert np.allclose(a, ref, atol=1e-5) and not np.allclose(a, b)
