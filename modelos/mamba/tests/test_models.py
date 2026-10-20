import pytest
import torch

from mambalite3d.models.blocks import FUE, GSC3D, ChannelLayerNorm
from mambalite3d.models.config import VARIANTS, make_config
from mambalite3d.models.modules import AMF3D, CGA3D, LGFM3D, MambaStage
from mambalite3d.models.network import build_model, stage_features
from mambalite3d.models.plan import check_patch, feature_map_sizes, plan_network
from mambalite3d.models.scans import ORDERS, flatten, unflatten


@pytest.mark.parametrize("order", ORDERS)
def test_scan_roundtrip(order):
    x = torch.randn(2, 6, 5, 3, 7)
    t = flatten(x, order)
    assert t.shape == (2, 5 * 3 * 7, 6)
    assert torch.equal(unflatten(t, order, x.shape[2:]), x)


def test_scan_orders_differ():
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 1, 2, 3, 4)
    seqs = {o: flatten(x, o)[0, :, 0].tolist() for o in ORDERS}
    assert len({tuple(s) for s in seqs.values()}) == 4
    assert seqs["slice"][:2] == [0.0, 12.0]  # D varies fastest
    assert seqs["ap"][:2] == [0.0, 4.0]  # H varies fastest


def test_plan_anisotropic_hecktor():
    strides, kernels = plan_network([48, 160, 160], [3.0, 1.0, 1.0])
    assert strides == [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    assert kernels[0] == [1, 3, 3] and kernels[1] == [3, 3, 3]
    check_patch([48, 160, 160], strides)
    assert feature_map_sizes([48, 160, 160], strides)[-1] == [6, 5, 5]


def test_plan_isotropic():
    strides, _ = plan_network([128, 128, 128], [1.0, 1.0, 1.0])
    assert strides == [[1, 1, 1]] + [[2, 2, 2]] * 5


def test_blocks_shapes():
    x = torch.randn(2, 8, 5, 6, 7)
    assert GSC3D(8)(x).shape == x.shape
    assert FUE()(x).shape == x.shape
    assert ChannelLayerNorm(8)(x).shape == x.shape


def test_modules_shapes_and_alpha_grad():
    cfg = make_config("base", chunk_size=16)
    x = torch.randn(2, 16, 4, 5, 6)
    amf = AMF3D(16, cfg)
    y = amf(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert amf.alpha.grad is not None and amf.alpha.grad.abs().item() > 0
    assert LGFM3D(16, 8)(x).shape == x.shape
    g = torch.randn_like(x)
    assert CGA3D(16, cfg)(x, g).shape == x.shape
    assert MambaStage(8, 16, cfg)(torch.randn(2, 8, 4, 5, 6)).shape == (2, 16, 4, 5, 6)


def test_stage_features():
    assert stage_features(6) == [16, 16, 32, 48, 64, 96]
    assert stage_features(5) == [16, 32, 48, 64, 96]
    assert stage_features(7) == [16, 16, 16, 32, 48, 64, 96]
    assert stage_features(5, [32, 64, 128, 256, 320, 320], keep="first") == [32, 64, 128, 256, 320]


@pytest.mark.parametrize(
    "patch,spacing",
    [([32, 64, 64], [3.0, 1.0, 1.0]), ([32, 32, 32], [1.0, 1.0, 1.0])],
)
def test_network_outputs(patch, spacing):
    strides, kernels = plan_network(patch, spacing)
    model = build_model("base", strides, kernels, chunk_size=32)
    x = torch.randn(1, 2, *patch)
    outs = model(x)
    sizes = feature_map_sizes(patch, strides)
    assert isinstance(outs, list) and len(outs) == len(strides) - 1
    for o, s in zip(outs, sizes):
        assert list(o.shape) == [1, 3, *s]
    model.decoder.deep_supervision = False
    out = model(x)
    assert isinstance(out, torch.Tensor) and list(out.shape) == [1, 3, *patch]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_all_variants_build_and_run(variant):
    patch = [16, 32, 32]
    strides, kernels = plan_network(patch, [3.0, 1.0, 1.0], min_size=2)
    model = build_model(variant, strides, kernels, chunk_size=32)
    outs = model(torch.randn(1, 2, *patch))
    assert len(outs) == len(strides) - 1
    assert all(torch.isfinite(o).all() for o in outs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_network_gpu_amp_backward():
    patch = [32, 64, 64]
    strides, kernels = plan_network(patch, [3.0, 1.0, 1.0])
    model = build_model("base", strides, kernels).cuda()
    x = torch.randn(2, 2, *patch, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        outs = model(x)
        loss = sum(o.float().square().mean() for o in outs)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_dual_encoder_streams_are_independent():
    patch = [16, 32, 32]
    strides, kernels = plan_network(patch, [3.0, 1.0, 1.0], min_size=2)
    model = build_model("dual", strides, kernels, chunk_size=32).eval()
    enc = model.encoder
    assert len(enc.fuse) == len(strides)
    x = torch.randn(1, 2, *patch)
    ct, pet = enc.ct(x[:, 0:1]), enc.pet(x[:, 1:2])
    x2 = x.clone()
    x2[:, 1] = torch.randn_like(x2[:, 1])  # changing PET must not change the CT stream
    assert all(torch.equal(a, b) for a, b in zip(ct, enc.ct(x2[:, 0:1])))
    assert not all(torch.equal(a, b) for a, b in zip(pet, enc.pet(x2[:, 1:2])))
    single = build_model("base", strides, kernels, chunk_size=32)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(model) > n(single)


def _encoder_skips_shapes(model, x):
    return [tuple(s.shape) for s in model.encoder(x)]


def test_dual_mid_fusion_shares_the_mamba_levels():
    patch = [16, 32, 32]
    strides, kernels = plan_network(patch, [3.0, 1.0, 1.0], min_size=2)
    mid = build_model("dualmid", strides, kernels, chunk_size=32).eval()
    enc, k = mid.encoder, min(mid.mamba_levels)
    assert len(enc.ct.stages) == len(enc.pet.stages) == len(enc.fuse) == k
    assert len(enc.shared.stages) == len(strides) - k
    assert not any(isinstance(m, MambaStage) for m in list(enc.ct.modules()) + list(enc.pet.modules()))
    x = torch.randn(1, 2, *patch)
    base = build_model("base", strides, kernels, chunk_size=32).eval()
    assert _encoder_skips_shapes(mid, x) == _encoder_skips_shapes(base, x)
    x2 = x.clone()
    x2[:, 1] = torch.randn_like(x2[:, 1])  # changing PET must not change the CT stream
    assert all(torch.equal(a, b) for a, b in zip(enc.ct(x[:, 0:1]), enc.ct(x2[:, 0:1])))
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(base) < n(mid) < n(build_model("dual", strides, kernels, chunk_size=32))


def test_global_attention_only_at_the_deepest_levels():
    patch = [16, 32, 32]
    strides, kernels = plan_network(patch, [3.0, 1.0, 1.0], min_size=2)
    model = build_model("m4lite", strides, kernels, chunk_size=32)
    n = len(strides)
    assert model.mamba_levels == list(range(n - 4, n)) and model.lgfm_levels == [n - 2, n - 1]
    for lvl in model.mamba_levels:
        for stage in (model.encoder.stages[lvl][1], model.decoder.block[lvl]):
            assert isinstance(stage, MambaStage)
            assert isinstance(stage.lgfm, LGFM3D) == (lvl >= n - 2)
    assert isinstance(model.bottleneck.lgfm, LGFM3D)
