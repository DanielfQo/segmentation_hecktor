"""Paso 1: codificador siames en los niveles 0-1 con compuerta de fusion."""
import pytest
import torch

from mambalite3d.models.config import make_config
from mambalite3d.models.network import MambaLite3D
from mambalite3d.models.plan import feature_map_sizes, plan_network
from mambalite3d.training.loss import DeepSupervisionLoss, DiceCELoss

from bifusion3d.fusion import GatedFusion3D
from bifusion3d.network import BiFusionNet, FusionSpec, build_model, count_parameters

SPACING_ZYX = [3.0, 1.0, 1.0]  # HECKTOR a 1 x 1 x 3 mm


def _plan(patch=(32, 64, 64)):
    return plan_network(list(patch), SPACING_ZYX)


def test_gated_fusion_shapes_and_gradients_reach_both_modalities():
    fusion = GatedFusion3D(8, kernel=(1, 3, 3))
    ct = torch.randn(2, 8, 4, 6, 6, requires_grad=True)
    pet = torch.randn(2, 8, 4, 6, 6, requires_grad=True)
    y = fusion(ct, pet)
    assert y.shape == (2, 8, 4, 6, 6)
    y.square().mean().backward()
    assert ct.grad.abs().sum() > 0 and pet.grad.abs().sum() > 0


def test_gated_fusion_gates_are_bounded_and_per_modality():
    fusion = GatedFusion3D(4)
    ct, pet = torch.randn(1, 4, 3, 5, 5), torch.randn(1, 4, 3, 5, 5)
    g_ct, g_pet = fusion.gates(ct, pet)
    assert g_ct.shape == g_pet.shape == (1, 4, 3, 5, 5)
    for g in (g_ct, g_pet):
        assert g.min() >= 0 and g.max() <= 1
    assert not torch.allclose(g_ct, g_pet)


def test_without_fusion_the_network_is_the_base_model():
    """Con FusionSpec() la red acepta los pesos de MambaLite3D y da exactamente su salida."""
    strides, kernels = _plan()
    cfg = make_config("base", chunk_size=16)
    torch.manual_seed(0)
    base = MambaLite3D(2, 3, strides, kernels, cfg).eval()
    net = BiFusionNet(2, 3, strides, kernels, cfg, FusionSpec()).eval()
    net.load_state_dict(base.state_dict(), strict=True)
    x = torch.randn(1, 2, 32, 64, 64)
    with torch.no_grad():
        for a, b in zip(base(x), net(x)):
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_shared_stem_processes_each_modality_on_its_own():
    """El truco (B, 2, ...) -> (2B, 1, ...): la salida del CT no depende del PET y es la de pasar el CT solo."""
    strides, kernels = _plan()
    net = build_model("siam_light", strides, kernels, chunk_size=16).eval()
    x = torch.randn(2, 2, 32, 64, 64)
    stage0 = net.encoder.stages[0]
    with torch.no_grad():
        both = stage0(x.reshape(4, 1, 32, 64, 64))
        both = both.reshape(2, 2, *both.shape[1:])
        ct_alone = stage0(x[:, :1])
        pet_alone = stage0(x[:, 1:])
    assert torch.allclose(both[:, 0], ct_alone, atol=1e-5)
    assert torch.allclose(both[:, 1], pet_alone, atol=1e-5)


def test_siam_light_structure_and_parameter_count():
    strides, kernels = plan_network([32, 128, 128], SPACING_ZYX)
    base = build_model("base", strides, kernels)
    net = build_model("siam_light", strides, kernels)
    assert net.n_siamese == 2 and net.fusion_ops == ["gate", "gate"]
    assert net.encoder.stages[0][0][0].in_channels == 1  # stem compartido de 1 canal
    p_base, p_net = count_parameters(base), count_parameters(net)
    print(f"\nparametros base {p_base / 1e6:.4f} M, siam_light {p_net / 1e6:.4f} M ({100 * (p_net / p_base - 1):+.2f} %)")
    assert abs(p_net - p_base) / p_base < 0.01  # solo se anaden las compuertas


def test_siam_light_outputs_and_deep_supervision():
    patch = [32, 64, 64]
    strides, kernels = _plan(patch)
    model = build_model("siam_light", strides, kernels, chunk_size=16)
    x = torch.randn(1, 2, *patch)
    outs = model(x)
    sizes = feature_map_sizes(patch, strides)
    assert isinstance(outs, list) and len(outs) == len(strides) - 1
    for o, s in zip(outs, sizes):
        assert list(o.shape) == [1, 3, *s]
    model.decoder.deep_supervision = False
    out = model(x)
    assert isinstance(out, torch.Tensor) and list(out.shape) == [1, 3, *patch]


def test_siam_light_trains_one_step_on_cpu():
    patch = [16, 32, 32]
    strides, kernels = plan_network(patch, SPACING_ZYX, min_size=2)
    model = build_model("siam_light", strides, kernels, chunk_size=16)
    loss_fn = DeepSupervisionLoss(DiceCELoss())
    x = torch.randn(2, 2, *patch)
    y = torch.randint(0, 3, (2, 1, *patch))
    loss = loss_fn(model(x), y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is not None for p in model.fusion.parameters())


def test_base_variants_still_build_mambalite3d():
    strides, kernels = _plan()
    for variant in ("base", "wide", "unet", "samescan"):
        assert type(build_model(variant, strides, kernels, chunk_size=16)) is MambaLite3D


@pytest.mark.skipif(not torch.cuda.is_available(), reason="necesita CUDA")
def test_siam_light_gpu_amp_training_step():
    patch = [32, 128, 128]
    strides, kernels = plan_network(patch, SPACING_ZYX)
    model = build_model("siam_light", strides, kernels).cuda()
    loss_fn = DeepSupervisionLoss(DiceCELoss())
    x = torch.randn(2, 2, *patch, device="cuda")
    y = torch.randint(0, 3, (2, 1, *patch), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = loss_fn(model(x), y)
    loss.backward()
    torch.cuda.synchronize()
    print(f"\nsiam_light, patch {patch}, batch 2: pico {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    assert torch.isfinite(loss)
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
