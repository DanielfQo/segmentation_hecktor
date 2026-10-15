"""Paso 3: CGA3D como compuerta bidireccional PET <-> CT en los niveles CGA."""
import pytest
import torch

from mambalite3d.models.config import make_config
from mambalite3d.models.modules import CGA3D
from mambalite3d.models.plan import feature_map_sizes, plan_network
from mambalite3d.training.loss import DeepSupervisionLoss, DiceCELoss

from bifusion3d.fusion import CGAFusion3D
from bifusion3d.network import build_model, count_parameters

SPACING_ZYX = [3.0, 1.0, 1.0]


def test_gate_is_the_psi_map_of_cga3d():
    """Con los mismos pesos, CGA3D(x, g) == gate(x, g) * x: la compuerta es exactamente la de MambaLite3D."""
    torch.manual_seed(0)
    cfg = make_config("base", chunk_size=8)
    cga = CGA3D(16, cfg).double().eval()
    fusion = CGAFusion3D(16, cfg).double().eval()
    fusion.load_state_dict(cga.state_dict(), strict=True)
    x, g = torch.randn(2, 16, 3, 4, 5, dtype=torch.float64), torch.randn(2, 16, 3, 4, 5, dtype=torch.float64)
    with torch.no_grad():
        psi = fusion.gate(x, g)
        assert psi.shape == (2, 1, 3, 4, 5)
        assert torch.allclose(cga(x, g), psi * x, atol=1e-12)


def test_fusion_is_a_voxelwise_convex_combination():
    torch.manual_seed(1)
    cfg = make_config("base", chunk_size=8)
    fusion = CGAFusion3D(16, cfg).double().eval()
    ct, pet = torch.randn(1, 16, 2, 4, 4, dtype=torch.float64), torch.randn(1, 16, 2, 4, 4, dtype=torch.float64)
    with torch.no_grad():
        psi = fusion.gate(ct, pet)
        out = fusion(ct, pet)
        assert torch.allclose(out, psi * ct + (1 - psi) * pet)
        assert psi.min() >= 0 and psi.max() <= 1
        assert torch.allclose(fusion(ct, ct), ct)  # si ambas modalidades coinciden, devuelve esa misma entrada


def test_gradients_reach_both_modalities():
    cfg = make_config("base", chunk_size=8)
    fusion = CGAFusion3D(16, cfg)
    ct = torch.randn(2, 16, 2, 4, 4, requires_grad=True)
    pet = torch.randn(2, 16, 2, 4, 4, requires_grad=True)
    fusion(ct, pet).square().mean().backward()
    assert ct.grad.abs().sum() > 0 and pet.grad.abs().sum() > 0
    assert all(p.grad is not None for p in fusion.parameters())


def test_siam_cga_structure_parameters_and_outputs():
    strides, kernels = plan_network([32, 128, 128], SPACING_ZYX)
    base = build_model("base", strides, kernels)
    model = build_model("siam_cga", strides, kernels)
    assert model.fusion_ops == ["gate"] * 3 + ["cga"] * 3  # niveles CGA de MambaLite3D: 3, 4 y 5
    p_base, p_model = count_parameters(base), count_parameters(model)
    print(f"\nparametros base {p_base / 1e6:.3f} M, siam_cga {p_model / 1e6:.3f} M")
    assert p_model < 1.25 * p_base

    patch = [32, 64, 64]
    strides, kernels = plan_network(patch, SPACING_ZYX)
    model = build_model("siam_cga", strides, kernels, chunk_size=16)
    assert model.fusion_ops == ["gate"] * 2 + ["cga"] * 3
    outs = model(torch.randn(1, 2, *patch))
    for o, s in zip(outs, feature_map_sizes(patch, strides)):
        assert list(o.shape) == [1, 3, *s]


def test_siam_cga_trains_one_step_on_cpu():
    patch = [16, 32, 32]
    strides, kernels = plan_network(patch, SPACING_ZYX, min_size=2)
    model = build_model("siam_cga", strides, kernels, chunk_size=16)
    loss = DeepSupervisionLoss(DiceCELoss())(model(torch.randn(2, 2, *patch)), torch.randint(0, 3, (2, 1, *patch)))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.fusion.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="necesita CUDA")
def test_siam_cga_gpu_amp_training_step():
    patch = [32, 128, 128]
    strides, kernels = plan_network(patch, SPACING_ZYX)
    model = build_model("siam_cga", strides, kernels).cuda()
    x = torch.randn(2, 2, *patch, device="cuda")
    y = torch.randint(0, 3, (2, 1, *patch), device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = DeepSupervisionLoss(DiceCELoss())(model(x), y)
    loss.backward()
    torch.cuda.synchronize()
    print(f"\nsiam_cga, patch {patch}, batch 2: pico {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    assert torch.isfinite(loss)
