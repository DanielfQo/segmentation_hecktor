"""Paso 4: BiFusion CNN <-> Mamba (TransFuse) dentro de cada MambaStage."""
import pytest
import torch

from mambalite3d.models.config import make_config
from mambalite3d.models.modules import MambaStage
from mambalite3d.models.plan import feature_map_sizes, plan_network
from mambalite3d.training.loss import DeepSupervisionLoss, DiceCELoss

from bifusion3d.fusion import SE3D, BiFusionStage, SpatialAttention3D
from bifusion3d.network import build_model, count_parameters

SPACING_ZYX = [3.0, 1.0, 1.0]


def test_attention_blocks_keep_shape_and_only_rescale():
    x = torch.rand(2, 16, 3, 4, 4) + 0.5  # positiva, para comparar el cociente salida / entrada
    for block in (SE3D(16), SpatialAttention3D(3)):
        y = block(x)
        assert y.shape == x.shape
        ratio = y / x
        assert ratio.min() >= 0 and ratio.max() <= 1 + 1e-6  # multiplica por una compuerta en [0, 1]


def test_bifusion_stage_shapes_and_gradients_reach_both_branches():
    cfg = make_config("base", chunk_size=8)
    stage = BiFusionStage(8, 16, cfg)
    x = torch.randn(2, 8, 4, 4, 4, requires_grad=True)
    y = stage(x)
    assert y.shape == (2, 16, 4, 4, 4)
    y.square().mean().backward()
    assert x.grad.abs().sum() > 0
    grads = {name: p.grad for name, p in stage.named_parameters()}
    for prefix in ("local.", "amf.", "lgfm.", "gsc.", "se.", "spatial.", "w_t.", "w_g.", "interact.", "fuse."):
        assert any(g is not None and g.abs().sum() > 0 for n, g in grads.items() if n.startswith(prefix)), prefix


@pytest.mark.parametrize("variant", ["base_bifusion", "siam_full_bifusion"])
def test_every_mamba_stage_is_replaced(variant):
    strides, kernels = plan_network([32, 128, 128], SPACING_ZYX)
    base = build_model("base", strides, kernels)
    model = build_model(variant, strides, kernels)
    n_mamba = sum(isinstance(m, MambaStage) for m in model.modules())
    n_bifusion = sum(isinstance(m, BiFusionStage) for m in model.modules())
    assert n_mamba == 0 and n_bifusion == 2 * len(model.mamba_levels) + 1  # codificador + decoder + cuello
    p_base, p_model = count_parameters(base), count_parameters(model)
    print(f"\nparametros base {p_base / 1e6:.3f} M, {variant} {p_model / 1e6:.3f} M")


def test_base_bifusion_keeps_early_fusion_and_outputs():
    patch = [32, 64, 64]
    strides, kernels = plan_network(patch, SPACING_ZYX)
    model = build_model("base_bifusion", strides, kernels, chunk_size=16)
    assert model.n_siamese == 0 and model.encoder.stages[0][0][0].in_channels == 2
    outs = model(torch.randn(1, 2, *patch))
    for o, s in zip(outs, feature_map_sizes(patch, strides)):
        assert list(o.shape) == [1, 3, *s]


@pytest.mark.parametrize("variant", ["base_bifusion", "siam_full_bifusion"])
def test_bifusion_variants_train_one_step_on_cpu(variant):
    patch = [16, 32, 32]
    strides, kernels = plan_network(patch, SPACING_ZYX, min_size=2)
    model = build_model(variant, strides, kernels, chunk_size=16)
    loss = DeepSupervisionLoss(DiceCELoss())(model(torch.randn(2, 2, *patch)), torch.randint(0, 3, (2, 1, *patch)))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="necesita CUDA")
@pytest.mark.parametrize("variant", ["base_bifusion", "siam_full_bifusion"])
def test_bifusion_gpu_amp_training_step(variant):
    patch = [32, 128, 128]
    strides, kernels = plan_network(patch, SPACING_ZYX)
    model = build_model(variant, strides, kernels).cuda()
    x = torch.randn(2, 2, *patch, device="cuda")
    y = torch.randint(0, 3, (2, 1, *patch), device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = DeepSupervisionLoss(DiceCELoss())(model(x), y)
    loss.backward()
    torch.cuda.synchronize()
    print(f"\n{variant}, patch {patch}, batch 2: pico {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    assert torch.isfinite(loss)
