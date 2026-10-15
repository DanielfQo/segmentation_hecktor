"""Paso 2: codificador siames completo con Cross/Concat Mamba en los niveles profundos."""
import pytest
import torch
import torch.nn.functional as F

from mambalite3d.models.config import make_config
from mambalite3d.models.plan import feature_map_sizes, plan_network
from mambalite3d.models.ssm import GroupedMambaTorch, selective_scan_ref
from mambalite3d.training.loss import DeepSupervisionLoss, DiceCELoss

from bifusion3d.fusion import CrossGroupedMamba, CrossMambaFusion3D
from bifusion3d.network import build_model, count_parameters

SPACING_ZYX = [3.0, 1.0, 1.0]


def _reference(gm: GroupedMambaTorch, x: torch.Tensor, c_from):
    """Cada grupo calculado por separado con el scan secuencial de referencia, leyendo con la C de c_from[i]."""
    _, g, length, _ = x.shape
    di, n, r = gm.d_inner, gm.d_state, gm.dt_rank
    parts = []
    for i in range(g):
        xi, z = (x[:, i] @ gm.in_proj_w[i].T).chunk(2, dim=-1)
        sl = slice(i * di, (i + 1) * di)
        u = F.silu(F.conv1d(xi.transpose(1, 2), gm.conv_w[sl], gm.conv_b[sl], padding=gm.d_conv - 1, groups=di)[..., :length])
        dt, bm, cm = torch.split(u.transpose(1, 2) @ gm.x_proj_w[i].T, [r, n, n], dim=-1)
        parts.append((u, (dt @ gm.dt_proj_w[i].T).transpose(1, 2), bm.transpose(1, 2), cm.transpose(1, 2), z.transpose(1, 2)))
    outs = []
    for i in range(g):
        u, dt, bm, _, z = parts[i]
        cm = parts[c_from[i]][3]
        y = selective_scan_ref(u, dt, -torch.exp(gm.A_log[i]), bm, cm, gm.D[i], z, gm.dt_proj_b[i], True)
        outs.append(y.transpose(1, 2) @ gm.out_proj_w[i].T)
    return torch.stack(outs, dim=1)


def test_cross_grouped_mamba_without_swap_is_grouped_mamba():
    torch.manual_seed(0)
    gm = GroupedMambaTorch(4, 12, chunk_size=8).double()
    cm = CrossGroupedMamba(4, 12, chunk_size=8).double()
    cm.load_state_dict(gm.state_dict())
    x = torch.randn(2, 4, 37, 12, dtype=torch.float64)
    assert torch.allclose(cm(x), gm(x), atol=1e-12, rtol=1e-10)
    assert torch.allclose(cm(x, c_from=[0, 1, 2, 3]), gm(x), atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("length,chunk,max_elems", [(33, 8, 3e7), (40, 4, 2 * 24 * 40 * 4 * 16 * 2)])
def test_cross_grouped_mamba_matches_sequential_reference(length, chunk, max_elems):
    """Intercambio de C contra el scan secuencial en float64 (tambien con el scan por tramos de grupos)."""
    torch.manual_seed(1)
    cm = CrossGroupedMamba(4, 12, chunk_size=chunk, max_scan_elems=max_elems).double()
    x = torch.randn(2, 4, length, 12, dtype=torch.float64, requires_grad=True)
    swap = [2, 3, 0, 1]
    y = cm(x, c_from=swap)
    y_ref = _reference(cm, x, swap)
    assert torch.allclose(y, y_ref, atol=1e-10, rtol=1e-8)
    gx = torch.autograd.grad(y.square().sum(), x)[0]
    gx_ref = torch.autograd.grad(y_ref.square().sum(), x)[0]
    assert torch.allclose(gx, gx_ref, atol=1e-9, rtol=1e-7)


def test_swapping_c_makes_each_modality_read_the_other():
    """Sin intercambio la salida del CT no depende del PET; con intercambio si."""
    torch.manual_seed(2)
    cm = CrossGroupedMamba(2, 8, chunk_size=8).double()
    x = torch.randn(1, 2, 20, 8, dtype=torch.float64)
    x2 = x.clone()
    x2[:, 1] += torch.randn(1, 20, 8, dtype=torch.float64)  # cambia solo el grupo PET
    assert torch.allclose(cm(x)[:, 0], cm(x2)[:, 0])
    assert not torch.allclose(cm(x, c_from=[1, 0])[:, 0], cm(x2, c_from=[1, 0])[:, 0])


def test_concat_block_mixes_in_both_directions():
    torch.manual_seed(3)
    cfg = make_config("base", chunk_size=8)
    fusion = CrossMambaFusion3D(16, cfg).double()
    ct, pet = torch.randn(1, 16, 2, 3, 4, dtype=torch.float64), torch.randn(1, 16, 2, 3, 4, dtype=torch.float64)
    o_ct, o_pet = fusion.concat_mix(ct, pet)
    o_ct2, _ = fusion.concat_mix(ct, pet + 1.0)  # cambia el PET: cambia la salida del CT (sentido invertido)
    _, o_pet2 = fusion.concat_mix(ct + 1.0, pet)  # cambia el CT: cambia la salida del PET (sentido directo)
    assert o_ct.shape == ct.shape and o_pet.shape == pet.shape
    assert not torch.allclose(o_ct, o_ct2) and not torch.allclose(o_pet, o_pet2)


def test_cross_mamba_fusion_shapes_and_gradients():
    cfg = make_config("base", chunk_size=8)
    fusion = CrossMambaFusion3D(16, cfg)
    ct = torch.randn(2, 16, 2, 4, 4, requires_grad=True)
    pet = torch.randn(2, 16, 2, 4, 4, requires_grad=True)
    y = fusion(ct, pet)
    assert y.shape == (2, 16, 2, 4, 4)
    y.square().mean().backward()
    assert ct.grad.abs().sum() > 0 and pet.grad.abs().sum() > 0
    for name in ("cross.in_proj_w", "cross.x_proj_w", "concat.in_proj_w", "scale_ct", "scale_pet", "proj.weight"):
        assert dict(fusion.named_parameters())[name].grad is not None, name


def test_siam_full_structure_parameters_and_outputs():
    strides, kernels = plan_network([32, 128, 128], SPACING_ZYX)
    base = build_model("base", strides, kernels)
    full = build_model("siam_full", strides, kernels)
    gate = build_model("siam_gate", strides, kernels)
    assert full.n_siamese == len(strides) == 6
    assert full.fusion_ops == ["gate"] * 4 + ["xmamba"] * 2  # Mamba en los niveles 4 y 5
    assert gate.fusion_ops == ["gate"] * 6
    p_base, p_full, p_gate = (count_parameters(m) for m in (base, full, gate))
    print(f"\nparametros base {p_base / 1e6:.3f} M, siam_full {p_full / 1e6:.3f} M, siam_gate {p_gate / 1e6:.3f} M")
    assert p_full < 1.25 * p_base  # el codificador comparte pesos; solo se anaden los modulos de fusion

    patch = [32, 64, 64]
    strides, kernels = plan_network(patch, SPACING_ZYX)
    model = build_model("siam_full", strides, kernels, chunk_size=16)
    assert model.fusion_ops == ["gate"] * 3 + ["xmamba"] * 2
    outs = model(torch.randn(1, 2, *patch))
    for o, s in zip(outs, feature_map_sizes(patch, strides)):
        assert list(o.shape) == [1, 3, *s]


@pytest.mark.parametrize("variant", ["siam_full", "siam_gate"])
def test_full_siamese_variants_train_one_step_on_cpu(variant):
    patch = [16, 32, 32]
    strides, kernels = plan_network(patch, SPACING_ZYX, min_size=2)
    model = build_model(variant, strides, kernels, chunk_size=16)
    loss = DeepSupervisionLoss(DiceCELoss())(model(torch.randn(2, 2, *patch)), torch.randint(0, 3, (2, 1, *patch)))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.fusion.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="necesita CUDA")
@pytest.mark.parametrize("variant", ["siam_full", "siam_gate"])
def test_full_siamese_gpu_amp_training_step(variant):
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
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
