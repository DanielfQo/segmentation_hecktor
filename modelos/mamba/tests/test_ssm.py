import pytest
import torch

from mambalite3d.models.ssm import (
    GroupedMambaTorch,
    MambaTorch,
    TokenMLP,
    count_params,
    make_mixer,
    matched_mlp_hidden,
    mamba_forward_torch,
    selective_scan_chunked,
    selective_scan_parallel,
    selective_scan_ref,
)


def _inputs(b, d, n, l, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(b, d, l, generator=g, dtype=dtype)
    delta = torch.randn(b, d, l, generator=g, dtype=dtype) * 0.5
    a = -torch.rand(d, n, generator=g, dtype=dtype) * 2 - 0.1
    bm = torch.randn(b, n, l, generator=g, dtype=dtype)
    cm = torch.randn(b, n, l, generator=g, dtype=dtype)
    dd = torch.randn(d, generator=g, dtype=dtype)
    z = torch.randn(b, d, l, generator=g, dtype=dtype)
    bias = torch.randn(d, generator=g, dtype=dtype) * 0.1
    return u, delta, a, bm, cm, dd, z, bias


@pytest.mark.parametrize("l,chunk", [(1, 64), (7, 4), (64, 64), (100, 32), (130, 64)])
def test_chunked_scan_matches_reference(l, chunk):
    u, delta, a, bm, cm, dd, z, bias = _inputs(2, 5, 8, l)
    ref = selective_scan_ref(u, delta, a, bm, cm, dd, z, bias, True)
    out = selective_scan_chunked(u, delta, a, bm, cm, dd, z, bias, True, chunk_size=chunk)
    assert torch.allclose(ref, out, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("l,chunk", [(1, 16), (7, 4), (16, 16), (17, 16), (100, 8), (300, 4), (600, 16)])
def test_parallel_scan_matches_reference(l, chunk):
    u, delta, a, bm, cm, dd, z, bias = _inputs(2, 5, 8, l)
    ref = selective_scan_ref(u, delta, a, bm, cm, dd, z, bias, True)
    out = selective_scan_parallel(u, delta, a, bm, cm, dd, z, bias, True, chunk_size=chunk)
    assert torch.allclose(ref, out, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("ckpt", [1e12, 0])
def test_parallel_scan_gradients_match_reference(ckpt):
    u, delta, a, bm, cm, dd, z, bias = _inputs(1, 3, 4, 77)
    tensors = [t.clone().requires_grad_(True) for t in (u, delta, a, bm, cm, dd, z, bias)]
    ref = selective_scan_ref(*tensors[:7], tensors[7], True)
    grads_ref = torch.autograd.grad(ref.sum(), tensors)
    tensors2 = [t.detach().clone().requires_grad_(True) for t in tensors]
    out = selective_scan_parallel(*tensors2[:7], tensors2[7], True, chunk_size=4, checkpoint_elems=ckpt)
    grads = torch.autograd.grad(out.sum(), tensors2)
    for g1, g2 in zip(grads_ref, grads):
        assert torch.allclose(g1, g2, atol=1e-8, rtol=1e-6)


def test_chunked_scan_gradients_match_reference():
    u, delta, a, bm, cm, dd, z, bias = _inputs(1, 3, 4, 37)
    tensors = [t.clone().requires_grad_(True) for t in (u, delta, a, bm, cm, dd, z, bias)]
    ref = selective_scan_ref(*tensors[:7], tensors[7], True)
    grads_ref = torch.autograd.grad(ref.sum(), tensors)
    tensors2 = [t.detach().clone().requires_grad_(True) for t in tensors]
    out = selective_scan_chunked(*tensors2[:7], tensors2[7], True, chunk_size=8)
    grads = torch.autograd.grad(out.sum(), tensors2)
    for g1, g2 in zip(grads_ref, grads):
        assert torch.allclose(g1, g2, atol=1e-8, rtol=1e-6)


def test_mamba_torch_shapes_and_grad():
    m = MambaTorch(16)
    x = torch.randn(2, 50, 16, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    names = {n for n, _ in m.named_parameters()}
    assert {"in_proj.weight", "conv1d.weight", "x_proj.weight", "dt_proj.weight", "dt_proj.bias", "A_log", "D", "out_proj.weight"} <= names


def test_mamba_forward_matches_sequential_reference():
    torch.manual_seed(0)
    m = MambaTorch(8, chunk_size=16).double()
    x = torch.randn(2, 45, 8, dtype=torch.float64)
    y = mamba_forward_torch(m, x, chunk_size=16)
    # recompute with the sequential scan
    import torch.nn.functional as F

    xz = m.in_proj(x)
    xx, zz = xz.chunk(2, dim=-1)
    xx = F.silu(m.conv1d(xx.transpose(1, 2))[..., :45])
    dbl = m.x_proj(xx.transpose(1, 2))
    dt, bm, cm = torch.split(dbl, [m.dt_rank, m.d_state, m.d_state], dim=-1)
    dt = F.linear(dt, m.dt_proj.weight)
    ys = selective_scan_ref(xx, dt.transpose(1, 2), -torch.exp(m.A_log), bm.transpose(1, 2), cm.transpose(1, 2), m.D, zz.transpose(1, 2), m.dt_proj.bias, True)
    y_ref = m.out_proj(ys.transpose(1, 2))
    assert torch.allclose(y, y_ref, atol=1e-9)


def test_token_mlp_matches_mamba_params():
    for d in (12, 16, 24, 32):
        p_mamba = count_params(MambaTorch(d))
        p_mlp = count_params(TokenMLP(d, matched_mlp_hidden(d)))
        assert abs(p_mlp - p_mamba) / p_mamba < 0.1
    assert make_mixer(16, "mlp")(torch.randn(1, 10, 16)).shape == (1, 10, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("impl", ["chunked", "parallel"])
def test_scans_on_gpu_fp16_autocast(impl):
    u, delta, a, bm, cm, dd, z, bias = (t.float().cuda() for t in _inputs(2, 16, 16, 700))
    ref = selective_scan_ref(u.cpu(), delta.cpu(), a.cpu(), bm.cpu(), cm.cpu(), dd.cpu(), z.cpu(), bias.cpu(), True)
    fn = selective_scan_chunked if impl == "chunked" else selective_scan_parallel
    with torch.autocast("cuda", dtype=torch.float16):
        out = fn(u, delta, a, bm, cm, dd, z, bias, True, chunk_size=64 if impl == "chunked" else 16)
    assert torch.allclose(ref, out.float().cpu(), atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize("groups,l,chunk,max_elems", [(1, 33, 8, 3e7), (4, 50, 8, 3e7), (8, 17, 4, 3e7), (8, 40, 8, 2 * 24 * 40 * 8 * 16 * 3)])
def test_grouped_mamba_matches_separate_layers(groups, l, chunk, max_elems):
    torch.manual_seed(groups)
    layers = [MambaTorch(12, chunk_size=chunk).double() for _ in range(groups)]
    gm = GroupedMambaTorch(groups, 12, chunk_size=chunk, max_scan_elems=max_elems).double()  # last case: slices of 3 groups
    gm.load_from(layers)
    x = torch.randn(2, groups, l, 12, dtype=torch.float64, requires_grad=True)
    y = gm(x)
    y_ref = torch.stack([m(x[:, i]) for i, m in enumerate(layers)], dim=1)
    assert y.shape == x.shape
    assert torch.allclose(y, y_ref, atol=1e-10, rtol=1e-8)
    gx = torch.autograd.grad(y.square().sum(), x)[0]
    gx_ref = torch.autograd.grad(y_ref.square().sum(), x)[0]
    assert torch.allclose(gx, gx_ref, atol=1e-9, rtol=1e-7)


def test_grouped_mamba_param_count_matches_separate_layers():
    for d in (8, 16, 24):
        assert count_params(GroupedMambaTorch(4, d)) == 4 * count_params(MambaTorch(d))


def test_grouped_scan_matches_reference_per_group():
    g, dg, n, l = 3, 4, 6, 40
    u, delta, _, _, _, dd, z, bias = _inputs(2, g * dg, n, l)
    gen = torch.Generator().manual_seed(1)
    a = -torch.rand(g * dg, n, generator=gen, dtype=torch.float64) - 0.1
    bm = torch.randn(2, g, n, l, generator=gen, dtype=torch.float64)
    cm = torch.randn(2, g, n, l, generator=gen, dtype=torch.float64)
    out = selective_scan_parallel(u, delta, a, bm, cm, dd, z, bias, True, chunk_size=8)
    for i in range(g):
        sl = slice(i * dg, (i + 1) * dg)
        ref = selective_scan_ref(u[:, sl], delta[:, sl], a[sl], bm[:, i], cm[:, i], dd[sl], z[:, sl], bias[sl], True)
        assert torch.allclose(out[:, sl], ref, atol=1e-9, rtol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_grouped_mamba_gpu_fp16_close_to_fp32():
    torch.manual_seed(0)
    gm = GroupedMambaTorch(4, 16).cuda()
    x = torch.randn(2, 4, 512, 16, device="cuda")
    y32 = gm(x)
    with torch.autocast("cuda", dtype=torch.float16):
        y16 = gm(x)
    assert torch.isfinite(y16).all()
    assert (y16.float() - y32).abs().max() < 5e-2
