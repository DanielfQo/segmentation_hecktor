"""Selective state space (Mamba) layer implemented in pure PyTorch.

`mamba_ssm` ships CUDA kernels only for Linux. This module reproduces the parameters, the
initialisation and the forward pass of `mamba_ssm.modules.mamba_simple.Mamba` so the same
network runs on Windows (and on CPU for tests). If `mamba_ssm` is importable (Linux/WSL) and the
input lives on CUDA, its fused kernels are used instead, with identical parameters.

Shapes follow `mamba_ssm.ops.selective_scan_interface.selective_scan_ref`:
    u, delta, z: (b, d, l)   A: (d, n)   B, C: (b, n, l)   D: (d,)
"""
from __future__ import annotations

import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_FORCE_TORCH = os.environ.get("MAMBALITE_FORCE_TORCH", "0") == "1"
try:  # pragma: no cover - only on Linux with mamba_ssm installed
    if _FORCE_TORCH:
        raise ImportError("forced torch backend")
    from mamba_ssm import Mamba as _MambaCUDA  # type: ignore

    HAS_MAMBA_SSM = True
except Exception:  # noqa: BLE001
    _MambaCUDA = None
    HAS_MAMBA_SSM = False


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """Sequential reference implementation (slow, used only in tests)."""
    dtype_in = u.dtype
    u = u.double() if dtype_in == torch.float64 else u.float()
    work = u.dtype
    delta = delta.to(work)
    if delta_bias is not None:
        delta = delta + delta_bias.to(work)[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    b, d, l = u.shape
    n = A.shape[1]
    A = A.to(work)
    B = B.to(work)
    C = C.to(work)
    state = u.new_zeros(b, d, n)
    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    delta_b_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    ys = []
    for i in range(l):
        state = delta_a[:, :, i] * state + delta_b_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", state, C[:, :, i]))
    y = torch.stack(ys, dim=2)
    if D is not None:
        y = y + u * D.to(work)[None, :, None]
    if z is not None:
        y = y * F.silu(z.to(work))
    return y.to(dtype_in)


def _scan_chunk(u_c, dt_c, A, B_c, C_c, h0):
    """Closed form of the recurrence inside one chunk of T steps.

    h_t = exp(cs_t) h0 + sum_{k<=t} exp(cs_t - cs_k) * dt_k B_k u_k,  cs = cumsum(dt * A)
    All exponents are <= 0, so nothing overflows.
    """
    T = u_c.shape[-1]
    d_a = dt_c.unsqueeze(-1) * A[None, :, None, :]  # (b, d, T, n), <= 0
    cs = torch.cumsum(d_a, dim=2)  # (b, d, T, n)
    d_b_u = (dt_c * u_c).unsqueeze(-1) * B_c.transpose(1, 2).unsqueeze(1)  # (b, d, T, n)
    diff = cs.unsqueeze(3) - cs.unsqueeze(2)  # (b, d, t, k, n) = cs_t - cs_k
    causal = torch.ones(T, T, dtype=torch.bool, device=u_c.device).tril()  # k <= t
    weights = torch.exp(diff.masked_fill(~causal[None, None, :, :, None], float("-inf")))
    h = torch.einsum("bdtkn,bdkn->bdtn", weights, d_b_u) + torch.exp(cs) * h0.unsqueeze(2)
    y = torch.einsum("bdtn,bnt->bdt", h, C_c)
    return y, h[:, :, -1]


def selective_scan_chunked(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=True,
    chunk_size: int = 64,
    use_checkpoint: bool = True,
):
    """Chunked selective scan in float32 (float64 if the inputs are float64)."""
    dtype_in = u.dtype
    with torch.autocast(device_type=u.device.type, enabled=False):
        work = torch.float64 if dtype_in == torch.float64 else torch.float32
        u = u.to(work)
        delta = delta.to(work)
        A = A.to(work)
        B = B.to(work)
        C = C.to(work)
        if delta_bias is not None:
            delta = delta + delta_bias.to(work)[None, :, None]
        if delta_softplus:
            delta = F.softplus(delta)
        b, d, l = u.shape
        n = A.shape[1]
        h = u.new_zeros(b, d, n)
        ys = []
        for s in range(0, l, chunk_size):
            e = min(s + chunk_size, l)
            args = (u[..., s:e], delta[..., s:e], A, B[..., s:e], C[..., s:e], h)
            if use_checkpoint and torch.is_grad_enabled():
                y_c, h = checkpoint(_scan_chunk, *args, use_reentrant=False)
            else:
                y_c, h = _scan_chunk(*args)
            ys.append(y_c)
        y = torch.cat(ys, dim=-1)
        if D is not None:
            y = y + u * D.to(work)[None, :, None]
        if z is not None:
            y = y * F.silu(z.to(work))
    return y.to(dtype_in)


def _linrec(log_a: torch.Tensor, x: torch.Tensor, chunk: int) -> torch.Tensor:
    """Parallel solution of h_t = exp(log_a_t) * h_{t-1} + x_t (h_{-1} = 0) along dim 2.

    log_a, x: (b, d, L, n). Chunks of `chunk` steps are solved in closed form all at once
    (exponents <= 0, no overflow); the chunk-to-chunk recurrence is solved recursively with the
    same function, so there is no Python loop over the sequence.
    """
    b, d, L, n = x.shape
    if L <= chunk:
        cs = torch.cumsum(log_a, dim=2)
        mask = ~torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        w = (cs.unsqueeze(3) - cs.unsqueeze(2)).masked_fill_(mask[:, :, None], float("-inf")).exp_()
        return (w * x.unsqueeze(2)).sum(3)
    nc = -(-L // chunk)
    pad = nc * chunk - L
    if pad:
        log_a = F.pad(log_a, (0, 0, 0, pad))
        x = F.pad(x, (0, 0, 0, pad))
    la = log_a.reshape(b, d, nc, chunk, n)
    xr = x.reshape(b, d, nc, chunk, n)
    cs = torch.cumsum(la, dim=3)  # (b, d, nc, T, n)
    mask = ~torch.ones(chunk, chunk, dtype=torch.bool, device=x.device).tril()
    w = (cs.unsqueeze(4) - cs.unsqueeze(3)).masked_fill_(mask[:, :, None], float("-inf")).exp_()  # (b,d,nc,t,k,n)
    h_loc = (w * xr.unsqueeze(3)).sum(4)  # local states, zero initial state per chunk
    h_end = _linrec(cs[:, :, :, -1], h_loc[:, :, :, -1], chunk)  # true state at the end of each chunk
    h_prev = torch.cat([torch.zeros_like(h_end[:, :, :1]), h_end[:, :, :-1]], dim=2)
    h = h_loc + torch.exp(cs) * h_prev.unsqueeze(3)
    return h.reshape(b, d, nc * chunk, n)[:, :, :L]


def selective_scan_parallel(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=True,
    chunk_size: int = 16,
    checkpoint_elems: float = 4e6,
):
    """Selective scan without a Python loop over time (float32, or float64 for float64 inputs).

    Large problems are wrapped in activation checkpointing so the (b, d, L, T, n) chunk matrices
    are not kept for the backward pass.
    """
    dtype_in = u.dtype
    with torch.autocast(device_type=u.device.type, enabled=False):
        work = torch.float64 if dtype_in == torch.float64 else torch.float32
        u = u.to(work)
        delta = delta.to(work)
        A = A.to(work)
        B = B.to(work)
        C = C.to(work)
        if delta_bias is not None:
            delta = delta + delta_bias.to(work)[None, :, None]
        if delta_softplus:
            delta = F.softplus(delta)

        def core(u_, delta_, A_, B_, C_):
            b_, d_, l_ = u_.shape
            log_a = delta_.unsqueeze(-1) * A_[None, :, None, :]  # (b, d, L, n)
            if B_.dim() == 3:  # B, C shared by all channels: (b, n, L)
                x = (delta_ * u_).unsqueeze(-1) * B_.transpose(1, 2).unsqueeze(1)  # (b, d, L, n)
                h = _linrec(log_a, x, chunk_size)
                return torch.einsum("bdln,bnl->bdl", h, C_)
            # grouped: B, C are (b, g, n, L) and channel d belongs to group d // (d / g)
            g_ = B_.shape[1]
            du = (delta_ * u_).reshape(b_, g_, d_ // g_, l_)
            x = (du.unsqueeze(-1) * B_.transpose(2, 3).unsqueeze(2)).reshape(b_, d_, l_, -1)
            h = _linrec(log_a, x, chunk_size).reshape(b_, g_, d_ // g_, l_, -1)
            return torch.einsum("bgdln,bgnl->bgdl", h, C_).reshape(b_, d_, l_)

        b, d, l = u.shape
        n = A.shape[1]
        if torch.is_grad_enabled() and b * d * l * chunk_size * n > checkpoint_elems:
            y = checkpoint(core, u, delta, A, B, C, use_reentrant=False)
        else:
            y = core(u, delta, A, B, C)
        if D is not None:
            y = y + u * D.to(work)[None, :, None]
        if z is not None:
            y = y * F.silu(z.to(work))
    return y.to(dtype_in)


SCAN_IMPL = os.environ.get("MAMBALITE_SCAN", "parallel")


def selective_scan(*args, chunk_size: int = 16, **kwargs):
    if SCAN_IMPL == "chunked" and args[3].dim() == 3:  # the loop scan only supports shared B/C
        return selective_scan_chunked(*args, chunk_size=max(chunk_size, 64), **kwargs)
    return selective_scan_parallel(*args, chunk_size=chunk_size, **kwargs)


class MambaTorch(nn.Module):
    """Same parameter names and initialisation as mamba_ssm.modules.mamba_simple.Mamba."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank="auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_scale: float = 1.0,
        conv_bias: bool = True,
        bias: bool = False,
        chunk_size: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.chunk_size = chunk_size

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, bias=conv_bias
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # (B, L, d_model)
        return mamba_forward_torch(self, hidden, chunk_size=self.chunk_size)


def mamba_forward_torch(m: nn.Module, hidden: torch.Tensor, chunk_size: int = 16) -> torch.Tensor:
    """Slow path of mamba_simple.Mamba.forward, usable with MambaTorch or mamba_ssm.Mamba parameters."""
    seqlen = hidden.shape[1]
    xz = m.in_proj(hidden)  # (B, L, 2*d_inner)
    x, z = xz.chunk(2, dim=-1)
    x = x.transpose(1, 2)  # (B, d_inner, L)
    x = F.silu(m.conv1d(x)[..., :seqlen])
    x_dbl = m.x_proj(x.transpose(1, 2))  # (B, L, dt_rank + 2 * d_state)
    dt, b_mat, c_mat = torch.split(x_dbl, [m.dt_rank, m.d_state, m.d_state], dim=-1)
    dt = F.linear(dt, m.dt_proj.weight)  # (B, L, d_inner), bias goes to the scan
    work = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    a = -torch.exp(m.A_log.to(work))
    y = selective_scan(
        x,
        dt.transpose(1, 2),
        a,
        b_mat.transpose(1, 2),
        c_mat.transpose(1, 2),
        m.D.to(work),
        z=z.transpose(1, 2),
        delta_bias=m.dt_proj.bias.to(work),
        delta_softplus=True,
        chunk_size=chunk_size,
    )
    return m.out_proj(y.transpose(1, 2))


class GroupedMambaTorch(nn.Module):
    """G independent Mamba layers evaluated together with batched operations.

    Same maths and initialisation as G separate MambaTorch layers, but one scan call instead of G,
    which matters on Windows where kernel launches are expensive. Input/output: (B, G, L, d_model).
    """

    def __init__(
        self,
        groups: int,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        chunk_size: int = 8,
        max_scan_elems: float = 3e7,
    ):
        super().__init__()
        g = groups
        self.max_scan_elems = max_scan_elems  # groups are scanned in slices of at most this many state elements
        self.groups, self.d_model, self.d_state, self.d_conv = g, d_model, d_state, d_conv
        self.d_inner = di = int(expand * d_model)
        self.dt_rank = r = math.ceil(d_model / 16)
        self.chunk_size = chunk_size

        def uniform(*shape, bound):
            return nn.Parameter(torch.empty(*shape).uniform_(-bound, bound))

        self.in_proj_w = uniform(g, 2 * di, d_model, bound=1 / math.sqrt(d_model))
        self.conv_w = uniform(g * di, 1, d_conv, bound=1 / math.sqrt(d_conv))
        self.conv_b = uniform(g * di, bound=1 / math.sqrt(d_conv))
        self.x_proj_w = uniform(g, r + 2 * d_state, di, bound=1 / math.sqrt(di))
        self.dt_proj_w = uniform(g, di, r, bound=r**-0.5)
        dt = torch.exp(torch.rand(g, di) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        self.dt_proj_b = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(g, di, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(g, di))
        self.out_proj_w = uniform(g, d_model, di, bound=1 / math.sqrt(di))

    @torch.no_grad()
    def load_from(self, layers) -> None:
        """Copies the weights of G MambaTorch layers (used by the equivalence test)."""
        for i, m in enumerate(layers):
            sl = slice(i * self.d_inner, (i + 1) * self.d_inner)
            self.in_proj_w[i].copy_(m.in_proj.weight)
            self.conv_w[sl].copy_(m.conv1d.weight)
            self.conv_b[sl].copy_(m.conv1d.bias)
            self.x_proj_w[i].copy_(m.x_proj.weight)
            self.dt_proj_w[i].copy_(m.dt_proj.weight)
            self.dt_proj_b[i].copy_(m.dt_proj.bias)
            self.A_log[i].copy_(m.A_log)
            self.D[i].copy_(m.D)
            self.out_proj_w[i].copy_(m.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, g, seqlen, _ = x.shape
        di, n, r = self.d_inner, self.d_state, self.dt_rank
        xz = torch.einsum("bgld,ged->bgle", x, self.in_proj_w)
        xi, z = xz.chunk(2, dim=-1)  # (B, G, L, di)
        xi = xi.permute(0, 1, 3, 2).reshape(bsz, g * di, seqlen)
        xi = F.silu(F.conv1d(xi, self.conv_w, self.conv_b, padding=self.d_conv - 1, groups=g * di)[..., :seqlen])
        dbl = torch.einsum("bgdl,ged->bgle", xi.reshape(bsz, g, di, seqlen), self.x_proj_w)  # (B, G, L, r + 2n)
        dt, b_mat, c_mat = torch.split(dbl, [r, n, n], dim=-1)
        dt = torch.einsum("bglr,gdr->bgdl", dt, self.dt_proj_w).reshape(bsz, g * di, seqlen)
        work = torch.float64 if x.dtype == torch.float64 else torch.float32
        a = -torch.exp(self.A_log.to(work))  # (G, di, n)
        d_skip = self.D.to(work)
        bias = self.dt_proj_b.to(work)
        zz = z.permute(0, 1, 3, 2).reshape(bsz, g * di, seqlen)
        # all groups in one call when small; slices of groups when the (b, d, L, chunk, n) state
        # tensors would be large, so peak memory stays close to that of separate layers
        per_group = bsz * di * seqlen * min(self.chunk_size, seqlen) * n
        step = max(1, min(g, int(self.max_scan_elems // max(per_group, 1))))
        ys = []
        for s0 in range(0, g, step):
            s1 = min(g, s0 + step)
            ch = slice(s0 * di, s1 * di)
            ys.append(
                selective_scan(
                    xi[:, ch],
                    dt[:, ch],
                    a[s0:s1].reshape(-1, n),
                    b_mat[:, s0:s1].transpose(2, 3),
                    c_mat[:, s0:s1].transpose(2, 3),
                    d_skip[s0:s1].reshape(-1),
                    z=zz[:, ch],
                    delta_bias=bias[s0:s1].reshape(-1),
                    delta_softplus=True,
                    chunk_size=self.chunk_size,
                )
            )
        y = ys[0] if len(ys) == 1 else torch.cat(ys, dim=1)
        y = y.reshape(bsz, g, di, seqlen)
        return torch.einsum("bgdl,ged->bgle", y, self.out_proj_w)


class MambaMixer(nn.Module):
    """Token mixer (B, L, C) -> (B, L, C) backed by mamba_ssm (CUDA, Linux) or MambaTorch."""

    _warned = False

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, chunk_size: int = 16):
        super().__init__()
        self.chunk_size = chunk_size
        if HAS_MAMBA_SSM:
            self.impl = _MambaCUDA(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.uses_cuda_kernels = True
        else:
            self.impl = MambaTorch(d_model, d_state=d_state, d_conv=d_conv, expand=expand, chunk_size=chunk_size)
            self.uses_cuda_kernels = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        if self.uses_cuda_kernels and x.is_cuda:
            return self.impl(x)
        return mamba_forward_torch(self.impl, x, chunk_size=self.chunk_size)


class TokenMLP(nn.Module):
    """Mamba-free control from the MambaLiteUNet supplementary (sec. 8.2): x + FC(GELU(FC(LN(x))))."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def matched_mlp_hidden(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> int:
    """Hidden width that gives TokenMLP about the same number of parameters as one Mamba layer."""
    target = count_params(MambaTorch(d_model, d_state=d_state, d_conv=d_conv, expand=expand))
    return max(4, int(round((target - 3 * d_model) / (2 * d_model + 1))))


def make_mixer(d_model: int, kind: str = "mamba", d_state: int = 16, d_conv: int = 4, expand: int = 2, chunk_size: int = 16):
    if kind == "mamba":
        if not HAS_MAMBA_SSM and not MambaMixer._warned and torch.cuda.is_available():
            MambaMixer._warned = True
            warnings.warn("mamba_ssm not available: using the pure PyTorch selective scan (slower, same maths).")
        return MambaMixer(d_model, d_state=d_state, d_conv=d_conv, expand=expand, chunk_size=chunk_size)
    if kind == "mlp":
        return TokenMLP(d_model, matched_mlp_hidden(d_model, d_state, d_conv, expand))
    raise ValueError(f"unknown mixer kind: {kind}")
