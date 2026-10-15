import torch

from mambalite3d.models.ssm import GroupedMambaTorch
from mambalite3d.tools.efficiency import _analytic_ops, count_ops


def test_conv_macs_are_exact():
    model = torch.nn.Conv3d(2, 4, 3, padding=1, bias=False)
    ops = count_ops(model, (4, 8, 8), torch.device("cpu"))
    assert round(ops["gmacs"] * 1e9) == 4 * (2 * 27) * (4 * 8 * 8)
    assert abs(ops["gflops"] - 2 * ops["gmacs"]) < 1e-15
    assert ops["gmacs_barrido_mamba"] == 0 and ops["gmacs_atencion"] == 0


def test_mamba_scan_and_attention_are_counted_analytically():
    g, d_model, n, bsz, length = 2, 8, 4, 1, 64
    m = GroupedMambaTorch(groups=g, d_model=d_model, d_state=n)
    acc = {"scan": 0, "attn": 0}
    with torch.no_grad(), _analytic_ops(acc):
        y = m(torch.randn(bsz, g, length, d_model))
        q = torch.randn(1, 2, 16, 8)
        a = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    di = g * m.d_inner
    assert y.shape == (bsz, g, length, d_model) and a.shape == (1, 2, 16, 8)
    assert acc["scan"] == 9 * bsz * length * di * n + 2 * bsz * di * length
    assert acc["attn"] == 2 * 16 * 16 * (8 + 8)
    # the original functions are restored afterwards
    from mambalite3d.models import ssm

    assert ssm.selective_scan.__name__ == "selective_scan"
    assert torch.nn.functional.scaled_dot_product_attention.__name__ != "sdpa_stub"
