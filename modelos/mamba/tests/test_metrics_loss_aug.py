import numpy as np
import pytest
import torch

from mambalite3d.data.augment import augment_batch, context_size
from mambalite3d.config import DEFAULTS
from mambalite3d.eval.metrics import aggregate, evaluate_case, match_components
from mambalite3d.training.loss import DeepSupervisionLoss, DiceCELoss, ds_weights


def _ball(shape, center, r):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= r * r


def _case(shape=(20, 40, 40)):
    gt = np.zeros(shape, np.uint8)
    gt[_ball(shape, (10, 20, 20), 4)] = 1
    gt[_ball(shape, (10, 8, 8), 2)] = 2
    gt[_ball(shape, (10, 32, 32), 2)] = 2
    return gt


def test_perfect_prediction():
    gt = _case()
    agg = aggregate([evaluate_case("a", gt.copy(), gt, 1.0)])
    assert agg["gtvp_dice_mean"] == 1.0 and agg["gtvn_dice_agg"] == 1.0 and agg["gtvn_f1_agg"] == 1.0


def test_missing_nodes_and_false_positive():
    gt = _case()
    pred = gt.copy()
    pred[pred == 2] = 0
    r = evaluate_case("a", pred, gt, 1.0)
    assert (r.gtvn_tp, r.gtvn_fp, r.gtvn_fn) == (0, 0, 2)
    pred2 = gt.copy()
    pred2[_ball(gt.shape, (3, 35, 5), 1)] = 2
    r2 = evaluate_case("b", pred2, gt, 1.0)
    assert (r2.gtvn_tp, r2.gtvn_fp, r2.gtvn_fn) == (2, 1, 0)


def test_low_iou_is_not_a_match():
    g = np.zeros((10, 10, 10), bool)
    g[2:6, 2:6, 2:6] = True  # 64 voxels
    p = np.zeros_like(g)
    p[2:6, 2:6, 5:9] = True  # overlap 16 -> IoU = 16 / 112 < 0.3
    assert match_components(p, g) == (0, 1, 1)


def test_merged_component_matches_once():
    g = np.zeros((10, 20, 10), bool)
    g[3:6, 2:5, 3:6] = True
    g[3:6, 8:11, 3:6] = True
    p = np.zeros_like(g)
    p[3:6, 2:11, 3:6] = True
    tp, fp, fn = match_components(p, g)
    assert tp <= 1 and tp + fn == 2 and tp + fp == 1


def test_aggregated_dice_uses_mm3():
    gt = np.zeros((10, 10, 10), np.uint8)
    gt[2:4, 2:4, 2:4] = 2
    pred = np.zeros_like(gt)
    a = evaluate_case("a", gt.copy(), gt, voxel_mm3=8.0)  # perfect, large voxels
    b = evaluate_case("b", pred, gt, voxel_mm3=1.0)  # missed, small voxels
    agg = aggregate([a, b])
    assert abs(agg["gtvn_dice_agg"] - (2 * 64) / (2 * 64 + 8)) < 1e-9


def test_empty_gtvp_conventions():
    gt = np.zeros((5, 5, 5), np.uint8)
    agg = aggregate([evaluate_case("a", gt.copy(), gt, 1.0)])
    assert agg["gtvp_dice_mean"] == 1.0 and np.isnan(agg["gtvp_dice_mean_ref_only"])
    pred = gt.copy()
    pred[1, 1, 1] = 1
    assert aggregate([evaluate_case("b", pred, gt, 1.0)])["gtvp_dice_mean"] == 0.0


def test_hecktor2022_aggregated_gtvp_dice():
    """Same predictions scored both ways: per-patient mean (2025) vs aggregated (2022)."""
    shape = (12, 12, 12)
    # A: large primary, prediction shifted by one voxel -> 900 of 1000 voxels overlap, Dice 0.9
    gt_a = np.zeros(shape, np.uint8)
    gt_a[1:11, 1:11, 1:11] = 1
    pred_a = np.zeros_like(gt_a)
    pred_a[1:11, 1:11, 2:12] = 1
    # B: small primary (8 voxels), half predicted -> Dice 8 / 12
    gt_b = np.zeros(shape, np.uint8)
    gt_b[2:4, 2:4, 2:4] = 1
    pred_b = np.zeros_like(gt_b)
    pred_b[2:3, 2:4, 2:4] = 1
    # C: no primary in the reference, one false-positive voxel -> Dice 0
    gt_c = np.zeros(shape, np.uint8)
    pred_c = np.zeros_like(gt_c)
    pred_c[5, 5, 5] = 1
    res = [evaluate_case("A", pred_a, gt_a, 1.0), evaluate_case("B", pred_b, gt_b, 1.0),
           evaluate_case("C", pred_c, gt_c, 1.0)]
    assert (res[0].gtvp_inter_mm3, res[0].gtvp_sum_mm3) == (900.0, 2000.0)
    agg = aggregate(res)
    assert abs(agg["gtvp_dice_mean"] - (0.9 + 8 / 12 + 0.0) / 3) < 1e-9
    assert abs(agg["gtvp_dice_agg"] - 2 * (900 + 4 + 0) / (2000 + 12 + 1)) < 1e-9
    assert agg["gtvn_dice_agg"] == 1.0  # no nodes anywhere
    assert abs(agg["dsc_agg_mean"] - 0.5 * (agg["gtvp_dice_agg"] + 1.0)) < 1e-9


def test_aggregate_without_gtvp_volumes_is_nan():
    """Per-case rows from older CSV files (no GTVp volumes) still aggregate; the 2022 metric is NaN."""
    from mambalite3d.eval.metrics import CaseResult

    r = CaseResult("a", 0.5, True, 10.0, 20.0, 1, 0, 0, 1.0)
    agg = aggregate([r])
    assert agg["gtvn_dice_agg"] == 1.0 and np.isnan(agg["gtvp_dice_agg"]) and np.isnan(agg["dsc_agg_mean"])


def test_loss_decreases_for_better_prediction():
    target = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    target[..., 2:6, 2:6, 2:6] = 1
    good = torch.zeros(1, 3, 8, 8, 8)
    good[:, 0] = 5
    good[:, 0][target[:, 0] == 1] = -5
    good[:, 1][target[:, 0] == 1] = 5
    bad = torch.zeros(1, 3, 8, 8, 8)
    loss = DiceCELoss()
    assert loss(good, target) < loss(bad, target)
    ds = DeepSupervisionLoss(DiceCELoss())
    out = ds([good, good[..., ::2, ::2, ::2]], target)
    assert torch.isfinite(out)
    assert ds_weights(3) == [pytest.approx(2 / 3), pytest.approx(1 / 3), 0.0]


def test_augment_disabled_is_exact_center_crop():
    patch = (8, 16, 16)
    ext = context_size(patch, 0.25)
    assert all((e - p) % 2 == 0 for e, p in zip(ext, patch))
    img = torch.randn(2, 2, *ext)
    lab = torch.randint(0, 3, (2, 1, *ext))
    gen = torch.Generator().manual_seed(0)
    x, y = augment_batch(img, lab, patch, (3.0, 1.0, 1.0), DEFAULTS["augment"], gen, enabled=False)
    oz, oy, ox = [(e - p) // 2 for e, p in zip(ext, patch)]
    ref = img[:, :, oz : oz + 8, oy : oy + 16, ox : ox + 16]
    assert torch.allclose(x, ref, atol=1e-5)
    assert torch.equal(y, lab[:, :, oz : oz + 8, oy : oy + 16, ox : ox + 16])


def test_augment_enabled_shapes_and_labels():
    patch = (8, 16, 16)
    ext = context_size(patch, 0.25)
    cfg = dict(DEFAULTS["augment"])
    cfg.update(p_rotation=1.0, p_scale=1.0, p_brightness=1.0, p_contrast=1.0, p_gamma=1.0, p_noise=1.0, p_blur=1.0)
    img = torch.rand(2, 2, *ext)
    lab = torch.randint(0, 3, (2, 1, *ext))
    x, y = augment_batch(img, lab, patch, (3.0, 1.0, 1.0), cfg, torch.Generator().manual_seed(1), enabled=True)
    assert x.shape == (2, 2, *patch) and y.shape == (2, 1, *patch)
    assert torch.isfinite(x).all()
    assert set(torch.unique(y).tolist()) <= {0, 1, 2}
