import numpy as np

from mambalite3d.eval.components import component_stats, score_cases
from mambalite3d.eval.metrics import aggregate, evaluate_case


def _random_case(seed):
    rng = np.random.default_rng(seed)
    gt = np.zeros((20, 40, 40), np.uint8)
    pred = np.zeros_like(gt)
    for _ in range(rng.integers(0, 3)):
        z, y, x = rng.integers(3, 17), rng.integers(5, 35), rng.integers(5, 35)
        gt[z - 2 : z + 2, y - 3 : y + 3, x - 3 : x + 3] = rng.integers(1, 3)
    for _ in range(rng.integers(0, 6)):
        z, y, x = rng.integers(2, 18), rng.integers(3, 37), rng.integers(3, 37)
        s = rng.integers(1, 4)
        pred[z - 1 : z + 1, y - s : y + s, x - s : x + s] = rng.integers(1, 3)
    probs = rng.random((3, *gt.shape)).astype(np.float32)
    return pred, gt, probs


def test_score_cases_matches_reference_metrics():
    stats, results = [], []
    for seed in range(30):
        pred, gt, probs = _random_case(seed)
        stats.append(component_stats(pred, gt, probs, probs[0], voxel_mm3=3.0))
        results.append(evaluate_case(str(seed), pred, gt, 3.0))
    ref = aggregate(results)
    got = score_cases(stats)
    for k in ("gtvp_dice_mean", "gtvn_dice_agg", "gtvn_f1_agg", "gtvn_tp", "gtvn_fp", "gtvn_fn",
              "gtvp_dice_agg", "dsc_agg_mean"):
        assert abs(ref[k] - got[k]) < 1e-9, (k, ref[k], got[k])


def test_score_cases_removing_components():
    gt = np.zeros((10, 30, 30), np.uint8)
    gt[2:6, 5:12, 5:12] = 2
    pred = np.zeros_like(gt)
    pred[2:6, 5:12, 5:12] = 2
    pred[8, 25, 25] = 2  # 1-voxel false positive
    st = component_stats(pred, gt, None, None, 1.0)
    before = score_cases([st])
    after = score_cases([st], keep_n=lambda c, comps: c["vox"] >= 2)
    assert before["gtvn_fp"] == 1 and after["gtvn_fp"] == 0 and after["gtvn_f1_agg"] == 1.0
    assert after["gtvn_dice_agg"] == 1.0 > before["gtvn_dice_agg"]


def test_postprocess_prediction_matches_rule_simulation():
    from mambalite3d.eval.components import postprocess_prediction
    from mambalite3d.tools.tune_postprocess import apply_rules

    rules = {"gtvp": {"largest": True, "min_ml": 0.0, "min_maxp": 0.3, "case_min_ml": 0.05},
             "gtvn": {"min_ml": 0.02, "min_maxp": 0.4}}
    for seed in range(20):
        pred, gt, probs = _random_case(seed)
        st = component_stats(pred, gt, probs, None, 3.0)
        simulated = score_cases(apply_rules([st], rules))
        real = component_stats(postprocess_prediction(pred, probs, rules, 3.0), gt, probs, None, 3.0)
        got = score_cases([real])
        for k in ("gtvp_dice_mean", "gtvn_dice_agg", "gtvn_tp", "gtvn_fp", "gtvn_fn"):
            assert abs(simulated[k] - got[k]) < 1e-9, (seed, k, simulated[k], got[k])
