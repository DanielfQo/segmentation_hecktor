import csv
import json

from mambalite3d.tools.compare_runs import main

COLS = ["case_id", "dice_gtvp", "gtvp_in_ref", "gtvn_inter_mm3", "gtvn_sum_mm3", "gtvn_tp", "gtvn_fp", "gtvn_fn",
        "dice_gtvn_case", "gtvp_inter_mm3", "gtvp_sum_mm3"]


def _write(run_dir, tag, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / f"per_case_test_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)


def _rows(shift):
    rows = []
    for i in range(20):
        d = min(1.0, 0.5 + 0.02 * i + shift)
        rows.append([f"C-{i:03d}", d, True, 40 + 10 * shift, 100, 1, 1, 0, 0.4, 50 * d, 100])
    return rows


def test_identical_runs_have_no_difference(tmp_path):
    _write(tmp_path / "a", "best", _rows(0.0))
    _write(tmp_path / "b", "final", _rows(0.0))
    out = tmp_path / "cmp.json"
    assert main(["--a", f"{tmp_path / 'a'}:best", "--b", f"{tmp_path / 'b'}:final", "--split", "test",
                 "--n-boot", "200", "--n-perm", "200", "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))["results"]
    for k in ("gtvp_mean", "gtvn_agg", "score_2025", "gtvp_agg", "dsc_agg_mean_2022"):
        assert abs(res[k]["diff"]) < 1e-12 and res[k]["p_perm"] == 1.0
    assert "gtvp_wilcoxon" not in res


def test_better_run_is_detected(tmp_path):
    _write(tmp_path / "a", "best", _rows(0.0))
    _write(tmp_path / "b", "best", _rows(0.1))
    out = tmp_path / "cmp.json"
    main(["--a", f"{tmp_path / 'a'}:best", "--b", f"{tmp_path / 'b'}:best", "--n-boot", "500", "--n-perm", "500",
          "--out", str(out)])
    res = json.loads(out.read_text(encoding="utf-8"))["results"]
    assert res["gtvp_mean"]["diff"] > 0 and res["gtvp_mean"]["ci95"][0] > 0 and res["gtvp_mean"]["p_perm"] < 0.01
    assert res["gtvn_agg"]["diff"] > 0 and res["dsc_agg_mean_2022"]["diff"] > 0
    assert res["gtvp_wilcoxon"]["b_better"] > 0
