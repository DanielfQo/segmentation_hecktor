import json
import math

from mambalite3d.eval.hecktor import hecktor2022, hecktor2025, ranking_2022, ranking_2025, selection_score
from mambalite3d.tools.report import main as report_main
from mambalite3d.tools.report import parse_model


def _agg(p_mean, n_agg, f1, p_agg=None):
    d = {"n_cases": 10, "gtvp_dice_mean": p_mean, "gtvn_dice_agg": n_agg, "gtvn_f1_agg": f1}
    if p_agg is not None:
        d["gtvp_dice_agg"] = p_agg
    return d


def test_metrics_of_each_edition():
    a = _agg(0.70, 0.67, 0.35, p_agg=0.77)
    assert hecktor2025(a) == {"GTVp_DSC": 0.70, "GTVn_DSCagg": 0.67, "GTVn_F1agg": 0.35}
    h22 = hecktor2022(a)
    assert h22["GTVp_DSCagg"] == 0.77 and abs(h22["DSCagg_media"] - 0.72) < 1e-12
    assert math.isnan(hecktor2022(_agg(0.7, 0.67, 0.35))["DSCagg_media"])  # old evaluation without GTVp volumes
    assert abs(selection_score(a, "2025") - 0.685) < 1e-12 and abs(selection_score(a, "2022") - 0.72) < 1e-12


def test_ranking_2025_is_borda_of_gtvp_and_gtvn():
    rows = {
        "A": hecktor2025(_agg(0.80, 0.60, 0.30)),  # best primary, worst nodes
        "B": hecktor2025(_agg(0.70, 0.70, 0.50)),
        "C": hecktor2025(_agg(0.75, 0.65, 0.60)),
    }
    r = ranking_2025(rows)
    # GTVp: A1 C2 B3; GTVn DSCagg: B1 C2 A3; F1agg: C1 B2 A3 -> GTVn sums A6 B3 C3 -> A3, B1.5, C1.5
    assert [r[n]["puesto_GTVp"] for n in "ABC"] == [1, 3, 2]
    assert [r[n]["puesto_GTVn"] for n in "ABC"] == [3, 1.5, 1.5]
    # final sums: A 4, B 4.5, C 3.5
    assert [r[n]["puesto_final"] for n in "ABC"] == [2, 3, 1]
    # leaderboard-style mean position over the three columns: A 7/3, B 6/3, C 5/3
    assert [r[n]["puesto_posicion_media"] for n in "ABC"] == [3, 2, 1]


def test_ranking_2022_by_mean_aggregated_dice():
    rows = {"A": hecktor2022(_agg(0, 0.6, 0, p_agg=0.8)), "B": hecktor2022(_agg(0, 0.7, 0, p_agg=0.8)),
            "C": hecktor2022(_agg(0, 0.7, 0))}
    r = ranking_2022(rows)
    assert r["B"] == 1 and r["A"] == 2 and math.isnan(r["C"])


def test_parse_model_specs(tmp_path):
    assert parse_model("base=runs/full_base_fold0")[2] is None
    assert parse_model("wide=runs/exp_wide:final")[2] == "final"
    name, path, tag = parse_model(f"x={tmp_path}")
    assert name == "x" and path == tmp_path and tag is None
    assert parse_model(f"x={tmp_path}:best")[2] == "best"


def _write(run, split, tag, agg):
    run.mkdir(parents=True, exist_ok=True)
    (run / f"metrics_{split}_{tag}.json").write_text(json.dumps({"original_space": agg}), encoding="utf-8")


def test_report_chooses_checkpoint_and_writes_tables(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    # model a: 'final' wins with the 2025 summary, 'best' wins with the 2022 mean aggregated Dice
    _write(a, "val", "final", _agg(0.70, 0.66, 0.3, p_agg=0.70))
    _write(a, "val", "best", _agg(0.64, 0.66, 0.3, p_agg=0.80))
    _write(a, "test", "final", _agg(0.71, 0.67, 0.3, p_agg=0.72))
    _write(a, "test", "best", _agg(0.66, 0.68, 0.3, p_agg=0.79))
    # model b: old evaluations without GTVp volumes
    _write(b, "val", "final", _agg(0.60, 0.70, 0.4))
    _write(b, "test", "final", _agg(0.62, 0.71, 0.4))
    out = tmp_path / "res"
    assert report_main(["--models", f"a={a}", f"b={b}", "--out", str(out)]) == 0
    res = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert res["modelos"]["a"]["checkpoint"] == "final" and res["modelos"]["b"]["checkpoint"] == "final"
    assert res["test"]["a"]["hecktor2025"]["GTVp_DSC"] == 0.71
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "HECKTOR 2025" in md and "HECKTOR 2022" in md and "| b | final |" in md and "| - |" in md
    report_main(["--models", f"a={a}", f"b={b}", "--select", "2022", "--out", str(out)])
    res = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert res["modelos"]["a"]["checkpoint"] == "best"
    assert res["modelos"]["b"]["eleccion"].startswith("validacion 2025")  # 2022 missing -> falls back to 2025
