"""Metricas y comparacion pareada de la evaluacion por variantes (sin GPU: con filas por caso sinteticas)."""
import json

import numpy as np
import pytest

from bifusion3d.tools import evaluar_variantes as ev


def _row(cid, dsc, inter, total, tp, fp, fn, down=270, restore="nearest", groups=None):
    tg, fg, ng = groups or (tp, fp, fn)
    return {"case_id": cid, "down": down, "restore": restore, "dice_gtvp": dsc, "gtvp_in_ref": True,
            "gtvn_inter_mm3": inter, "gtvn_sum_mm3": total, "tp": tp, "fp": fp, "fn": fn,
            "tp_grupos": tg, "fp_grupos": fg, "fn_grupos": ng}


def test_metricas_oficiales():
    rows = [_row("A", 0.8, 50, 120, 2, 1, 0), _row("B", 0.6, 30, 80, 1, 0, 1, groups=(1, 0, 0))]
    m = ev.metric_values(rows)
    assert m["n"] == 2 and m["gtvp_dsc"] == pytest.approx(0.7)
    assert m["gtvn_dscagg"] == pytest.approx(2 * 80 / 200)
    assert m["f1_hungaro"] == pytest.approx(2 * 3 / (2 * 3 + 1 + 1))
    assert m["f1_grupos"] == pytest.approx(2 * 3 / (2 * 3 + 1 + 0))


def test_bootstrap_pareado():
    ref = [_row(f"C{i}", 0.6, 40, 100, 1, 1, 1) for i in range(30)]
    better = [_row(f"C{i}", 0.7, 45, 100, 2, 0, 0, down=300) for i in range(30)]
    c = ev.paired_bootstrap(ref, better, n_boot=200)
    assert c["gtvp_dsc"]["diferencia"] == pytest.approx(0.1)
    lo, hi = c["gtvp_dsc"]["ic95"]
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)  # la misma mejora en todos: sin incertidumbre
    assert c["f1_hungaro"]["diferencia"] > 0
    same = ev.paired_bootstrap(ref, ref, n_boot=50)
    assert all(v["diferencia"] == 0 for v in same.values())


def test_resumen_con_referencia_y_variantes():
    rows = []
    rng = np.random.default_rng(0)
    for i in range(20):
        d = float(rng.uniform(0.5, 0.8))
        for down in (270, 300):
            for restore in ("nearest", "linear"):
                bonus = 0.02 if restore == "linear" else 0.0
                rows.append(_row(f"C{i}", d + bonus, 40, 100, 1, 0, 0, down, restore))
    s = ev.summarize(rows, [270, 300], ["nearest", "linear"], n_boot=100)
    assert set(s["variantes"]) == {"270mm_nearest", "270mm_linear", "300mm_nearest", "300mm_linear"}
    assert "frente_a_referencia" not in s["variantes"]["270mm_nearest"]
    assert s["variantes"]["270mm_linear"]["frente_a_referencia"]["gtvp_dsc"]["diferencia"] == pytest.approx(0.02)
    assert s["variantes"]["300mm_nearest"]["frente_a_referencia"]["gtvp_dsc"]["diferencia"] == pytest.approx(0.0)


def test_reanudar_repite_el_caso_cortado_entre_vueltas(tmp_path):
    p = tmp_path / "casos.jsonl"
    full = [_row("A", 0.8, 50, 120, 2, 1, 0, restore=m) for m in ("nearest", "linear")]
    half = [_row("B", 0.7, 40, 100, 1, 0, 0)]  # el corte cayo despues de la vuelta cercana
    p.write_text("".join(json.dumps(r) + "\n" for r in full + half) + '{"case_id": "C', encoding="utf-8")
    rows = ev._read_jsonl(p)
    assert ev._done_keys(rows, ["nearest", "linear"]) == {("A", 270)}
    assert ev._done_keys(rows, ["nearest"]) == {("A", 270), ("B", 270)}
    ev._end_line(p)  # la linea a medias se cierra: la fila siguiente no se pega a ella
    ev._end_line(p)  # y no anade lineas vacias si ya termina en salto
    with open(p, "a", encoding="utf-8") as f:
        for m in ("nearest", "linear"):
            f.write(json.dumps(_row("B", 0.75, 40, 100, 1, 0, 0, restore=m)) + "\n")
    rows = ev._dedup(ev._read_jsonl(p))
    assert len(rows) == 4 and sorted(r["dice_gtvp"] for r in rows if r["case_id"] == "B") == [0.75, 0.75]
    assert ev._done_keys(rows, ["nearest", "linear"]) == {("A", 270), ("B", 270)}
    assert p.read_text(encoding="utf-8").count("\n\n") == 0


def test_metricas_de_un_caso():
    gt = np.zeros((10, 10, 10), np.uint8)
    gt[2:5, 2:5, 2:5] = 1
    gt[6:8, 6:8, 6:8] = 2
    pred = gt.copy()
    pred[6:8, 6:8, 6:7] = 0  # medio ganglio
    m = ev.case_metrics("X", pred, gt, 2.0)
    assert m["dice_gtvp"] == 1.0 and m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0
    assert m["gtvn_inter_mm3"] == 8.0 and m["gtvn_sum_mm3"] == 24.0
