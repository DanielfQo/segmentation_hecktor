"""Simulacion de la cache rodante de mambalite3d a partir de un log sintetico."""
import json

from bifusion3d.tools import cache_coverage as cc


def write_run(tmp_path, sessions, n_cases=20, cached=6, step=2, epochs=12):
    run = tmp_path / "run"
    run.mkdir()
    ids = [f"SYN-{i:03d}" for i in range(n_cases)]
    cfg = {"experiment": "syn", "seed": 7,
           "data": {"max_cached_cases": cached, "refresh_per_epoch": step, "load_threads": 2},
           "train": {"iters_per_epoch": 5, "batch_size": 2, "epochs": epochs}}
    (run / "config_used.json").write_text(json.dumps(cfg), encoding="utf-8")
    (run / "split.json").write_text(json.dumps({"train": ids, "val": [], "test": []}), encoding="utf-8")
    lines = ["[00:00:00] arranque sin entrenar (no cuenta)"]
    for start, n in sessions:
        lines.append("[00:00:01] experimento syn  variante base  device cpu  mamba_ssm=False")
        for i in range(n):
            rec = {"epoch": start + i, "loss": 0.1, "cache_new_cases": 0 if i == 0 else step}
            lines.append(f"[00:00:02] epoca {json.dumps(rec)}")
        lines.append("[00:00:03] evaluacion [ep] {}")
    (run / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run, ids


def test_simulacion_exacta_y_cuentas(tmp_path):
    run, ids = write_run(tmp_path, [(0, 7), (6, 6)])  # la epoca 6 se repite tras un corte
    res = cc.analyze_run(run)
    assert [(s["desde_epoca"], s["epocas"]) for s in res["sesiones"]] == [(0, 7), (6, 6)]
    assert res["validacion"]["exacta"]
    real = res["real"]
    assert sum(real["epocas_por_caso"].values()) == 12 * 6  # 12 epocas efectivas x 6 casos en cache
    assert res["propuesta_ventana"]["sin_ver"] == 0


def test_cada_sesion_empieza_con_la_misma_cache(tmp_path):
    run, ids = write_run(tmp_path, [(0, 7), (7, 5)])
    sessions = cc.parse_sessions((run / "log.txt").read_text(encoding="utf-8"))
    eff, checks = cc.simulate_store(ids, sessions, {"max_cached_cases": 6, "refresh_per_epoch": 2, "load_threads": 2}, 7)
    assert eff[0] == eff[7]  # al reanudar, la cache vuelve a la de la epoca 0
    assert eff[1] == eff[0] and eff[2] != eff[1]  # el primer recambio solo encarga cargas; el segundo cambia 2 casos
    assert len(eff[2] - eff[1]) == 2
    assert checks["coinciden"] == checks["epocas"] == 12


def test_log_que_no_coincide_se_detecta(tmp_path):
    run, ids = write_run(tmp_path, [(0, 5)])
    text = (run / "log.txt").read_text(encoding="utf-8").replace('"cache_new_cases": 2', '"cache_new_cases": 3', 1)
    (run / "log.txt").write_text(text, encoding="utf-8")
    res = cc.analyze_run(run)
    assert not res["validacion"]["exacta"] and res["validacion"]["distintas"]
