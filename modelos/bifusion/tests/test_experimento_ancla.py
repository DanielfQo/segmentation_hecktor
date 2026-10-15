"""Experimento del ancla relativa sobre casos sinteticos."""
import json

from bifusion3d.tools import experimento_ancla as ea
from mambalite3d.config import load_config
from test_audit import make_case
from test_experimento_preproc import _scale_pet


def test_pet_fuera_de_suv_solo_tiene_ancla_relativa(tmp_path):
    files = make_case(tmp_path, "SYN-004", node_below=False)
    _scale_pet(files, 0.02)
    r = ea.ancla_case(files, load_config(None)["data"], "train")
    assert r["absoluta"]["ancla"] is False and r["relativa"]["ancla"] is True
    assert r["relativa"]["alto_mm"] < r["absoluta"]["alto_mm"] and r["mueve_max_mm"] > 0
    json.dumps(r)


def test_main_y_resumen(tmp_path):
    root = tmp_path / "datos"
    make_case(root, "SYN-001")
    make_case(root, "SYN-002", face=False, node_below=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"seed: 1\ndata:\n  root: {json.dumps(root.as_posix())}\n", encoding="utf-8")
    out = tmp_path / "out"
    assert ea.main(["--config", str(cfg), "--out", str(out), "--workers", "1"]) == 0
    s = json.loads((out / "resumen.json").read_text(encoding="utf-8"))
    assert s["n_casos"] == 2 and not s["errores"] and s["sin_ancla"]["relativa"] == []
    assert ea.main(["--config", str(cfg), "--out", str(out), "--workers", "1"]) == 0
    assert len((out / "casos.jsonl").read_text(encoding="utf-8").splitlines()) == 2
