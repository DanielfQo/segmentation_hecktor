"""Entrenamiento de las variantes de fusion con el mismo bucle de mambalite3d.

    python -m bifusion3d.train --config configs/siam_light.yaml --resume
    python -m bifusion3d.train --config configs/siam_light.yaml --resume --no-test

Acepta todas las opciones de mambalite3d.training.train (--name, --variant, --epochs, --patch, --resume, --fold...).
Opciones propias:
  --no-test        al terminar evalua solo la validacion; el test se evalua una vez, al final, con bifusion3d.evaluate.
  --thermal 83:72  pausa el entrenamiento si la GPU llega a 83 C y sigue al bajar a 72 C (laptop); con 83:72:120:10,
                   ademas descansa 10 min cada 120 min. Tambien se puede poner en el config: train.thermal.
"""
from __future__ import annotations

import sys

from . import patch, thermal
from .guard import refuse_inside_mamba


def thermal_from_config(argv) -> dict:
    """train.thermal del config que recibe --config (vacio si no hay)."""
    path = None
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            path = argv[i + 1]
        elif a.startswith("--config="):
            path = a.split("=", 1)[1]
    if not path:
        return {}
    from mambalite3d.config import load_config

    return dict(load_config(path).get("train", {}).get("thermal") or {})


def skip_test_evaluations(evaluate_fn):
    """Envuelve `evaluate` de mambalite3d.training.train para que no evalue los conjuntos de test."""

    def evaluate(model, cases, cfg, patch_zyx, device, out_dir, tag, log, *args, **kwargs):
        if str(tag).startswith("test"):
            log(f"[bifusion] evaluacion de test omitida ({tag}); evalua el test al final con bifusion3d.evaluate")
            return {}
        return evaluate_fn(model, cases, cfg, patch_zyx, device, out_dir, tag, log, *args, **kwargs)

    return evaluate


def labels_from_config(argv) -> str:
    """data.etiquetas del config que recibe --config (cercano si no hay)."""
    for i, a in enumerate(argv):
        path = argv[i + 1] if a == "--config" and i + 1 < len(argv) else (a.split("=", 1)[1] if a.startswith("--config=") else None)
        if path:
            from mambalite3d.config import load_config

            return str(load_config(path)["data"].get("etiquetas", "cercano"))
    return "cercano"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    no_test = "--no-test" in argv
    argv = [a for a in argv if a != "--no-test"]
    refuse_inside_mamba(argv, flags=())
    argv, thermal_cfg = thermal.config_from_argv(argv, thermal_from_config(argv))
    patch.install()
    import mambalite3d.training.train as base_train

    if thermal_cfg:
        thermal.install(thermal_cfg)
    if labels_from_config(argv) == "suave":
        from . import soft_train

        soft_train.install()
    if no_test:
        base_train.evaluate = skip_test_evaluations(base_train.evaluate)
    return base_train.main(argv)


if __name__ == "__main__":
    sys.exit(main())
