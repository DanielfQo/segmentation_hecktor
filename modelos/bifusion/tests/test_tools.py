"""Enganche con las herramientas de mambalite3d sin modificarlas."""
import os

import numpy as np
import pytest
import torch

import bifusion3d
from bifusion3d import network, patch
from bifusion3d.cross_validate import fold_command
from bifusion3d.guard import refuse_inside_mamba
from bifusion3d.train import skip_test_evaluations

from mambalite3d.config import load_config
from mambalite3d.models.plan import plan_network

SPACING_ZYX = [3.0, 1.0, 1.0]


def test_patch_installs_build_model_in_every_tool():
    patch.install()
    import importlib

    for name in patch.TARGETS:
        assert importlib.import_module(name).build_model is network.build_model, name


def test_checkpoint_roundtrip_through_the_evaluate_loader(tmp_path):
    """Un checkpoint de una variante de fusion se reconstruye con load_member de mambalite3d.tools.evaluate."""
    patch.install()
    from mambalite3d.tools.evaluate import load_member

    cfg = load_config(None)
    cfg["model"]["variant"] = "siam_light"
    patch_zyx = [int(p) for p in cfg["model"]["patch_size_zyx"]]
    strides, kernels = plan_network(patch_zyx, SPACING_ZYX)
    model = network.build_model("siam_light", strides, kernels, chunk_size=cfg["model"]["chunk_size"])
    torch.save({"model": model.state_dict(), "config": cfg, "strides": strides, "kernels": kernels},
               tmp_path / "checkpoint_final.pth")
    loaded, cfg_loaded = load_member(tmp_path, "final", torch.device("cpu"))
    assert type(loaded) is type(model) and cfg_loaded["model"]["variant"] == "siam_light"
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)


def test_efficiency_counts_the_siamese_stem():
    patch.install()
    from mambalite3d.tools.efficiency import count_ops

    patch_zyx = [32, 64, 64]
    strides, kernels = plan_network(patch_zyx, SPACING_ZYX)
    base = network.build_model("base", strides, kernels, chunk_size=16)
    siam = network.build_model("siam_light", strides, kernels, chunk_size=16)
    ops_base = count_ops(base, patch_zyx, torch.device("cpu"))
    ops_siam = count_ops(siam, patch_zyx, torch.device("cpu"))
    print(f"\nGMACs (patch {patch_zyx}): base {ops_base['gmacs']:.3f}, siam_light {ops_siam['gmacs']:.3f}")
    assert ops_siam["gmacs"] > ops_base["gmacs"]
    assert ops_siam["gmacs_barrido_mamba"] == pytest.approx(ops_base["gmacs_barrido_mamba"])


def test_no_test_flag_skips_only_the_test_evaluations():
    calls, logs = [], []

    def fake_evaluate(model, cases, cfg, patch_zyx, device, out_dir, tag, log, **kwargs):
        calls.append(tag)
        return {"tag": tag}

    wrapped = skip_test_evaluations(fake_evaluate)
    assert wrapped(None, None, None, None, None, None, "test_final", logs.append) == {}
    assert wrapped(None, None, None, None, None, None, "val_final", logs.append, per_center=True) == {"tag": "val_final"}
    assert calls == ["val_final"] and len(logs) == 1


def test_guard_refuses_to_write_inside_mamba():
    inside = os.path.join(bifusion3d.MAMBA_DIR, "runs", "full_base_fold0")
    with pytest.raises(SystemExit):
        refuse_inside_mamba(["--run", inside, "--split", "val"], check_cwd=False)
    with pytest.raises(SystemExit):
        refuse_inside_mamba([f"--run={inside}"], check_cwd=False)
    refuse_inside_mamba(["--run", os.path.join(bifusion3d.PROJECT_DIR, "runs", "x")], check_cwd=False)
    # leer miembros de un ensamble desde mamba esta permitido (no es --run)
    refuse_inside_mamba(["--ensemble", inside + ":best", "runs/x:final"], check_cwd=False)


def test_cross_validation_launches_bifusion_train_in_this_folder():
    cmd, cwd = fold_command("configs/siam_light_cv7.yaml", 3, ["--epochs", "2"])
    assert cmd[1:5] == ["-m", "bifusion3d.train", "--config", "configs/siam_light_cv7.yaml"]
    assert cmd[cmd.index("--fold") + 1] == "3" and "--resume" in cmd and cmd[-2:] == ["--epochs", "2"]
    assert os.path.samefile(cwd, bifusion3d.PROJECT_DIR)
    assert not bifusion3d.inside_mamba(cwd)


def test_configs_point_to_the_same_data_and_protocol_as_the_base():
    import glob

    import yaml

    base_cfg = load_config(os.path.join(bifusion3d.MAMBA_DIR, "configs", "full.yaml"))
    paths = sorted(glob.glob(os.path.join(bifusion3d.PROJECT_DIR, "configs", "*.yaml")))
    assert paths
    for path in paths:
        if "smoke" in os.path.basename(path):
            continue
        cfg = load_config(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert cfg["data"]["root"] == base_cfg["data"]["root"], path
        assert cfg["seed"] == base_cfg["seed"] and cfg["data"]["split"] == base_cfg["data"]["split"], path
        for key in ("epochs", "iters_per_epoch", "batch_size", "val_every", "val_cases_during_training"):
            assert cfg["train"][key] == base_cfg["train"][key], (path, key)
        if not os.path.basename(path).startswith("ctx"):  # los experimentos de contexto cambian patch y spacing
            assert cfg["model"]["patch_size_zyx"] == base_cfg["model"]["patch_size_zyx"], path
            assert cfg["data"]["target_spacing_xyz"] == base_cfg["data"]["target_spacing_xyz"], path
        variant = raw["model"]["variant"]
        assert variant in network.FUSION_VARIANTS or variant == "base", path


def test_sliding_window_and_ensemble_accept_fusion_models():
    """Paso 6 (votacion como cota superior): las variantes de fusion funcionan con la inferencia y EnsembleModel."""
    from mambalite3d.models.ensemble import EnsembleModel
    from mambalite3d.training.inference import sliding_window_predict

    patch_zyx = [16, 32, 32]
    strides, kernels = plan_network(patch_zyx, SPACING_ZYX, min_size=2)
    torch.manual_seed(0)
    siam = network.build_model("siam_light", strides, kernels, chunk_size=16)
    base = network.build_model("base", strides, kernels, chunk_size=16)
    image = np.random.default_rng(0).normal(size=(2, 20, 40, 36)).astype(np.float32)
    probs = sliding_window_predict(siam, image, patch_zyx, 3, torch.device("cpu"), amp=False, batch_size=2)
    assert probs.shape == (3, 20, 40, 36) and np.allclose(probs.sum(0), 1, atol=1e-4)
    assert siam.decoder.deep_supervision is True  # la inferencia la restaura al terminar
    probs_ens = sliding_window_predict(EnsembleModel([siam, base]), image, patch_zyx, 3, torch.device("cpu"), amp=False,
                                       batch_size=2)
    assert probs_ens.shape == probs.shape and np.allclose(probs_ens.sum(0), 1, atol=1e-4)
