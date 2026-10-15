"""Guardia de temperatura: pausas, descansos programados y enganche al bucle de mambalite3d (con lecturas falsas)."""
import pytest

from bifusion3d import thermal


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def reader_from(values):
    it = iter(values)
    return lambda: next(it, None)


def test_pausa_hasta_bajar_y_sigue():
    clock, logs = FakeClock(), []
    g = thermal.ThermalGuard({"max_c": 83, "resume_c": 72, "every": 1, "poll_s": 30}, log=logs.append,
                             reader=reader_from([60, 85, 80, 74, 71, 70]), sleep=clock.sleep, clock=clock)
    g.tick()  # lee 85: pausa; 80 y 74 siguen altas; 71 reanuda
    assert g.pauses == 1 and clock.t == 90 and g.paused_s == 90
    g.tick()  # lee 70: sin pausa
    assert g.pauses == 1
    assert any("pausa hasta" in m for m in logs) and any("sigue" in m for m in logs)


def test_solo_mira_cada_every_llamadas():
    calls = []

    def reader():
        calls.append(1)
        return 50

    g = thermal.ThermalGuard({"every": 25}, log=lambda m: None, reader=reader, sleep=lambda s: None)
    for _ in range(100):
        g.tick()
    assert len(calls) == 1 + 4  # la lectura inicial y una cada 25 llamadas
    g.tick(force=True)
    assert len(calls) == 6


def test_sin_nvidia_smi_se_apaga_pero_descansa():
    clock, logs = FakeClock(), []
    g = thermal.ThermalGuard({"rest_every_min": 60, "rest_min": 5}, log=logs.append, reader=lambda: None,
                             sleep=clock.sleep, clock=clock)
    assert not g.enabled and any("guardia apagada" in m for m in logs)
    clock.t = 59 * 60
    g.tick()
    assert g.paused_s == 0
    clock.t = 61 * 60
    g.tick()
    assert g.paused_s == 300 and clock.t == 61 * 60 + 300


def test_el_lector_por_defecto_se_busca_al_llamar(monkeypatch):
    reads = []
    monkeypatch.setattr(thermal, "gpu_temperature", lambda: reads.append(1) or 50)
    g = thermal.ThermalGuard({"every": 1}, log=lambda m: None, sleep=lambda s: None)
    g.tick()
    assert g.enabled and len(reads) == 2  # lectura inicial + la del tick, ambas con la funcion sustituida


def test_spec_y_argv():
    cfg = thermal.parse_spec("83:72:120:10")
    assert (cfg["max_c"], cfg["resume_c"], cfg["rest_every_min"], cfg["rest_min"]) == (83, 72, 120, 10)
    with pytest.raises(ValueError):
        thermal.parse_spec("72:83")
    argv, cfg = thermal.config_from_argv(["--config", "a.yaml", "--thermal", "80:70", "--resume"], None)
    assert argv == ["--config", "a.yaml", "--resume"] and cfg["max_c"] == 80
    argv, cfg = thermal.config_from_argv(["--resume"], {"max_c": 85})
    assert argv == ["--resume"] and cfg["max_c"] == 85 and cfg["resume_c"] == thermal.DEFAULTS["resume_c"]
    assert thermal.config_from_argv(["--resume"], None) == (["--resume"], None)


def test_install_engancha_el_muestreo_y_la_inferencia(monkeypatch):
    pytest.importorskip("torch")
    import mambalite3d.training.train as mtrain

    monkeypatch.setattr(mtrain, "PatchSampler", mtrain.PatchSampler)  # se restauran al terminar la prueba
    monkeypatch.setattr(mtrain, "sliding_window_predict", mtrain.sliding_window_predict)
    reads = []
    monkeypatch.setattr(thermal, "gpu_temperature", lambda: reads.append(1) or 50)
    guard = thermal.install({"every": 1}, log=lambda m: None)
    assert mtrain.PatchSampler.__name__ == "GuardedSampler"
    assert getattr(mtrain.sliding_window_predict, "__wrapped__", None) is not None

    class Store:
        def cases(self):
            raise RuntimeError("muestreo alcanzado")

    with pytest.raises(RuntimeError, match="muestreo alcanzado"):
        mtrain.PatchSampler(Store(), (8, 8, 8)).sample(2)
    assert guard.calls == 1 and len(reads) == 2  # lectura inicial + la del tick
