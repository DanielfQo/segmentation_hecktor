"""Pausas por temperatura para entrenar y evaluar en una laptop que se calienta.

La laptop se apaga a veces por calor. Con esta guardia el entrenamiento y la evaluacion esperan cuando la GPU llega a
`max_c` grados y siguen al bajar a `resume_c`. Opcionalmente descansan `rest_min` minutos cada `rest_every_min`.
Un apagado igual puede pasar (la CPU tambien calienta): por eso cada epoca deja su checkpoint y el runner v2 vuelve a
poner en cola el trabajo cortado.

La temperatura se lee con nvidia-smi. Si no esta o falla, la guardia lo dice una vez y se apaga sola; nunca detiene
el entrenamiento por un error suyo.

    python -m bifusion3d.train --config configs/siam_full.yaml --resume --thermal 83:72
    python -m bifusion3d.train --config ... --thermal 83:72:120:10      # ademas, 10 min de descanso cada 120 min

o en el config:

    train:
      thermal: {max_c: 83, resume_c: 72, every: 25, rest_every_min: 0, rest_min: 0}
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Callable, Optional

DEFAULTS = {"max_c": 83, "resume_c": 72, "every": 25, "rest_every_min": 0, "rest_min": 0, "poll_s": 30, "report_min": 15}


def _nvidia_smi() -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if exe is None and os.name == "nt":
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        exe = cand if os.path.exists(cand) else None
    return exe


def gpu_temperature() -> Optional[int]:
    """Temperatura de la GPU 0 en grados C, o None si no se puede leer."""
    exe = _nvidia_smi()
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def parse_spec(spec: str) -> dict:
    """'83:72' o '83:72:120:10' -> max_c, resume_c[, rest_every_min, rest_min]."""
    parts = [float(p) for p in str(spec).split(":") if p != ""]
    keys = ("max_c", "resume_c", "rest_every_min", "rest_min")
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in zip(keys, parts)})
    if cfg["resume_c"] >= cfg["max_c"]:
        raise ValueError(f"--thermal {spec}: la temperatura de reanudar debe ser menor que la de pausa")
    return cfg


class ThermalGuard:
    def __init__(self, cfg: Optional[dict] = None, log: Callable[[str], None] = print,
                 reader: Optional[Callable[[], Optional[int]]] = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        if reader is None:  # se busca al llamar, no al definir la clase, para que las pruebas puedan sustituirla
            def reader() -> Optional[int]:
                return gpu_temperature()
        self.log, self.reader, self.sleep, self.clock = log, reader, sleep, clock
        self.calls = 0
        self.enabled = True
        self.paused_s = 0.0
        self.pauses = 0
        self.last_rest = clock()
        self.last_report = clock()
        self.max_seen: Optional[int] = None
        t = reader()
        if t is None:
            self.enabled = False
            log("[termico] no se pudo leer la temperatura con nvidia-smi: guardia apagada (solo descansos programados)")
        else:
            self.max_seen = t
            log(f"[termico] guardia activa: pausa a {self.cfg['max_c']:.0f} C, sigue a {self.cfg['resume_c']:.0f} C; GPU ahora a {t} C")

    def _rest(self) -> None:
        every = float(self.cfg["rest_every_min"] or 0)
        if every <= 0 or self.clock() - self.last_rest < every * 60:
            return
        minutes = float(self.cfg["rest_min"] or 0)
        if minutes > 0:
            self.log(f"[termico] descanso programado de {minutes:.0f} min")
            self.sleep(minutes * 60)
            self.paused_s += minutes * 60
        self.last_rest = self.clock()

    def tick(self, force: bool = False) -> None:
        """Llamar una vez por iteracion (o por caso con force=True): cada `every` llamadas mira la temperatura."""
        self.calls += 1
        self._rest()
        if not self.enabled or (not force and self.calls % int(self.cfg["every"])):
            return
        t = self.reader()
        if t is None:
            return
        self.max_seen = t if self.max_seen is None else max(self.max_seen, t)
        if self.clock() - self.last_report >= float(self.cfg["report_min"]) * 60:
            self.log(f"[termico] GPU a {t} C (maxima {self.max_seen} C, {self.pauses} pausas, {self.paused_s / 60:.1f} min parado)")
            self.last_report = self.clock()
        if t < self.cfg["max_c"]:
            return
        self.pauses += 1
        start = self.clock()
        self.log(f"[termico] GPU a {t} C: pausa hasta bajar a {self.cfg['resume_c']:.0f} C")
        while t is not None and t > self.cfg["resume_c"]:
            self.sleep(float(self.cfg["poll_s"]))
            t = self.reader()
        waited = self.clock() - start
        self.paused_s += waited
        self.log(f"[termico] sigue: GPU a {t} C tras {waited / 60:.1f} min de pausa")


def install(cfg: dict, log: Callable[[str], None] = print) -> ThermalGuard:
    """Engancha la guardia al bucle de mambalite3d sin modificarlo: antes de cada lote de entrenamiento y de cada caso
    que se evalua o se analiza. Solo en memoria y solo en este proceso."""
    import mambalite3d.training.train as mtrain

    guard = ThermalGuard(cfg, log=log)
    base_sampler = mtrain.PatchSampler

    class GuardedSampler(base_sampler):
        def sample(self, batch_size):
            guard.tick()
            return super().sample(batch_size)

    mtrain.PatchSampler = GuardedSampler

    def guarded(fn):
        def wrapper(*args, **kwargs):
            guard.tick(force=True)
            return fn(*args, **kwargs)

        wrapper.__wrapped__ = fn
        return wrapper

    mtrain.sliding_window_predict = guarded(mtrain.sliding_window_predict)
    try:
        import mambalite3d.tools.analyze as manalyze

        manalyze.sliding_window_predict = guarded(manalyze.sliding_window_predict)
    except ImportError:
        pass
    return guard


def config_from_argv(argv: list, cfg_thermal: Optional[dict]) -> tuple:
    """Quita --thermal SPEC de argv. Devuelve (argv sin el flag, config de la guardia o None)."""
    out, spec, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--thermal" and i + 1 < len(argv):
            spec = argv[i + 1]
            i += 2
            continue
        if a.startswith("--thermal="):
            spec = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    if spec is not None:
        return out, parse_spec(spec)
    if cfg_thermal:
        return out, {**DEFAULTS, **cfg_thermal}
    return out, None
