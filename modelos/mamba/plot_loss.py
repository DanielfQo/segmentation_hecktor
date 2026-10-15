#!/usr/bin/env python3
"""Wrapper para ejecutar plot_loss.py desde la raíz de mamba."""
import sys
from pathlib import Path

runs_script = Path(__file__).resolve().parent / "runs" / "plot_loss.py"
if runs_script.exists():
    import runpy
    sys.argv[0] = str(runs_script)
    runpy.run_path(str(runs_script), run_name="__main__")
else:
    print(f"No se encontró {runs_script}")
    sys.exit(1)
