#!/usr/bin/env bash
source "$(dirname "$0")/../entorno.sh"
echo "=== B1: B0 con etiquetas suaves (unas 15 h) ==="
python -m bifusion3d.train --config configs/b1_siam_full_suave.yaml --resume --no-test --thermal 83:72 || { echo "ERROR: entrenamiento"; exit 11; }
python -m bifusion3d.evaluate --run runs/b1_siam_full_suave --split val --checkpoint best
