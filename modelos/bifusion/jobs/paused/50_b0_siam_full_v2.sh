#!/usr/bin/env bash
source "$(dirname "$0")/../entorno.sh"
echo "=== B0: siam_full con ROI 300 mm, ancla relativa y etiquetas por ganglio (unas 15 h) ==="
python -m bifusion3d.train --config configs/b0_siam_full_v2.yaml --resume --no-test --thermal 83:72 || { echo "ERROR: entrenamiento"; exit 11; }
python -m bifusion3d.evaluate --run runs/b0_siam_full_v2 --split val --checkpoint best
