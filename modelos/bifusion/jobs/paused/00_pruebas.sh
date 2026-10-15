#!/usr/bin/env bash
source "$(dirname "$0")/../entorno.sh"
echo "=== Pruebas de bifusion ==="
python -m pytest -q tests
