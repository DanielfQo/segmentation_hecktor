#!/usr/bin/env bash
# Resumen del avance de la cola de experimentos, para mirarlo por SSH.
# Uso:  ./estado.sh          una vez
#       ./estado.sh -s       se refresca cada 60 s (Ctrl+C para salir)
set -uo pipefail
cd "$(dirname "$0")"

resumen() {
  echo "=== $(date '+%F %T')  $(hostname) ==="
  if [[ -e runs/.cola.lock ]] && ! flock -n runs/.cola.lock true 2>/dev/null; then
    echo "cola: CORRIENDO"
  else
    echo "cola: detenida"
  fi
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
             --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'
  echo
  ./cola_experimentos.sh --lista 2>/dev/null
  echo
  # experimento con el log mas reciente: ultima epoca, perdida y ETA
  local ult
  ult=$(ls -t runs/*/history.json 2>/dev/null | head -1)
  if [[ -n "$ult" ]]; then
    local run; run=$(dirname "$ult")
    echo "--- $run ---"
    python - "$ult" <<'PY' 2>/dev/null
import json, sys
h = json.load(open(sys.argv[1]))
if h:
    e = h[-1]
    print(f"epoca {e['epoch']}  loss {e.get('loss')}  {e.get('s_per_iter')} s/iter  "
          f"ETA {e.get('eta_h')} h  VRAM pico {e.get('peak_reserved_gb')} GB")
PY
    tail -5 "$run/log.txt" 2>/dev/null
  fi
}

if [[ "${1:-}" == "-s" ]]; then
  while true; do clear; resumen; sleep 60; done
else
  resumen
fi
