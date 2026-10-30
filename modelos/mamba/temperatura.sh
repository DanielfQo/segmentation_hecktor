#!/usr/bin/env bash
# Registro historico de temperatura de la GPU durante la cola de experimentos.
#   ./temperatura.sh             # muestrea cada 60 s en runs/temperatura.csv (lanzar con nohup)
#   ./temperatura.sh --resumen   # max, media y ultimas lecturas
#   ./temperatura.sh --intervalo 30
set -uo pipefail
cd "$(dirname "$0")"
CSV=runs/temperatura.csv
intervalo=60

case "${1:-}" in
  --resumen)
    [[ -s $CSV ]] || { echo "aun no hay lecturas en $CSV"; exit 0; }
    echo "lecturas: $(( $(wc -l < "$CSV") - 1 ))  desde $(sed -n 2p "$CSV" | cut -d, -f1)"
    awk -F, 'NR>1 {t=$2; s+=t; n++; if (t>mx) {mx=t; mxh=$1}; if (mn=="" || t<mn) mn=t}
             END {printf "temperatura  max %d C (%s)   media %.1f C   min %d C\n", mx, mxh, s/n, mn}' "$CSV"
    awk -F, 'NR>1 && $6 != "0x0000000000000000" {n++} END {printf "muestras con throttle activo: %d\n", n+0}' "$CSV"
    echo "ultimas 5:"; tail -5 "$CSV" | column -t -s,
    exit 0 ;;
  --intervalo) intervalo="${2:?falta los segundos}" ;;
esac

mkdir -p runs
[[ -s $CSV ]] || echo "hora,temp_c,util_pct,vram_mib,clock_sm_mhz,throttle" > "$CSV"
while true; do
  linea=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,clocks.sm,clocks_throttle_reasons.active \
          --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [[ -n $linea ]] && echo "$(date '+%F %T'),$linea" >> "$CSV"
  sleep "$intervalo"
done
