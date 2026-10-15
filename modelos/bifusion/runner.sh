#!/usr/bin/env bash
# Ejecutor de trabajos para Linux: el equivalente de runner.bat.
# Ejecuta en orden alfabetico los .sh de jobs/pending, uno a la vez, y guarda la salida en jobs/logs.
# Si se corto (apagon, reinicio, Ctrl+C), al abrirlo de nuevo el trabajo a medias vuelve a la cola
# y continua con --resume. Uso:  ./runner.sh      (detener: Ctrl+C)
set -u
cd "$(dirname "$0")"
mkdir -p jobs/pending jobs/running jobs/done jobs/logs
exec 9>jobs/runner.lock
if ! flock -n 9; then echo "Ya hay otro ejecutor abierto en esta carpeta."; exit 1; fi
if pgrep -f "python.*(bifusion3d|mambalite3d)" >/dev/null; then
  echo "Ya hay un entrenamiento o una evaluacion en marcha (python con bifusion3d o mambalite3d)."; exit 1
fi
for f in jobs/running/*.sh; do
  [ -e "$f" ] || continue
  n=$(basename "$f" .sh)
  echo "[$(date '+%F %T')] $n quedo a medias: vuelve a la cola"
  echo "[$(date '+%F %T')] ejecutor reiniciado: $n vuelve a la cola y continua" >> "jobs/logs/$n.log"
  mv "$f" jobs/pending/
done
echo "Ejecutor de trabajos de modelos/bifusion (Ctrl+C para detenerlo)"
while true; do
  date '+%F %T' > jobs/runner_heartbeat.txt
  for f in $(ls jobs/pending/*.sh 2>/dev/null | sort); do
    n=$(basename "$f" .sh)
    echo "[$(date '+%F %T')] Iniciando $n"
    mv "$f" "jobs/running/$n.sh"
    date '+%F %T' > "jobs/logs/$n.started"
    echo "===== [$(date '+%F %T')] inicio de $n =====" >> "jobs/logs/$n.log"
    bash "jobs/running/$n.sh" >> "jobs/logs/$n.log" 2>&1
    rc=$?
    echo "$rc" > "jobs/logs/$n.exit"
    mv "jobs/running/$n.sh" "jobs/done/$n.sh"
    echo "[$(date '+%F %T')] Terminado $n con codigo $rc"
  done
  sleep 5
done
