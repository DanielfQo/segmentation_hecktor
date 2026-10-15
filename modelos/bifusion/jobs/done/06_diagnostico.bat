@echo off
echo === GPU ===
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,pstate --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo === procesos python ===
tasklist /fi "imagename eq python.exe" /fo csv
echo === ventanas de ejecutores (cmd) ===
tasklist /v /fi "imagename eq cmd.exe" /fo csv
echo === energia (solo lectura) ===
powercfg /getactivescheme
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
exit /b 0
