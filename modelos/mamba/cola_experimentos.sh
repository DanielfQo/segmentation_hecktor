#!/usr/bin/env bash
# Cola de experimentos de MambaLite3D para Linux (Ubuntu), ordenada del mas al menos importante.
# Ejecuta uno por uno; cada experimento entrena (reanudable), evalua val y test con el checkpoint final
# y aplica el post-procesado ajustado en validacion, el mismo protocolo que base y wide.
#
#   ./cola_experimentos.sh --lista               # muestra el orden y el estado de cada experimento
#   ./cola_experimentos.sh                       # corre los que falten, en orden
#   ./cola_experimentos.sh --solo exp_dual       # corre solo los nombrados
#   ./cola_experimentos.sh --desde arq_m3lite    # empieza en ese experimento
#   nohup ./cola_experimentos.sh > runs/cola.log 2>&1 &    # en segundo plano, sobrevive al cerrar la terminal
#
# Es seguro relanzarlo: salta lo que ya termino y reanuda el entrenamiento desde la ultima epoca guardada.
# Entorno conda: MAMBALITE_ENV (por defecto mambalite3d_cuda, con mamba_ssm). Los checkpoints entrenados
# aqui solo cargan en ese entorno; los de full_base_fold0 (PyTorch puro) solo cargan en mambalite3d.
set -uo pipefail
cd "$(dirname "$0")"

# nombre | argumentos de train | horas aproximadas (entrenamiento + evaluacion) | por que importa
EXPERIMENTOS=(
  # fase 1: arquitectura
  "exp_dual|--config configs/exp_dual.yaml|8|arquitectura: doble codificador, fusion en cada nivel, base (prioridad)"
  "arq_dualmid|--config configs/full.yaml --variant dualmid --name arq_dualmid|8|arquitectura: fusion intermedia CT/PET, base"
  "full_base_heldout_chup|--config configs/full_heldout_chup.yaml --name full_base_heldout_chup|6|generalizacion a un centro no visto"
  "arq_m3lite|--config configs/full.yaml --variant m3lite --name arq_m3lite|6|arquitectura: Mamba tambien en el nivel 3, base"
  "arq_m4lite|--config configs/full.yaml --variant m4lite --name arq_m4lite|7|arquitectura: Mamba tambien en los niveles 2 y 3, base"
  "exp_dualwide|--config configs/exp_dualwide.yaml|14|arquitectura: doble codificador, wide (prioridad)"
  # despues: experimentos que no son ablaciones
  "exp_suv|--config configs/exp_suv.yaml|6|PET en SUV absoluto (ganglios falsos)"
  "full_ctx2mm|--config configs/exp_ctx2mm.yaml|7|mas contexto con parche 48x96x96 a 2 mm"
  "exp_patch160|--config configs/exp_patch160.yaml|15|mas contexto con parche 48x160x160"
  "full_base_e500|--config configs/full.yaml --epochs 500 --name full_base_e500|11|500 epocas (base)"
)
# Ablaciones (sin Mamba, sin CGA3D, mismo barrido en todos los grupos): fuera de la cola hasta elegir la
# arquitectura final, porque deben hacerse sobre ese modelo y no sobre base.

campo() { echo "$1" | cut -d'|' -f"$2"; }
nombre_de() { campo "$1" 1; }
terminado() { [[ -f "runs/$1/metrics_test_final_pp.json" ]]; }

estado() {
  local n="$1"
  if terminado "$n"; then echo "hecho"
  elif [[ -f "runs/$n/done.json" ]]; then echo "entrenado, falta evaluar"
  elif [[ -f "runs/$n/checkpoint_latest.pth" ]]; then echo "en curso (reanuda)"
  else echo "pendiente"; fi
}

listar() {
  local i=0 total=0
  printf '%2s  %-22s %-25s %5s  %s\n' "#" "experimento" "estado" "horas" "motivo"
  for e in "${EXPERIMENTOS[@]}"; do
    i=$((i + 1)); n=$(nombre_de "$e"); h=$(campo "$e" 3)
    printf '%2d  %-22s %-25s %5s  %s\n' "$i" "$n" "$(estado "$n")" "~$h" "$(campo "$e" 4)"
    terminado "$n" || total=$((total + h))
  done
  echo "Horas aproximadas pendientes: ~$total (con mamba_ssm; ctx2mm y patch160 estan por medir)"
}

correr() {
  local e="$1" n args
  n=$(nombre_de "$e"); args=$(campo "$e" 2)
  if terminado "$n"; then echo "[$(date +%T)] $n ya esta hecho, se salta"; return 0; fi
  echo "[$(date +%T)] ===== $n ====="
  mkdir -p runs
  {
    if [[ ! -f "runs/$n/done.json" ]]; then
      # shellcheck disable=SC2086
      python -m mambalite3d.training.train $args --resume || return 11
    fi
    # train ya deja metrics_{val,test}_final.json; analyze hace falta igual para ajustar el post-procesado
    for split in val test; do
      [[ -f "runs/$n/components_${split}_final.json" ]] ||
        python -m mambalite3d.tools.analyze --run "runs/$n" --split "$split" --checkpoint final || return 12
      [[ -f "runs/$n/metrics_${split}_final.json" ]] ||
        python -m mambalite3d.tools.evaluate --run "runs/$n" --split "$split" --checkpoint final || return 13
    done
    [[ -f "runs/$n/postprocess_final.json" ]] || python -m mambalite3d.tools.tune_postprocess --run "runs/$n" --checkpoint final || return 14
    python -m mambalite3d.tools.evaluate --run "runs/$n" --split test --checkpoint final \
        --postprocess "runs/$n/postprocess_final.json" || return 15
  } 2>&1 | tee -a "runs/$n.log"
  return "${PIPESTATUS[0]}"
}

# ---------------- argumentos ----------------
modo=correr; desde=""; solo=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lista) modo=lista ;;
    --desde) desde="${2:?falta el nombre}"; shift ;;
    --solo) shift; while [[ $# -gt 0 && "$1" != --* ]]; do solo+=("$1"); shift; done; continue ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "argumento desconocido: $1" >&2; exit 2 ;;
  esac
  shift
done
if [[ $modo == lista ]]; then listar; exit 0; fi

# ---------------- entorno ----------------
if ! command -v conda >/dev/null 2>&1; then
  for raiz in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" /opt/conda; do
    [[ -f "$raiz/etc/profile.d/conda.sh" ]] && { source "$raiz/etc/profile.d/conda.sh"; break; }
  done
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "${MAMBALITE_ENV:-mambalite3d_cuda}" || { echo "no se pudo activar el entorno ${MAMBALITE_ENV:-mambalite3d_cuda}" >&2; exit 1; }
export PYTHONUTF8=1
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "PyTorch no ve la GPU" >&2; exit 1; }

# una sola cola a la vez, y sin suspender el equipo mientras corre
mkdir -p runs
exec 9>runs/.cola.lock
flock -n 9 || { echo "ya hay una cola corriendo" >&2; exit 1; }
# systemd-inhibit solo funciona dentro de una sesion de logind: lanzada con nohup/setsid o desde cron
# polkit la deniega, asi que se comprueba antes y, si no se puede, la cola sigue igual (sin la proteccion).
if [[ -z "${COLA_INHIBIDA:-}" ]] && command -v systemd-inhibit >/dev/null 2>&1; then
  export COLA_INHIBIDA=1
  if systemd-inhibit --what=sleep:idle --who=mambalite3d --why="comprobacion" true 2>/dev/null; then
    exec systemd-inhibit --what=sleep:idle --who=mambalite3d --why="entrenamiento" "$0" "$@"
  fi
  echo "aviso: no se pudo inhibir la suspension (sin sesion de logind); la cola sigue sin ella" >&2
fi

listar
fallos=()
activo=0; [[ -z "$desde" ]] && activo=1
for e in "${EXPERIMENTOS[@]}"; do
  n=$(nombre_de "$e")
  [[ $n == "$desde" ]] && activo=1
  (( activo )) || continue
  if (( ${#solo[@]} )); then [[ " ${solo[*]} " == *" $n "* ]] || continue; fi
  correr "$e" || { rc=$?; echo "[$(date +%T)] FALLO $n (codigo $rc); sigue con el siguiente" >&2; fallos+=("$n"); }
done
echo "[$(date +%T)] cola terminada"
(( ${#fallos[@]} )) && { echo "fallaron: ${fallos[*]}" >&2; exit 1; }
exit 0
