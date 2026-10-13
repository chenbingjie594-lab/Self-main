#!/usr/bin/env bash
# Linux only. Audit first; never install/upgrade packages, never SSH elsewhere.
set -euo pipefail
TASK25_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$TASK25_REPO"
test -f tools/run_plastic_bomo_stage25_msdf_v2.py
test -f configs/plastic_bomo_stage25_server_code_manifest.json
TASK25_PYTHON=${STAGE25_PYTHON:-/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python}
test -x "$TASK25_PYTHON" || { echo 'Set STAGE25_PYTHON to the existing Stage20B Linux Python; no installation.' >&2; exit 1; }
TASK25_MODE=${1:-all}
if [ "$#" -gt 0 ]; then shift; fi
TASK25_STAMP=$(date +%Y%m%d_%H%M%S_%N)
TASK25_LOGROOT="$TASK25_REPO/logs/plastic_bomo_stage25_msdf_v2_requalification"
mkdir -p -- "$TASK25_LOGROOT"
TASK25_LOG="$TASK25_LOGROOT/launcher_${TASK25_STAMP}.log"
nohup env CUDA_VISIBLE_DEVICES="${STAGE25_GPU:-3}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$TASK25_PYTHON" -u tools/run_plastic_bomo_stage25_msdf_v2.py \
  --mode "$TASK25_MODE" --device 0 "$@" > "$TASK25_LOG" 2>&1 &
TASK25_PID=$!
printf '%s\n' "$TASK25_PID" > "$TASK25_LOGROOT/launcher_${TASK25_STAMP}.pid"
printf 'PID=%s\nLOG=%s\nREPO=%s\n' "$TASK25_PID" "$TASK25_LOG" "$TASK25_REPO"
