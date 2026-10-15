#!/usr/bin/env bash
# Foreground terminal runner. Linux only; no package changes or remote access.
set -euo pipefail
trap 'TASK25_EXIT_CODE=$?; printf "\n[已退出] Stage25 launcher 已结束，退出码=%s。\n" "$TASK25_EXIT_CODE"' EXIT
TASK25_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$TASK25_REPO"
for TASK25_FILE in tools/run_plastic_bomo_stage25_msdf_v2.py configs/plastic_bomo_stage25_server_code_manifest.json; do
  test -f "$TASK25_FILE" || { printf '[启动失败] 缺少文件：%s\n' "$TASK25_FILE" >&2; exit 1; }
done
TASK25_PYTHON=${STAGE25_PYTHON:-/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python}
test -x "$TASK25_PYTHON" || { printf '[启动失败] Python 不存在或不能执行：%s。请设置 STAGE25_PYTHON，不要安装/升级环境。\n' "$TASK25_PYTHON" >&2; exit 1; }
TASK25_MODE=${1:-all}
if [ "$#" -gt 0 ]; then shift; fi
TASK25_STAMP=$(date +%Y%m%d_%H%M%S_%N)
TASK25_LOGROOT="$TASK25_REPO/logs/plastic_bomo_stage25_msdf_v2_requalification"
mkdir -p -- "$TASK25_LOGROOT"
TASK25_LOG="$TASK25_LOGROOT/launcher_${TASK25_STAMP}.log"
printf '%s\n' "$$" > "$TASK25_LOGROOT/launcher_${TASK25_STAMP}.pid"
printf '[前台运行] PID=%s\nLOG=%s\nREPO=%s\n按 Ctrl+C 可停止本次任务；不要重复启动。\n' "$$" "$TASK25_LOG" "$TASK25_REPO"
env CUDA_VISIBLE_DEVICES="${STAGE25_GPU:-3}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$TASK25_PYTHON" -u tools/run_plastic_bomo_stage25_msdf_v2.py \
  --mode "$TASK25_MODE" --device 0 "$@" 2>&1 | tee "$TASK25_LOG"
