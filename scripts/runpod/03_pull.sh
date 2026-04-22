#!/usr/bin/env bash
# Stage 5 step 3/3: 从 Pod rsync 回标注结果 + schema 校验。
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/runpod.yaml}"
if [ ! -f "$CONFIG" ]; then
  echo "[ERR] 配置文件不存在: $CONFIG"; exit 1
fi

echo "[pull] config = $CONFIG"
echo "[pull] starting download.py (rsync labels back → validate schemas)"
START_TS=$(date +%s)

python -u -m src.runpod.download --config "$CONFIG" "${@:2}"

ELAPSED=$(( $(date +%s) - START_TS ))
echo "[pull] done in ${ELAPSED}s"
