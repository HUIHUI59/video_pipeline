#!/usr/bin/env bash
# Stage 5 step 1/3: 把筛选后的 clips + manifest rsync 到 Runpod Pod。
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/runpod.yaml}"
if [ ! -f "$CONFIG" ]; then
  echo "[ERR] 配置文件不存在: $CONFIG"
  echo "  提示：cp configs/runpod.yaml.example configs/runpod.yaml 再填真实 Pod 信息"
  exit 1
fi

echo "[push] config = $CONFIG"
echo "[push] starting upload.py (filter shots → write manifest → rsync clips → rsync code)"
START_TS=$(date +%s)

python -u -m src.runpod.upload --config "$CONFIG" "${@:2}"

ELAPSED=$(( $(date +%s) - START_TS ))
echo "[push] done in ${ELAPSED}s"
