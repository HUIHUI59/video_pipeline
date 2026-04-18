#!/usr/bin/env bash
# Stage 5 辅助：只推代码（src/runpod/ + tools/pod_setup.sh + delivery_v1/ + runpod.yaml），
# 跳过 clips + manifest。用于迭代 pod_runner 代码时快速部署。
#
# 典型工作流：
#   改完本地 src/runpod/pod_runner.py → bash scripts/runpod/00_push_code.sh
#   → bash scripts/runpod/02_run.sh    # ssh 进 Pod 重启 runner
#
# 对比 01_push.sh：01 推 clips+manifest+代码（首次或数据变时用），00 只推代码。

set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/runpod.yaml}"
if [ ! -f "$CONFIG" ]; then
  echo "[ERR] 配置文件不存在: $CONFIG"
  echo "  提示：cp configs/runpod.yaml.example configs/runpod.yaml 再填真实 Pod 信息"
  exit 1
fi

python -m src.runpod.upload --config "$CONFIG" --code-only "${@:2}"
