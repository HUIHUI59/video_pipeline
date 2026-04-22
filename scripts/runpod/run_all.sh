#!/usr/bin/env bash
# Stage 5 一键跑：push → run → pull
#   02_run.sh 在 tail 期间会阻塞；Ctrl+C 后再跑 03_pull.sh
#
# 用法：
#   bash run_all.sh [<config_path>] [extra upload.py args...]
#   - $1 = config YAML 文件路径，默认 configs/runpod.yaml
#   - 之后所有参数会原样转发给 01_push.sh → upload.py
#     例：bash run_all.sh configs/runpod.yaml --include-bad-quality --max-shots 50
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/runpod.yaml}"
shift || true              # 把 $1 弹掉，"$@" 现在只剩 upload.py extra args

echo "══ 1/3 push ══"
bash scripts/runpod/01_push.sh "$CONFIG" "$@"

echo
echo "══ 2/3 run (tail 日志；跑完后 Ctrl+C 继续 pull) ══"
bash scripts/runpod/02_run.sh "$CONFIG" || true   # 允许用 Ctrl+C 打断 tail

echo
echo "══ 3/3 pull ══"
bash scripts/runpod/03_pull.sh "$CONFIG"
