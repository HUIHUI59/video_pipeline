#!/usr/bin/env bash
# Stage 5 一键跑：push → run → pull
#   02_run.sh 在 tail 期间会阻塞；Ctrl+C 后再跑 03_pull.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/runpod.yaml}"

echo "══ 1/3 push ══"
bash scripts/runpod/01_push.sh "$CONFIG"

echo
echo "══ 2/3 run (tail 日志；跑完后 Ctrl+C 继续 pull) ══"
bash scripts/runpod/02_run.sh "$CONFIG" || true   # 允许用 Ctrl+C 打断 tail

echo
echo "══ 3/3 pull ══"
bash scripts/runpod/03_pull.sh "$CONFIG"
