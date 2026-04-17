#!/usr/bin/env bash
# Stage 5 step 2/3: SSH 进 Pod，跑 pod_setup（首次）+ pod_runner（可重复）。
#   - 会实时 tail pod 上的日志到本地终端
#   - Ctrl+C 退出本地 tail 不影响 Pod 进程（Pod 里是 nohup + 日志文件）
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/runpod.yaml}"
if [ ! -f "$CONFIG" ]; then
  echo "[ERR] 配置文件不存在: $CONFIG"; exit 1
fi

# 从 yaml 里抽 Pod 连接信息
eval "$(python - <<PY
import yaml, os
c = yaml.safe_load(open("$CONFIG"))
p = c["pod"]
print(f"POD_HOST={p['host']}")
print(f"POD_PORT={p['port']}")
print(f"POD_USER={p['user']}")
print(f"POD_KEY={os.path.expanduser(p['ssh_key'])}")
print(f"POD_WS={c['paths']['pod_workspace']}")
PY
)"

echo "  Pod: ${POD_USER}@${POD_HOST}:${POD_PORT}  workspace=${POD_WS}"

# 远端命令：如果 pod_setup 标志文件不存在就跑一次安装，然后后台跑 runner
REMOTE_CMD="set -e; cd '${POD_WS}'; \
  if [ ! -f .pod_setup_done ]; then \
    echo '首次 setup 环境...'; bash tools/pod_setup.sh && touch .pod_setup_done; \
  fi; \
  mkdir -p output; \
  echo '启动 pod_runner（nohup，断开 SSH 不影响）'; \
  nohup python src/runpod/pod_runner.py --config runpod.yaml \
    >> output/pod_runner.stdout 2>&1 & \
  RUNNER_PID=\$!; \
  echo 'runner pid ='\$RUNNER_PID; \
  echo 'tail -f output/pod_runner.log (Ctrl+C 退出 tail 不杀 runner)'; \
  sleep 2; \
  tail -f output/pod_runner.log 2>/dev/null || true"

ssh -i "$POD_KEY" -p "$POD_PORT" \
    -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    "${POD_USER}@${POD_HOST}" "$REMOTE_CMD"
