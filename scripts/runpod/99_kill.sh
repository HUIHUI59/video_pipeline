#!/usr/bin/env bash
# Stage 5 清理脚本：ssh 进 Pod，一把杀掉所有 runner / vllm / 残留进程，释放 GPU。
#
# 典型用法：
#   bash scripts/runpod/99_kill.sh                    # 默认 configs/runpod.yaml
#   bash scripts/runpod/99_kill.sh configs/other.yaml
#   bash scripts/runpod/99_kill.sh --clean-logs       # 顺手清掉旧 stdout/log
#
# 为什么需要：
#   - `pkill -f pod_runner` 只杀主进程，vLLM fork 出来的 EngineCore / worker 还占 GPU
#   - CUDA IPC / NCCL 线程有时不立刻释放显存
#   - 这个脚本用 nvidia-smi --query-compute-apps 把所有占 GPU 的 PID 一锅端
set -euo pipefail
cd "$(dirname "$0")/../.."

CLEAN_LOGS=0
CONFIG=""
for arg in "$@"; do
  case "$arg" in
    --clean-logs) CLEAN_LOGS=1 ;;
    *)            CONFIG="${CONFIG:-$arg}" ;;
  esac
done
CONFIG="${CONFIG:-configs/runpod.yaml}"

if [ ! -f "$CONFIG" ]; then
  echo "[ERR] 配置文件不存在: $CONFIG"; exit 1
fi

# 从 yaml 抽 Pod 连接信息（和 02_run.sh 保持一致）
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

REMOTE_CMD="
set +e
echo '── [1/4] 杀 pod_runner 主进程'
pkill -9 -f pod_runner.py 2>/dev/null
pkill -9 -f 'src/runpod/pod_runner' 2>/dev/null

echo '── [2/4] 杀所有占 GPU 的 PID (EngineCore / worker / ray)'
PIDS=\$(timeout 5 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
if [ -n \"\$PIDS\" ]; then
  echo \"  待杀 PID: \$PIDS\"
  echo \"\$PIDS\" | xargs -r kill -9 2>/dev/null
else
  echo '  (无 GPU 进程)'
fi

echo '── [3/4] 兜底：干掉残留 vllm / multiprocessing / nohup 包装'
# EngineCore 被 setproctitle 改名为 'VLLM::EngineCore'，同时在 cmdline 里也要匹配到。
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
pkill -9 -f 'vllm.v1.engine' 2>/dev/null
pkill -9 -f 'vllm.engine'    2>/dev/null
pkill -9 -f 'EngineCore'     2>/dev/null
pkill -9 -f 'multiprocessing.spawn' 2>/dev/null
pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null
pkill -9 -f 'nohup env'      2>/dev/null
sleep 2

# 循环兜底：如果 nvidia-smi 还看到占 GPU 的 PID，就再硬杀（最多 3 轮 × 2s）
for i in 1 2 3; do
  LEFT=\$(timeout 5 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  [ -z \"\$LEFT\" ] && break
  echo \"  二轮 [\$i/3] 还有 GPU 占用者 \$LEFT，硬杀中...\"
  echo \"\$LEFT\" | xargs -r kill -9 2>/dev/null
  sleep 2
done

if [ '$CLEAN_LOGS' = '1' ]; then
  echo '── 附加：清理旧 log / stdout'
  rm -f ${POD_WS}/output/pod_runner.log ${POD_WS}/output/pod_runner.stdout
  rm -f ${POD_WS}/output/.checkpoint.jsonl.tmp 2>/dev/null
fi

echo '── [4/4] 等显存真正释放（内核回收 CUDA context / NCCL / IPC handle 需要几秒到几十秒）'
# 进程杀掉后信号已送到，但显存释放是异步的。这里 poll 显存占用，直到降到驱动基线
# （通常 < 512 MiB 只剩 CUDA driver 自己占的）或超时。
THRESHOLD_MIB=512
TIMEOUT_SEC=45
ELAPSED=0
while [ \"\$ELAPSED\" -lt \"\$TIMEOUT_SEC\" ]; do
  USED=\$(timeout 5 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  USED=\${USED:-0}
  if [ \"\$USED\" -lt \"\$THRESHOLD_MIB\" ]; then
    echo \"  显存已释放到 \${USED} MiB (< \${THRESHOLD_MIB} MiB 驱动基线)\"
    break
  fi
  echo \"  [\${ELAPSED}s] 显存占用 \${USED} MiB，等释放...\"
  sleep 3
  ELAPSED=\$((ELAPSED + 3))
done

# 最终状态
echo ''
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

REMAIN_PIDS=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
FINAL_USED=\$(timeout 5 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
FINAL_USED=\${FINAL_USED:-0}

if [ \"\$REMAIN_PIDS\" -gt 0 ]; then
  echo \"[WARN] 还有 \$REMAIN_PIDS 个进程占着 GPU:\"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  echo '再跑一次本脚本；仍占用只能 Runpod 网页 Stop Pod → Start Pod。'
  exit 2
fi

if [ \"\$FINAL_USED\" -ge \"\$THRESHOLD_MIB\" ]; then
  echo \"[WARN] 没有进程占 GPU 了，但显存还有 \${FINAL_USED} MiB 没释放。\"
  echo '这是 CUDA context 残留（内核持有），kill 无效。'
  echo '处理：Runpod 网页 Stop Pod (不要 Terminate) → 等 10s → Start Pod。'
  exit 3
fi

echo '✅ GPU 干净（显存 '\${FINAL_USED}' MiB），可以重新 bash scripts/runpod/02_run.sh'
"

ssh -i "$POD_KEY" -p "$POD_PORT" \
    -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    "${POD_USER}@${POD_HOST}" "$REMOTE_CMD"
