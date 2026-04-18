#!/usr/bin/env bash
# tools/kill_gpu.sh — 在 Pod 内直接运行，杀掉 pod_runner 和所有占 GPU 的进程，
# 等显存释放后返回。无需本地 ssh 转发，ssh 进 Pod 之后直接跑：
#
#   bash /workspace/labeling/tools/kill_gpu.sh
#
# 或在 /workspace/labeling 目录下：
#
#   bash tools/kill_gpu.sh
#
# 退出码：
#   0 — GPU 干净（显存 < 512 MiB，无进程）
#   2 — 还有进程占 GPU（极少见，再跑一次）
#   3 — CUDA context 卡死，命令行救不了，需要 Runpod 网页 Stop → Start Pod
set +e

THRESHOLD_MIB=512
KILL_ROUND_MAX=3
MEMORY_WAIT_ROUNDS=15   # 15 × 3s = 45s 上限

echo "── [1/3] pkill pod_runner + vLLM / EngineCore / multiprocessing 全家桶"
pkill -9 -f pod_runner.py 2>/dev/null
pkill -9 -f 'src/runpod/pod_runner' 2>/dev/null
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
pkill -9 -f 'vllm.v1' 2>/dev/null
pkill -9 -f 'vllm.engine' 2>/dev/null
pkill -9 -f 'EngineCore' 2>/dev/null
pkill -9 -f 'multiprocessing.spawn' 2>/dev/null
pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null
pkill -9 -f 'nohup env' 2>/dev/null
sleep 1

echo "── [2/3] 按 nvidia-smi 报告的 compute PID 循环硬杀（最多 ${KILL_ROUND_MAX} 轮）"
for round in $(seq 1 "$KILL_ROUND_MAX"); do
  PIDS=$(timeout 5 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  if [ -z "$PIDS" ]; then
    echo "  轮 ${round}: 无 GPU 进程"
    break
  fi
  echo "  轮 ${round}: kill -9 ${PIDS}"
  echo "$PIDS" | xargs -r kill -9 2>/dev/null
  sleep 2
done

echo "── [3/3] poll 显存释放到 < ${THRESHOLD_MIB} MiB（等内核回收 CUDA context / NCCL / IPC）"
for i in $(seq 1 "$MEMORY_WAIT_ROUNDS"); do
  USED=$(timeout 5 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  USED=${USED:-99999}
  if [ "$USED" -lt "$THRESHOLD_MIB" ]; then
    echo "  显存 ${USED} MiB（释放完成）"
    break
  fi
  echo "  [${i}/${MEMORY_WAIT_ROUNDS}] 显存占用 ${USED} MiB，等释放..."
  sleep 3
done

echo ''
timeout 5 nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo ''

REMAIN_PIDS=$(timeout 5 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
FINAL_USED=$(timeout 5 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
FINAL_USED=${FINAL_USED:-99999}

if [ "$REMAIN_PIDS" -gt 0 ]; then
  echo "[WARN] 还有 ${REMAIN_PIDS} 个进程占 GPU:"
  timeout 5 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  echo "  再跑一次 bash tools/kill_gpu.sh；仍占用只能 Runpod 网页 Stop → Start Pod。"
  exit 2
fi

if [ "$FINAL_USED" -ge "$THRESHOLD_MIB" ]; then
  echo "[WARN] 无进程占 GPU，但显存还有 ${FINAL_USED} MiB 没释放。"
  echo "  这是 CUDA context 残留（内核层面锁着），kill 已经救不了。"
  echo "  处理：Runpod 网页 Stop Pod（不要 Terminate）→ 等 10s → Start Pod。"
  exit 3
fi

echo "✅ GPU 干净（显存 ${FINAL_USED} MiB），可以重跑 pod_runner"
