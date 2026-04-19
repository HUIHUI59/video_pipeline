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

# 从 yaml 抽 Pod 连接信息 + 模型信息（MODEL_NAME / MODEL_SIZE_GB / MODEL_DIR）
eval "$(python - <<'PY' "$CONFIG"
import os, sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
p = c["pod"]
print(f"POD_HOST={p['host']}")
print(f"POD_PORT={p['port']}")
print(f"POD_USER={p['user']}")
print(f"POD_KEY={os.path.expanduser(p['ssh_key'])}")
print(f"POD_WS={c['paths']['pod_workspace']}")

# 派生给 pod_setup.sh 用的三个变量
m = (c.get("model") or {})
name = m.get("name") or "Qwen/Qwen3-VL-32B-Instruct"
nl = name.lower()
# 保守估算权重磁盘占用（GB）；yaml 里可通过 model.weights_gb 显式覆盖
if   "qwen3-vl-32b"  in nl and "fp8" in nl:               size = 35
elif "qwen3-vl-32b"  in nl:                               size = 67
elif "122b"          in nl and ("gptq" in nl or "int4" in nl or "awq" in nl):
                                                          size = 68
elif "122b"          in nl and "fp8" in nl:               size = 122
elif "122b"          in nl:                               size = 234
else:                                                     size = int(m.get("weights_gb") or 70)
slug = name.rsplit("/", 1)[-1].lower()
# 优先 yaml 里显式 model.path；否则默认 Network Volume /workspace/models/<slug>
mdir = m.get("path") or f"/workspace/models/{slug}"
print(f"MODEL_NAME={name}")
print(f"MODEL_SIZE_GB={size}")
print(f"MODEL_DIR={mdir}")
PY
)"

echo "  Pod:   ${POD_USER}@${POD_HOST}:${POD_PORT}  workspace=${POD_WS}"
echo "  Model: ${MODEL_NAME} (~${MODEL_SIZE_GB} GB) → ${MODEL_DIR}"

# 远端命令：如果 pod_setup 标志文件不存在就跑一次安装，然后后台跑 runner
REMOTE_CMD="set -e; cd '${POD_WS}'; \
  VENV_PY=/opt/labeling-env/bin/python; \
  # venv 存在性是权威检查：新 Pod 的 container disk 是空的，
  # 即使 Network Volume 上还有 .pod_setup_done 残留也要重装
  if [ ! -x \"\$VENV_PY\" ]; then \
    echo '首次 setup 环境（venv 不存在）...'; \
    rm -f .pod_setup_done; \
    MODEL_NAME='${MODEL_NAME}' MODEL_SIZE_GB='${MODEL_SIZE_GB}' MODEL_DIR='${MODEL_DIR}' \
      bash tools/pod_setup.sh \
      || { echo '[ERR] pod_setup.sh 失败（模型下载或依赖装失败）。不继续启动 runner。'; exit 1; }; \
    touch .pod_setup_done; \
  elif [ ! -f .pod_setup_done ]; then \
    echo 'venv 在但没 .pod_setup_done（可能上次只装了依赖没下模型），补跑 pod_setup...'; \
    MODEL_NAME='${MODEL_NAME}' MODEL_SIZE_GB='${MODEL_SIZE_GB}' MODEL_DIR='${MODEL_DIR}' \
      bash tools/pod_setup.sh \
      || { echo '[ERR] pod_setup.sh 失败（模型下载或依赖装失败）。不继续启动 runner。'; exit 1; }; \
    touch .pod_setup_done; \
  fi; \
  if [ ! -x \"\$VENV_PY\" ]; then \
    echo \"[ERR] 跑完 pod_setup.sh 后 \$VENV_PY 仍不存在。手动 ssh 进 Pod 看 tools/pod_setup.sh 的报错。\"; exit 1; \
  fi; \
  mkdir -p output; \
  # 把所有 cache 指向 Network Volume (/workspace)，避免容器盘 40GB 被 triton/torch/hf 挤爆
  export HF_HOME=/workspace/.cache/huggingface; \
  export XDG_CACHE_HOME=/workspace/.cache; \
  export TRITON_CACHE_DIR=/workspace/.cache/triton; \
  export VLLM_CACHE_ROOT=/workspace/.cache/vllm; \
  export TMPDIR=/workspace/.tmp; \
  mkdir -p \$HF_HOME \$XDG_CACHE_HOME \$TRITON_CACHE_DIR \$VLLM_CACHE_ROOT \$TMPDIR; \
  # CUDA 工具链 PATH（flashinfer JIT 调 nvcc 编译 MoE/gdn kernel 必需）。
  # Runpod 的 pytorch:cu128 镜像里 nvcc 在 /usr/local/cuda-12.8/bin，但默认
  # PATH 不含这个目录，nohup 继承父 shell 的 PATH，所以要在这里显式注入。
  if   [ -d /usr/local/cuda-12.8 ]; then export CUDA_HOME=/usr/local/cuda-12.8; \
  elif [ -d /usr/local/cuda ];      then export CUDA_HOME=/usr/local/cuda; \
  elif [ -d /usr/local/cuda-12 ];   then export CUDA_HOME=/usr/local/cuda-12; \
  else echo '[WARN] 找不到 /usr/local/cuda*；flashinfer JIT 可能失败'; export CUDA_HOME=; \
  fi; \
  if [ -n \"\$CUDA_HOME\" ]; then \
    export PATH=\"\$CUDA_HOME/bin:\$PATH\"; \
    export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}\"; \
    echo \"  CUDA_HOME=\$CUDA_HOME (nvcc=\$(\$CUDA_HOME/bin/nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' || echo unknown))\"; \
  fi; \
  echo \"启动 pod_runner (nohup, using \$VENV_PY)\"; \
  echo \"  HF_HOME=\$HF_HOME\"; \
  echo \"  TRITON_CACHE_DIR=\$TRITON_CACHE_DIR\"; \
  echo \"  VLLM_CACHE_ROOT=\$VLLM_CACHE_ROOT\"; \
  nohup env HF_HOME=\"\$HF_HOME\" XDG_CACHE_HOME=\"\$XDG_CACHE_HOME\" \
             TRITON_CACHE_DIR=\"\$TRITON_CACHE_DIR\" VLLM_CACHE_ROOT=\"\$VLLM_CACHE_ROOT\" \
             TMPDIR=\"\$TMPDIR\" \
             CUDA_HOME=\"\$CUDA_HOME\" PATH=\"\$PATH\" LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH\" \
             \"\$VENV_PY\" src/runpod/pod_runner.py --config runpod.yaml \
    >> output/pod_runner.stdout 2>&1 & \
  RUNNER_PID=\$!; \
  echo 'runner pid ='\$RUNNER_PID; \
  sleep 5; \
  if kill -0 \$RUNNER_PID 2>/dev/null; then \
    echo 'runner 还活着。tail -f 同时跟两个文件（log = python logger/心跳，stdout = vLLM 内部进度条 + ninja/nvcc JIT 输出）。Ctrl+C 退出 tail 不杀 runner。'; \
    # 等 log 至少出现一次（通常 <2 秒），否则先 tail stdout 顶一会儿
    for _i in 1 2 3 4 5; do [ -f output/pod_runner.log ] && break; sleep 1; done; \
    if [ -f output/pod_runner.log ] && [ -f output/pod_runner.stdout ]; then \
      tail -n +1 -f output/pod_runner.log output/pod_runner.stdout; \
    elif [ -f output/pod_runner.log ]; then \
      tail -f output/pod_runner.log; \
    else \
      echo '(pod_runner.log 还没创建，tail stdout 代替)'; \
      tail -f output/pod_runner.stdout; \
    fi; \
  else \
    echo '[ERR] runner 启动后立刻退出了。pod_runner.stdout 末尾 80 行:'; \
    tail -80 output/pod_runner.stdout; \
    echo '---'; \
    echo '[ERR] pod_runner.log (若存在):'; \
    tail -40 output/pod_runner.log 2>/dev/null || echo '(日志文件不存在)'; \
    exit 1; \
  fi"

ssh -i "$POD_KEY" -p "$POD_PORT" \
    -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    "${POD_USER}@${POD_HOST}" "$REMOTE_CMD"
