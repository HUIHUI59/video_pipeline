#!/usr/bin/env bash
# tools/pod_setup.sh — 仅在 Runpod Pod 内执行。
# 作用：装 vLLM + 依赖，下载 Qwen3-VL-32B。成功后由调用方 touch .pod_setup_done。
# 幂等：重复跑不会重复下载大模型（HF cache 去重）。
set -euo pipefail

echo "══ pod_setup.sh  @ $(hostname)  $(date -Is) ══"

# 1) 系统依赖
apt-get update -qq
apt-get install -y -qq ffmpeg git rsync python3-venv

# 2) Python venv（不污染系统 python）
VENV=/opt/labeling-env
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip wheel

# 3) 依赖（vllm 自带 torch，固定最低版本）
pip install \
    "vllm>=0.6.3" \
    "torch>=2.4" \
    "decord>=0.6" \
    "pillow>=10.0" \
    "pydantic>=2" \
    "pyyaml>=6" \
    "rich" \
    "huggingface_hub[cli]>=0.24"

# 4) 模型下载到 Pod 容器盘（本地 NVMe，加载快 40×）
# 为什么不放 Network Volume：MooseFS 冷读 ~50 MB/s，67GB 要 22 min；容器盘 NVMe 加载只需 30s。
# 代价：Pod Terminate 后容器盘被擦，下次新 Pod 要重新从 HF 下载（~5 min，HF CDN ~300 MB/s）。
# 总账：新 Pod 冷启 = 8 min(venv+deps) + 5 min(模型下载) + 30s(加载) ≈ 14 min
# 对比旧方案 Network Volume：8 min + 0 min(模型持久化) + 22 min(冷读) = 30 min
MODEL_NAME="Qwen/Qwen3-VL-32B-Instruct"
MODEL_DIR="${MODEL_DIR:-/root/qwen3-vl-32b}"  # 默认容器盘；环境变量可覆盖
# 兼容旧部署：若 /workspace/models/qwen3-vl-32b 已存在且有权重，直接复用
LEGACY_DIR="/workspace/models/qwen3-vl-32b"
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ] && \
   [ -f "$LEGACY_DIR/model.safetensors.index.json" ]; then
  echo "发现旧 Network Volume 模型 $LEGACY_DIR，直接复用（不再重复下载）。"
  echo "提示：下次租 Pod 时可以把它删了省 Network Volume 费用（rm -rf $LEGACY_DIR）"
  MODEL_DIR="$LEGACY_DIR"
fi

mkdir -p "$MODEL_DIR"
# 看到 safetensors index 就认为已下载完（防止空目录通过 ls -A 但其实没下）
if [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "模型已存在 $MODEL_DIR，跳过下载。"
else
  # 磁盘空间预检：模型 67GB，预留 10GB 头
  AVAIL_GB=$(df -BG --output=avail "$(dirname "$MODEL_DIR")" | tail -1 | tr -dc '0-9')
  if [ "${AVAIL_GB:-0}" -lt 77 ]; then
    echo "[ERR] $(dirname "$MODEL_DIR") 剩余 ${AVAIL_GB}GB < 77GB，放不下 Qwen3-VL-32B。"
    echo "  如果用的是默认 40GB 容器盘，请 Terminate 后重租 Pod，Container Disk 调到 100GB+。"
    exit 1
  fi
  echo "下载 $MODEL_NAME → $MODEL_DIR (可用 ${AVAIL_GB}GB)"
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_NAME" --local-dir "$MODEL_DIR"
  else
    huggingface-cli download "$MODEL_NAME" --local-dir "$MODEL_DIR"
  fi
fi
echo "模型就位: $MODEL_DIR"

# 5) 把 venv 激活写进 bashrc（方便用户 ssh 进 pod 后直接跑 python）
if ! grep -q "labeling-env/bin/activate" ~/.bashrc 2>/dev/null; then
  echo "source $VENV/bin/activate" >> ~/.bashrc
fi

echo "══ pod_setup 完成 ══"
echo "python 路径: $(which python)"
python -c "import vllm, torch, pydantic, decord; print('vllm', vllm.__version__, 'torch', torch.__version__)"
