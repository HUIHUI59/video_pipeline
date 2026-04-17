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

# 4) 模型下载到 Pod 本地磁盘
MODEL_NAME="Qwen/Qwen3-VL-32B-Instruct"
MODEL_DIR="/workspace/models/qwen3-vl-32b"
mkdir -p "$MODEL_DIR"
if [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
  echo "下载 $MODEL_NAME → $MODEL_DIR"
  huggingface-cli download "$MODEL_NAME" \
      --local-dir "$MODEL_DIR" \
      --local-dir-use-symlinks False
else
  echo "模型已存在，跳过下载。"
fi

# 5) 把 venv 激活写进 bashrc（方便用户 ssh 进 pod 后直接跑 python）
if ! grep -q "labeling-env/bin/activate" ~/.bashrc 2>/dev/null; then
  echo "source $VENV/bin/activate" >> ~/.bashrc
fi

echo "══ pod_setup 完成 ══"
echo "python 路径: $(which python)"
python -c "import vllm, torch, pydantic, decord; print('vllm', vllm.__version__, 'torch', torch.__version__)"
