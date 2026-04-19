#!/usr/bin/env bash
# tools/pod_setup.sh — 仅在 Runpod Pod 内执行。
# 作用：装 vLLM + 依赖，下载 VLM 权重。成功后由调用方 touch .pod_setup_done。
# 幂等：已有权重（见 safetensors index）则跳过下载。
#
# 环境变量（02_run.sh 从 configs/runpod*.yaml 解析并传入）：
#   MODEL_NAME     HF 模型 repo id  （默认 Qwen/Qwen3-VL-32B-Instruct）
#   MODEL_SIZE_GB  权重磁盘占用     （默认 67 = 32B BF16；122B-AWQ≈68；32B FP8≈35）
#   MODEL_DIR      下载目标路径     （默认 /workspace/models/<slug>，Network Volume）
set -euo pipefail

echo "══ pod_setup.sh  @ $(hostname)  $(date -Is) ══"

# 1) 系统依赖
# ninja-build：flashinfer JIT 编译 kernel 必需（Qwen3.5 的 gdn_linear_attn
# 在 vLLM 里没预编译，运行时动态构建）。不装会在第一个 shot 推理时崩：
#   FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
apt-get update -qq
apt-get install -y -qq ffmpeg git rsync python3-venv ninja-build

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

# 4) 模型下载 —— 默认去 Network Volume（/workspace/models/...）
# 为什么改默认：容器盘通常 50-80 GB，装完 vLLM+torch 依赖 ~16 GB 后只剩 ~64 GB；
# 32B/122B-AWQ 权重 67-68 GB 正好踩过门槛（加 10 GB headroom 共需 77-78 GB），
# 实测会报"剩余 65GB < 77GB"。Network Volume 100-500 GB 常见，模型持久化，下次
# 开 Pod 秒级复用。冷读比 NVMe 慢（~200-500 MB/s vs ~3 GB/s），67 GB 首次加载
# 多花 3-5 分钟，但避免每次新 Pod 重下 5 分钟 + 爆仓风险。
# 想走容器盘：export MODEL_DIR=/root/<slug>，但确保 Container Disk ≥ (权重+30) GB。
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"
MODEL_SIZE_GB="${MODEL_SIZE_GB:-67}"
_slug="$(basename "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')"
MODEL_DIR="${MODEL_DIR:-/workspace/models/${_slug}}"

# 兼容检测：几个已知路径里找现成模型，直接复用（不再重复下载）
for _legacy in \
    "/workspace/models/${_slug}" \
    "/workspace/models/$(basename "$MODEL_NAME")" \
    "/workspace/models/qwen3-vl-32b" \
    "/root/${_slug}" \
    "/root/qwen3-vl-32b" \
    "/root/models/${_slug}"; do
  if [ -f "$_legacy/model.safetensors.index.json" ]; then
    echo "发现已下载模型 $_legacy，复用（不再重复下载）"
    MODEL_DIR="$_legacy"
    break
  fi
done

mkdir -p "$MODEL_DIR"
# 看到 safetensors index 就认为已下载完（防止空目录通过 ls -A 但其实没下）
if [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "模型已存在 $MODEL_DIR，跳过下载。"
else
  # 磁盘空间预检：权重 + 10 GB headroom（HF 临时文件 / index / 校验 hash）
  NEEDED_GB=$((MODEL_SIZE_GB + 10))
  AVAIL_GB=$(df -BG --output=avail "$(dirname "$MODEL_DIR")" | tail -1 | tr -dc '0-9')
  if [ "${AVAIL_GB:-0}" -lt "$NEEDED_GB" ]; then
    _parent="$(dirname "$MODEL_DIR")"
    echo "[ERR] $_parent 剩余 ${AVAIL_GB}GB < 需要 ${NEEDED_GB}GB"
    echo "      （模型 ${MODEL_SIZE_GB}GB + 10GB 头；MODEL_NAME=$MODEL_NAME）"
    echo "  解决方案（按推荐程度）："
    echo "    1) 改下到 Network Volume："
    echo "         MODEL_DIR=/workspace/models/${_slug} bash tools/pod_setup.sh"
    echo "       （前提：Pod 已挂 Network Volume，100GB+ 通常够）"
    echo "    2) Terminate Pod 重租，Container Disk ≥ $((MODEL_SIZE_GB + 30))GB"
    echo "    3) 手动清理 $_parent 腾空间后重跑"
    exit 1
  fi
  echo "下载 $MODEL_NAME → $MODEL_DIR (可用 ${AVAIL_GB}GB / 需要 ${NEEDED_GB}GB)"
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_NAME" --local-dir "$MODEL_DIR"
  else
    huggingface-cli download "$MODEL_NAME" --local-dir "$MODEL_DIR"
  fi
fi
echo "模型就位: $MODEL_DIR"
# 写一个标记文件，便于 ssh 进 Pod 查看当前使用的模型路径
echo "$MODEL_DIR" > /tmp/.pod_setup_model_dir

# 5) 把 venv 激活写进 bashrc（方便用户 ssh 进 pod 后直接跑 python）
if ! grep -q "labeling-env/bin/activate" ~/.bashrc 2>/dev/null; then
  echo "source $VENV/bin/activate" >> ~/.bashrc
fi

echo "══ pod_setup 完成 ══"
echo "python 路径: $(which python)"
python -c "import vllm, torch, pydantic, decord; print('vllm', vllm.__version__, 'torch', torch.__version__)"
