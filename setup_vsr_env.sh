#!/bin/bash
# setup_vsr_env.sh  v2
# 在三台机器上创建统一的 vsr conda 环境（Python 3.12）
# 用法：bash ~/video_pipeline/setup_vsr_env.sh
#
# CUDA 版本路由：
#   CUDA >= 12.4  →  torch 2.7.0 + paddle cu126（均使用 cuDNN 9，无冲突）
#   CUDA <  12.4  →  torch 2.4.1 + paddle cu118（均使用 cuDNN 8，无冲突）

set -eo pipefail

VSR_DIR="${HOME}/video-subtitle-remover"
ENV_NAME="vsr"

echo "================================================"
echo "  VSR 环境安装脚本 v2"
echo "  机器: $(hostname)"
echo "================================================"

# ── 前置检查 ─────────────────────────────────────
[ -d "$VSR_DIR" ] || { echo "❌ 未找到 ${VSR_DIR}，请先 git clone VSR 仓库"; exit 1; }
command -v conda &>/dev/null || { echo "❌ 未找到 conda"; exit 1; }

# nvidia-smi 路径：WSL2 下位于 /usr/lib/wsl/lib/，不在默认 PATH 中
NVIDIA_SMI=""
if command -v nvidia-smi &>/dev/null; then
    NVIDIA_SMI="nvidia-smi"
elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
fi
[ -n "$NVIDIA_SMI" ] || { echo "❌ 未检测到 NVIDIA GPU（nvidia-smi 不在 PATH 中）"; exit 1; }
$NVIDIA_SMI &>/dev/null || { echo "❌ nvidia-smi 执行失败"; exit 1; }

# ── 检测 CUDA 版本 ────────────────────────────────
CUDA_VER=$($NVIDIA_SMI | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
echo "✅ 检测到 CUDA ${CUDA_VER}"

# 根据 CUDA 版本选择安装策略
# cu118: PaddlePaddle 需要 cuDNN 8 → 需配合 PyTorch 2.4.1 cu118（也用 cuDNN 8）
# cu126: PaddlePaddle 需要 cuDNN 9 → 配合 PyTorch 2.7.0 cu126（也用 cuDNN 9）
if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
    TORCH_CMD="pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126"
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"
    CUDA_TAG="cu126"
    echo "  策略: cu126（PyTorch 2.7.0 + PaddlePaddle cu126，cuDNN 9）"
else
    # cu118：PyTorch 2.4.1 与 PaddlePaddle 3.0.0 均使用 cuDNN 8，相互兼容
    TORCH_CMD="pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118"
    PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu118/"
    CUDA_TAG="cu118"
    echo "  策略: cu118（PyTorch 2.4.1 + PaddlePaddle cu118，cuDNN 8）"
fi

# ── 激活 conda ────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"

# ── 创建 Python 3.12 环境 ─────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠  环境 ${ENV_NAME} 已存在，跳过创建"
    echo "   如需重建: conda env remove -n ${ENV_NAME} -y"
else
    echo ""
    echo "=== [1/6] 创建 Python 3.12 conda 环境 ==="
    conda create -n "$ENV_NAME" python=3.12 -y
fi

conda activate "$ENV_NAME"
echo "✅ 已激活: $ENV_NAME  ($(python --version))"

# ── PaddlePaddle ──────────────────────────────────
echo ""
echo "=== [2/6] 安装 PaddlePaddle-GPU (${CUDA_TAG}) ==="
pip install paddlepaddle-gpu==3.0.0 \
    -i "${PADDLE_INDEX}" \
    --trusted-host www.paddlepaddle.org.cn

# cu118 的 PaddlePaddle 链接 libcudnn.so.8（cuDNN 8），但某些 VSR 依赖会拉入
# nvidia-cudnn-cu12（cuDNN 9），导致找不到 .so.8。确保 cuDNN 8 也可用。
if [ "$CUDA_TAG" = "cu118" ]; then
    echo "  cu118: 确保 cuDNN 8 可用..."
    pip install "nvidia-cudnn-cu11>=8.9,<9" 2>/dev/null || true
fi

# ── PyTorch ───────────────────────────────────────
echo ""
echo "=== [3/6] 安装 PyTorch (${CUDA_TAG}) ==="
eval "$TORCH_CMD"

# 验证 cuDNN 不冲突
python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print(f'  PyTorch {torch.__version__} CUDA OK')"

# ── 修复 requirements.txt ─────────────────────────
echo ""
echo "=== [4/6] 修复 VSR requirements.txt ==="
cd "$VSR_DIR"
sed -i 's/av==14\.[34]\.[0-9]*/av==14.2.0/' requirements.txt
echo "  av 版本已修正"

# ── 有版本冲突的包单独处理 ────────────────────────
echo ""
echo "=== [5/6] 安装 VSR 依赖 ==="

# antlr4 4.9.* 已从 PyPI 撤除；omegaconf 用 --no-deps 跳过该依赖检查
pip install "antlr4-python3-runtime==4.13.2"
pip install "omegaconf==2.3.0" --no-deps

# av 必须用预编译 wheel
pip install "av==14.2.0" --only-binary=:all:

# filesplit 只能源码编译（无额外 C 依赖，直接装）
pip install filesplit==3.0.2

# 其余依赖：排除已特殊处理的包，全用预编译 wheel
TMPFILE=$(mktemp /tmp/vsr_reqs_XXXXXX.txt)
grep -vE "^(av|omegaconf|filesplit|onnxruntime-directml)" "$VSR_DIR/requirements.txt" > "$TMPFILE"

pip install -r "$TMPFILE" \
    --ignore-requires-python \
    --only-binary=:all:

rm -f "$TMPFILE"

# ── Pipeline 自身依赖 ─────────────────────────────
echo ""
echo "=== [6/6] 安装 Pipeline 依赖 ==="
pip install "rich>=13.0" pyyaml "scenedetect[opencv]"

# ── 修正 VSR 默认配置（画质 & 检测精度）────────────
echo ""
echo "=== [补丁] 修正 VSR 配置 ==="

# 1. 启用字幕检测（默认 True 时跳过检测，对整帧 inpaint，会误删人脸/眼睛）
sed -i 's/^STTN_SKIP_DETECTION = True/STTN_SKIP_DETECTION = False/' \
    "$VSR_DIR/backend/config.py"
echo "  STTN_SKIP_DETECTION → False（启用字幕区域检测）"

# 2. 中间临时文件改用 MJPG（帧内压缩，避免 mp4v 跨帧压缩损失）
sed -i "s/NamedTemporaryFile(suffix='.mp4', delete=False)/NamedTemporaryFile(suffix='.avi', delete=False)/" \
    "$VSR_DIR/backend/main.py"
sed -i "s/VideoWriter_fourcc(\*'mp4v')/VideoWriter_fourcc(*'MJPG')/" \
    "$VSR_DIR/backend/main.py"
echo "  VideoWriter: mp4v → MJPG（帧内压缩，画质更好）"

# 3. ffmpeg 最终编码加入高质量参数（CRF 17 替代默认 CRF 23）
export _VSR_MAIN="$VSR_DIR/backend/main.py"
python3 - <<'PYEOF'
import os, pathlib
p = pathlib.Path(os.environ["_VSR_MAIN"])
src = p.read_text()
old = '"-vcodec", "libx264" if config.USE_H264 else "copy",'
new = '"-vcodec", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p",'
if old in src:
    p.write_text(src.replace(old, new))
    print("  ffmpeg 编码: 添加 -crf 17 -preset medium（高画质输出）")
elif new in src:
    print("  ffmpeg 编码: 已是高质量配置，跳过")
else:
    print("  ⚠ ffmpeg 编码行未找到，请手动检查 backend/main.py")
PYEOF

# ── 验证 ──────────────────────────────────────────
echo ""
echo "================================================"
echo "  验证安装"
echo "================================================"
python - <<'EOF'
import sys, importlib.metadata as meta
print(f"Python : {sys.version.split()[0]}")

import torch
cuda_ok = torch.cuda.is_available()
print(f"PyTorch: {torch.__version__}  CUDA: {'✅ ' + torch.version.cuda if cuda_ok else '❌ 不可用'}")
if cuda_ok:
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

try:
    import paddle
    gpu_ok = paddle.device.is_compiled_with_cuda()
    print(f"Paddle : {paddle.__version__}  GPU: {'✅' if gpu_ok else '❌'}")
except Exception as e:
    print(f"Paddle : ❌ {e}")

for pkg in ("scenedetect", "rich", "yaml"):
    try:
        name = "PyYAML" if pkg == "yaml" else pkg
        ver  = meta.version(name)
        print(f"{name:<12}: ✅ {ver}")
    except Exception as e:
        print(f"{name:<12}: ❌ {e}")

# 最关键：验证 VSR 能否导入
import sys, os
vsr_dir = os.path.expanduser("~/video-subtitle-remover")
sys.path.insert(0, vsr_dir)
try:
    from backend.main import SubtitleRemover
    print("VSR API: ✅ SubtitleRemover 可导入")
except Exception as e:
    print(f"VSR API: ❌ {e}")
EOF

echo ""
echo "✅ 安装完成！(CUDA 策略: ${CUDA_TAG})"
echo ""
echo "后续步骤："
echo "  conda activate vsr"
echo "  servers.yaml 中 conda_env 改为 vsr"
