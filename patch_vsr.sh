#!/bin/bash
# patch_vsr.sh
# 修正 video-subtitle-remover 的默认配置，提升字幕检测精度和输出画质。
# 幂等：已打补丁的机器重复执行无副作用。
#
# 用法：
#   本地执行        : bash ~/video_pipeline/patch_vsr.sh
#   远端批量执行    : ssh user@host "bash ~/video_pipeline/patch_vsr.sh"

set -eo pipefail

VSR_DIR="${HOME}/video-subtitle-remover"
CONFIG="$VSR_DIR/backend/config.py"
MAIN="$VSR_DIR/backend/main.py"

echo "================================================"
echo "  VSR 质量补丁"
echo "  机器: $(hostname)"
echo "  VSR : $VSR_DIR"
echo "================================================"

[ -d "$VSR_DIR" ] || { echo "❌ 未找到 $VSR_DIR，请先 clone VSR 仓库"; exit 1; }

# ── 补丁 1：启用字幕区域检测 ──────────────────────────
# 默认 STTN_SKIP_DETECTION=True 跳过检测，未指定 sub_area 时对整帧做 inpaint，
# 会误删人脸、眼睛等非字幕内容。
if grep -q "^STTN_SKIP_DETECTION = True" "$CONFIG"; then
    sed -i 's/^STTN_SKIP_DETECTION = True/STTN_SKIP_DETECTION = False/' "$CONFIG"
    echo "✅ [1/3] STTN_SKIP_DETECTION → False（启用字幕区域检测）"
else
    echo "⏭  [1/3] STTN_SKIP_DETECTION 已是 False，跳过"
fi

# ── 补丁 2：中间临时文件改用 MJPG ─────────────────────
# 原 mp4v（MPEG-4 Part 2）有跨帧压缩损失；
# MJPG（Motion JPEG）为帧内压缩，画质更好。
if grep -q "suffix='.mp4', delete=False" "$MAIN"; then
    sed -i "s/NamedTemporaryFile(suffix='.mp4', delete=False)/NamedTemporaryFile(suffix='.avi', delete=False)/" "$MAIN"
    sed -i "s/VideoWriter_fourcc(\*'mp4v')/VideoWriter_fourcc(*'MJPG')/" "$MAIN"
    echo "✅ [2/3] VideoWriter: mp4v → MJPG（帧内压缩）"
else
    echo "⏭  [2/3] VideoWriter 已是 MJPG，跳过"
fi

# ── 补丁 3：ffmpeg 最终编码加入高质量参数 ─────────────
# 原命令无质量参数（默认 CRF 23）；改为 CRF 17 + preset medium。
export _VSR_MAIN="$MAIN"
python3 - << 'PYEOF'
import pathlib, os, sys

p = pathlib.Path(os.environ.get("_VSR_MAIN", ""))
if not p.exists():
    sys.exit(0)

src = p.read_text()
old = '"-vcodec", "libx264" if config.USE_H264 else "copy",'
new = '"-vcodec", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p",'

if old in src:
    p.write_text(src.replace(old, new))
    print("✅ [3/3] ffmpeg 编码: 添加 -crf 17 -preset medium -pix_fmt yuv420p")
elif new in src:
    print("⏭  [3/3] ffmpeg 已是高质量配置，跳过")
else:
    print("⚠  [3/3] ffmpeg 编码行未找到，请手动检查 backend/main.py")
PYEOF

echo ""
echo "✅ VSR 补丁完成！($(hostname))"
