# 🎬 视频批量预处理 Pipeline

> 专为视频模型训练设计：批量将多格式、多分辨率视频统一转码为 **1080p / MP4 / H.264**，支持三台 GPU 服务器并行处理。

---

## 📁 文件结构

```
video_pipeline/
├── process_videos.py       # 单机处理主程序（含 GPU 调度）
├── distributed_dispatch.py # 多机分布式调度器
├── servers.yaml            # 服务器配置
└── requirements.txt        # Python 依赖
```

---

## ⚙️ 环境准备

### 1. 安装系统依赖（每台服务器）

```bash
# Ubuntu 22.04
sudo apt update
sudo apt install -y ffmpeg

# 验证 NVENC（需要 NVIDIA 驱动 + CUDA）
ffmpeg -encoders | grep nvenc
# 应看到 h264_nvenc

# 验证 GPU
nvidia-smi
```

### 2. 安装 Python 依赖

```bash
pip install rich tqdm pyyaml
```

---

## 🚀 快速使用

### 单机模式（推荐先测试）

```bash
# 基本用法（自动检测 GPU）
python process_videos.py /path/to/movies /path/to/output

# 指定 GPU ID
python process_videos.py /path/to/movies /path/to/output --gpu-ids 0

# CPU 模式（无 GPU 时）
python process_videos.py /path/to/movies /path/to/output --no-nvenc

# 强制重新处理所有文件（忽略断点续传）
python process_videos.py /path/to/movies /path/to/output --force

# 先预览任务列表（不实际处理）
python process_videos.py /path/to/movies /path/to/output --dry-run

# 指定并发数（默认 = GPU 数量）
python process_videos.py /path/to/movies /path/to/output --workers 4
```

### 多机分布式模式（三台服务器）

```bash
# 1. 修改 servers.yaml 中的 IP / 用户名 / SSH key
# 2. 同步脚本 + 启动处理
python distributed_dispatch.py \
  --input-dir /mnt/nas/movies \
  --output-dir /mnt/nas/output \
  --servers servers.yaml \
  --deploy    # 首次运行加此参数，会自动 rsync 脚本到各服务器
```

> **前提**：三台服务器都能访问同一个 NAS 存储路径（NFS/SMB），或者分别有独立的输入源。

---

## 📐 输出规格

| 参数 | 值 |
|------|------|
| 分辨率 | 1920×1080（保持宽高比，黑边补齐）|
| 帧率 | 24 fps |
| 视频编码 | H.264（NVENC GPU 优先，自动回退 libx264）|
| 音频编码 | AAC 128kbps 双声道 |
| 容器格式 | MP4（faststart，流媒体友好）|
| 文件大小 | 动态码率：目标 ≤ 2GB/文件，800kbps ~ 4Mbps |

---

## 📂 输出命名规范

```
00001_Movie_Name_abc12345.mp4
00002_Another_Film_de456789.mp4
...
```

- **5位序号**：保证排序稳定
- **清理后的原文件名**：保留可读性（最多60字符）
- **MD5前8位**：保留来源追溯能力
- **统一 .mp4 后缀**

同时会生成 `manifest.json`，记录每个文件的处理状态：
```json
[
  {
    "src_path": "/movies/action/film.mkv",
    "dst_path": "/output/00001_film_abc12345.mp4",
    "success": true,
    "skip": false,
    "error": "",
    "duration_s": 42.3
  }
]
```

---

## ⚡ 性能说明

### NVENC 硬件加速
- 使用 `h264_nvenc` 编码，速度比 CPU 快 **5-10x**
- 同时使用 `-hwaccel cuda` 硬件解码，减少 CPU 负担
- 自动回退：NVENC 失败时自动切换 libx264

### 并发策略
- 单机：每个 GPU 同时处理 1 个文件（多文件并发=GPU数量）
- 多机：按算力权重（weight）分配任务，A8000 > A6000 > 4090

### 断点续传
- 默认检查输出文件是否存在（>1KB），跳过已处理文件
- 使用 `--force` 强制重新处理

---

## 🔧 针对 PySceneDetect 的后处理建议

处理完成后，统一的 1080p 24fps 文件非常适合分镜头剪切：

```bash
# 安装 scenedetect
pip install scenedetect[opencv]

# 批量分镜头检测（示例）
for f in /output/*.mp4; do
    scenedetect -i "$f" \
        detect-content \
        split-video \
        save-images \
        -o "/scenes/$(basename $f .mp4)/"
done
```

---

## 🛠️ 常见问题

**Q: `h264_nvenc` 不可用？**  
A: 确保安装了 NVIDIA 驱动（≥520）和 CUDA，并使用 `ffmpeg-full` 版本。

**Q: 分辨率比 1080p 低的视频会 upscale？**  
A: 会，但这对训练模型有利（统一分辨率）。如不想 upscale，可修改 `needs_transcode()` 函数。

**Q: 有些视频没有音频怎么办？**  
A: 程序会自动检测，无音频的文件添加 `-an` 参数，不会报错。

**Q: 想要 4K 输出而不是 1080p？**  
A: 修改 `process_videos.py` 顶部的 `TARGET_WIDTH = 3840`, `TARGET_HEIGHT = 2160`。