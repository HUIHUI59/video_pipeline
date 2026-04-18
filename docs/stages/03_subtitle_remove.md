# Stage 3 — 字幕去除（Subtitle Remove，可选）

## 状态：**默认不跑**

Stage 3 **不在 `--stage all` 默认链路里**（默认只跑 1→2→4）。它是独立的可选模块。只有当 Stage 2 产出的 clips **有硬字幕**且你需要训练数据**不带字幕**时，才显式触发 `--stage 3`。

当前项目的数据源是无字幕电影，所以这一步一般不需要。

## 作用

对 Stage 2 输出的每个 clip 调用 [**video-subtitle-remover (VSR)**](https://github.com/YaoFANGUK/video-subtitle-remover) 做硬字幕擦除：

- 只在帧**底部 25%** 做检测（Stage 3 的优化，省 75% 检测开销）
- 基于 inpainting 把字幕像素替换成邻近帧推断出来的背景
- 保持 Stage 2 clips 的目录层级

## 代码入口

- 根目录 shim：`subtitle_remove.py`（可执行）
- 真正实现：`src/workers/subtitle_remove.py`
- VSR 本身是 **外部项目**，需要先单独装到 `~/video-subtitle-remover`
- dispatcher 调用：`src/dispatcher/distributed_dispatch.py:build_cmd_stage3`

## 输入输出

```
output_dir/clips/                        output_dir/clean/
├── MovieA/                      ──>     ├── MovieA/
│   ├── shot_0001.mp4                    │   ├── shot_0001.mp4      (底部字幕擦除)
│   └── ...                              │   └── ...
└── MovieB/                              └── MovieB/
    └── ...                                  └── ...
```

## 前置安装（**只需一次**）

VSR 不是 pip 包，得 clone + 装它自己的依赖。**每台要跑 Stage 3 的机器都要装**。

```bash
bash tools/setup_vsr_env.sh
# 这个脚本会：
#   1. 在 ~/miniconda3/envs 下建 conda env `vsr`（Python 3.12）
#   2. git clone https://github.com/YaoFANGUK/video-subtitle-remover ~/video-subtitle-remover
#   3. pip install paddlepaddle-gpu==... + torch + 其他 VSR 依赖
#   4. 应用 tools/patch_vsr.sh 里的补丁（开启底部检测 + MJPG 中间帧 + ffmpeg 参数调优）

# 验证
conda activate vsr
cd ~/video-subtitle-remover
python -c "from backend import main as vsr_main; print('VSR ready')"
```

## 运行方式

### 1) 单机本地模式

```bash
conda activate vsr
python subtitle_remove.py \
  <clips_root> <clean_root> \
  --vsr-dir ~/video-subtitle-remover

# 例
python subtitle_remove.py \
  /mnt/movies/Films/output/clips \
  /mnt/movies/Films/output/clean \
  --vsr-dir ~/video-subtitle-remover \
  --workers 1
```

### 2) 分布式（3 机协作）

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/forCloudKor \
  --output-dir /mnt/movies/Films/forCloudKorOutput \
  --stage 3
```

Stage 3 的 `--vsr-dir` 来源优先级：
1. CLI 参数 `--vsr-dir` （覆盖一切）
2. `configs/servers.yaml` 里该机的 `vsr_dir` 字段
3. 默认 `~/video-subtitle-remover`

## 参数速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `input_dir` (必需) | — | clips 根目录（Stage 2 输出） |
| `output_dir` (必需) | — | clean 输出根目录 |
| `--vsr-dir` | `~/video-subtitle-remover` | VSR 项目 clone 路径 |
| `--workers N` | 1 | 同机并发（默认 1，VSR 本身已占满 GPU） |
| `--queue-dir` | — | 共享队列目录 |
| `--worker-id` | `hostname` | 本机标识 |
| `--pid-file` | — | PID 落地路径 |
| `--log-file` | `subtitle_remove.log` | 日志路径 |

## 队列名

Stage 3 使用 `subtitle_queue`。与 Stage 2 的 `split_queue`、Stage 4 的 `classify_queue` 互不干扰。

## 断点续传

- 已存在且**非空**的输出文件自动跳过（看 `output_dir/<movie>/shot_0001.mp4` 大小 > 0 就不再重跑）
- 结合共享队列：队列 `done` 状态记忆电影级别的完成度

## 性能参考（RTX4090 单卡）

- 2-5 秒的 shot → 去字幕约 10-30 秒（视分辨率和字幕复杂度）
- 一部电影 ~5000 shots → 约 **15-30 小时**（慢！是目前最耗时的 stage）
- 这也是为什么它是可选模块——没硬字幕就别跑，浪费 GPU 时间

## 常见问题

**Q: 看 `subtitle_remove.log` 报 `ModuleNotFoundError: paddle`？**
A: 没在 `vsr` conda env 里跑，或者 `setup_vsr_env.sh` 中途失败。重跑：
```bash
conda activate vsr
bash tools/setup_vsr_env.sh
```

**Q: 去完字幕的视频底部有模糊色块？**
A: VSR 的 inpainting 效果取决于字幕周围的背景纹理。纯色背景几乎无痕，高纹理（草地、格纹）会留痕迹。目前没有完美解决方案，可人工抽查 `clean/<movie>/shot_*.mp4` 的前几十个 shot，效果差的整部电影建议不跑 Stage 3（保留原始 clips）。

**Q: VSR 检测区域为什么只有底部 25%？**
A: 见 `tools/patch_vsr.sh`——这是我们加的优化，因为影视字幕 95% 以上都在底部。想改回全帧检测的话改 `tools/patch_vsr.sh` 里的 `y_start_ratio`。

**Q: 跑了一小时后 GPU 显存爆了？**
A: VSR 有显存泄漏的倾向。我们的 worker 每处理完一批就释放推理对象。如果还 OOM：
- 减小 `--workers`（最小 1）
- 给 VSR 降批大小：改它的 config_demo.py
- 或者每 N 个 shot 重启 VSR 进程（目前未实现）

**Q: 我的视频没硬字幕，有必要跑吗？**
A: **不要跑**。Stage 3 是为有硬字幕素材设计的，无字幕跑它既浪费时间又可能在底部引入小量画质损伤。

## 相关代码位置

| 功能 | 文件:行 |
|------|---------|
| 主入口 | `src/workers/subtitle_remove.py:main()` |
| 单片去字幕 | `src/workers/subtitle_remove.py:remove_one()` |
| VSR 调用 | `src/workers/subtitle_remove.py`（内部 subprocess 启动 VSR） |
| dispatcher 调用 | `src/dispatcher/distributed_dispatch.py:build_cmd_stage3()` |
| VSR 环境安装 | `tools/setup_vsr_env.sh` |
| VSR 补丁 | `tools/patch_vsr.sh` |

## 下一步

去完字幕 → [Stage 4 镜头分类](04_shot_classify.md)。Stage 4 的 `--input-dir` 改成 `clean/` 目录就行。
