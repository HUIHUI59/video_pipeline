# Stage 1 — 转码（Transcode）

## 作用

把来源电影（任意容器、任意分辨率、任意码率）统一转成：

| 参数 | 值 |
|------|------|
| 分辨率 | 宽度 1920px，高度按原始比例等比缩放（21:9 / 2.39:1 都不加黑边） |
| 帧率 | 24 fps |
| 视频编码 | H.264（NVENC GPU 优先，自动回退 libx264） |
| 音频 | AAC 128 kbps 双声道 |
| 容器 | MP4，`+faststart`（利于流式播放） |
| 码率 | 动态：4K→8 Mbps，1080p→3.5 Mbps，720p→2 Mbps。单文件上限 2 GB |
| 命名 | `{原名前6字}_{uuid8}.mp4` |

## 代码入口

- 根目录 shim：`process_videos.py`（可执行）
- 真正实现：`src/workers/process_videos.py`
- 被 dispatcher 作为 Stage 1 调用时的命令构造：`src/dispatcher/distributed_dispatch.py:build_cmd_stage1`

## 输入输出

```
input_dir/                    output_dir/
├── MovieA.mkv         ──>    ├── MovieA_a3f2c18b.mp4
├── MovieB.mp4         ──>    ├── MovieB_7d9e0f12.mp4
└── MovieC.ts          ──>    └── MovieC_5b8c4e9a.mp4
```

**保证**：单次运行幂等——已处理过的源文件不会重复转码（通过共享队列 `pipeline_queue.json` 的 `done` 状态记忆）。

## 运行方式

### 1) 单机本地模式（最简单，不走队列）

```bash
python process_videos.py ./input ./output
python process_videos.py ./input ./output --workers 1           # 限并发 1 路
python process_videos.py ./input ./output --no-nvenc            # 强制 CPU 编码（libx264）
python process_videos.py ./input ./output --gpu-ids 0,1         # 指定 GPU
```

### 2) 分布式（3 机协作，走 dispatcher）

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/Baiduyun \
  --output-dir /mnt/movies/Films/output \
  --stage 1
```

或配合其他阶段一起跑：
```bash
# --stage all 默认按顺序跑 1 → 2 → 4
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/Baiduyun \
  --output-dir /mnt/movies/Films/output \
  --git-pull
```

### 3) 单机加入分布式队列补位

```bash
conda activate vsr
cd ~/video_pipeline
python process_videos.py /mnt/movies/Films /mnt/movies/Films/output \
  --queue-dir /mnt/movies/Films/output/.queue \
  --worker-id 我的机器名
```

## 参数速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `input_dir` (必需) | — | 输入视频目录，递归扫描所有视频扩展名 |
| `output_dir` (必需) | — | 输出 mp4 落地位置 |
| `--workers N` | 2 | 同机并发转码数（每路独占一个 GPU） |
| `--gpu-ids 0,1` | `0` | NVENC 使用的 GPU 编号列表 |
| `--no-nvenc` | off | 关闭 GPU 编码，强制 libx264（慢但省 GPU） |
| `--resume` | off | 强制从上次中断处续跑（队列模式下自动） |
| `--force` | off | 全部重新转码（忽略已完成） |
| `--dry-run` | off | 只扫描、打印计划，不实际转码 |
| `--queue-dir` | — | 共享队列目录；指定则进入**队列模式**和其他 worker 协作 |
| `--worker-id` | `hostname` | 本机在队列里的标识 |
| `--pid-file` | — | PID 落地路径（dispatcher 杀进程用） |
| `--log-file` | `pipeline.log` | 日志文件路径 |

## 进程组 kill 机制

`process_videos.py` 在模块顶部就调用 `os.setsid()`，让自己成为进程组 leader (PGID == PID)。所有 ffmpeg 子进程继承同一组。dispatcher 用：
```bash
kill -TERM/-KILL -- -$PGID
```
一条命令就能**一次杀干净** Python 进程 + 所有 ffmpeg 子进程。

## NVENC 回退

如果第一次 ffmpeg 命令用 NVENC 失败（某些特殊颜色空间、编码器不支持的格式），worker 会自动用 libx264 重跑一次并打 WARNING 日志。这是为什么有的 clip 速度明显慢——它走了 CPU 路径。

## 输出命名说明

形如 `MovieA_a3f2c18b.mp4`：
- `MovieA` — 源文件名前 6 个字符（防止中文文件名造成路径问题）
- `a3f2c18b` — `uuid.uuid4().hex[:8]`，保证唯一

如果两部电影前 6 字符相同，UUID 保证文件名仍唯一。

## 常见问题

**Q: 为什么 `nvidia-smi` 看 GPU 利用率低？**
A: 多台机器同时读同一块共享盘，IO 会成瓶颈。用 `iotop` / `iostat` 查一下网络盘吞吐。另外 NVENC 本身就是硬件单元，GPU 计算核心利用率看起来不高但 NVENC 满载。

**Q: 某些视频一直失败？**
A: 看 `pipeline.log` 里该文件的 ERR。常见：
- 源文件损坏（ffprobe 都读不出 duration）→ 手动从 input 目录剔除
- 源文件编码极冷门（VC-1 等）→ `--no-nvenc` 用 libx264 试试

**Q: 进度显示 `done=X claimed=0 pending=Y`，Y 一直不动？**
A: 所有 worker 可能都卡在单台机器最大并发。`python distributed_dispatch.py --status --output-dir ...` 看哪台 worker 的 claimed 数不为 0。

**Q: 转完的 mp4 怎么验证？**
A: `ffprobe output/SomeMovie_xxx.mp4` 看：
- `width: 1920`
- `r_frame_rate: 24/1`
- `codec_name: h264`
- `audio: aac, 2 channels, 128000 bps`

## 性能参考（RTX4090 单机）

- 1080p 电影 90 min → 转码 ≈ 6-8 分钟（NVENC）
- 4K 电影 90 min → 转码 ≈ 12-15 分钟（NVENC）
- libx264 回退路径 → 视 CPU 核数，一般慢 3-5 倍

## 相关代码位置

| 功能 | 文件:行 |
|------|---------|
| 主入口 | `src/workers/process_videos.py:main()` |
| ffmpeg 命令构造 | `src/workers/process_videos.py:build_cmd()` |
| 码率计算 | `src/workers/process_videos.py:calc_br()` |
| 单任务处理 | `src/workers/process_videos.py:process_one()` |
| dispatcher 调用 | `src/dispatcher/distributed_dispatch.py:build_cmd_stage1()` |

## 下一步

转完之后进 [Stage 2 镜头切分](02_scene_split.md)。
