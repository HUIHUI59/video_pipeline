# Stage 2 — 镜头切分（Scene Split）

## 作用

把 Stage 1 输出的统一规格 mp4（一部电影一份，90-180 分钟）按**镜头边界**切成小片段。一部电影典型产出 2000-6000 个 shot clip。

- 检测算法：PySceneDetect 的 `ContentDetector`（基于帧间 HSV 差）
- **去掉首尾各 10 个镜头**（默认 `--trim-shots 10`）——避开片头 logo、片尾 credits，都不是有用的叙事镜头
- 如果一部电影总镜头数 ≤ 2×trim_shots，则全部保留（不切掉）

## 代码入口

- 根目录 shim：`scene_split.py`（可执行）
- 真正实现：`src/workers/scene_split.py`
- 被 dispatcher 作为 Stage 2 调用时的命令构造：`src/dispatcher/distributed_dispatch.py:build_cmd_stage2`

## 输入输出

```
output_dir/                                 output_dir/clips/
├── MovieA_a3f2c18b.mp4           ──>       ├── MovieA_a3f2c18b/
├── MovieB_7d9e0f12.mp4                     │   ├── shot_0001.mp4
└── MovieC_5b8c4e9a.mp4                     │   ├── shot_0002.mp4
                                            │   └── ...
                                            ├── MovieB_7d9e0f12/
                                            │   ├── shot_0001.mp4
                                            │   └── ...
                                            └── MovieC_5b8c4e9a/
                                                └── ...
```

**重要**：Stage 2 扫描输入目录时，会**跳过** `clips/`、`clean/`、`.queue/` 这三个 pipeline 自产子目录（防止把自己切出来的 shot 再当电影切一遍）。

## 运行方式

### 1) 单机本地模式

```bash
python scene_split.py <stage1_output_dir> <clips_output_dir>
# 最常见写法：Stage 1 输出目录当输入，clips/ 当输出
python scene_split.py /mnt/movies/Films/output /mnt/movies/Films/output/clips

# 首尾各切 20 个镜头（默认是 10）
python scene_split.py /mnt/movies/Films/output /mnt/movies/Films/output/clips --trim-shots 20
```

### 2) 分布式（3 机协作）

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/Baiduyun \
  --output-dir /mnt/movies/Films/output \
  --stage 2
```

这里 `--input-dir` Stage 2 不读（它读 `--output-dir` 底下的 mp4），但 dispatcher 接口要求传一下。

或者一次性跑 Stage 1 + 2 + 4：
```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/Baiduyun \
  --output-dir /mnt/movies/Films/output \
  --git-pull
# --stage all 默认 1 → 2 → 4
```

## 参数速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `input_dir` (必需) | — | 输入视频目录，自动跳过 clips/ clean/ .queue/ 子目录 |
| `output_dir` (必需) | — | clips 根目录（每部电影一个子文件夹） |
| `--workers N` | 2 | 并发切分数（每路走独立 ffmpeg） |
| `--trim-shots N` | 10 | 首尾各去掉的镜头数 |
| `--queue-dir` | — | 共享队列目录（队列模式） |
| `--worker-id` | `hostname` | 本机标识 |
| `--pid-file` | — | PID 落地路径 |
| `--log-file` | `scene_split.log` | 日志路径 |

## 队列名

Stage 2 使用 `split_queue`（存在 `<queue_dir>/split_queue.json`）。和 Stage 1 的 `pipeline_queue.json`、Stage 3 的 `subtitle_queue.json`、Stage 4 的 `classify_queue.json` 各自独立。

## 切分算法说明

PySceneDetect 的 `detect(src, ContentDetector())` 返回 `[(start_ts, end_ts), ...]` 的镜头列表。然后 `split_video_ffmpeg(src, scenes, output_dir, output_file_template="shot_$SCENE_NUMBER.mp4")` 走 ffmpeg `-ss` / `-to` stream-copy 切出每段。

**不重新编码**——每个 shot 继承 Stage 1 输出的 H.264 + AAC，所以切分非常快（IO 限制，不是 CPU/GPU）。

## 单 shot 典型时长

- 2-10 秒：~60-70% (标准叙事镜头)
- 10-30 秒：~20-25% (长镜头 / 对白)
- 30+ 秒：~5-10% (静态场景 / 演员独白)
- < 2 秒：被 ContentDetector 算法合并进相邻 shot，一般不会单独出现

## 性能参考

- 单部 1080p 电影 ≈ **1-3 分钟**（PySceneDetect 的 HSV 差分是扫帧操作，瓶颈在磁盘读取）
- 200 部电影 × 2 min / 2 机并行 ≈ 3-4 小时

## 常见问题

**Q: 为什么 Stage 2 跑完看到 clips/ 下有重复文件夹（带 `-<hash>` 后缀）？**
A: 不应该有。如果有，可能是 Stage 1 出了两份 uuid 不同的同名电影 mp4。去重建议：
```bash
ls /mnt/movies/Films/output/clips/ | sort | head
# 人工确认后删除重复的文件夹
```

**Q: 一部电影切出几千个 shot 正常吗？**
A: 动作片常见 4000-6000 个 shot，文艺片 1500-3000 个。超过 8000 极可能是 ContentDetector 阈值太敏感（默认就行，一般别调）。

**Q: 切出来的 shot 不能播放？**
A: `ffprobe output/clips/<movie>/shot_0001.mp4` 看有没有错。一般是 Stage 1 输出的母带有问题（末尾帧损坏），切到最后一个 shot 时失败。手动删除该 mp4 即可。

**Q: 想先跳过首尾 5 个镜头怎么办？**
A: `--trim-shots 5`。但电影片头 logo 和片尾 credits 通常需要至少 8-10 个 shot 才能跳过，不建议调低。

**Q: 某部电影没切出任何 shot？**
A: 看 `scene_split.log`，多半是：
- 源视频太短（<30s）
- 纯黑/纯白开头太长导致 ContentDetector 没触发
- 源视频编码 PySceneDetect 解不出（极少见）

## 相关代码位置

| 功能 | 文件:行 |
|------|---------|
| 主入口 | `src/workers/scene_split.py:main()` |
| 单片切分 | `src/workers/scene_split.py:split_one()` |
| 递归扫描 + 跳过自产目录 | `src/workers/scene_split.py:scan_videos()` |
| dispatcher 调用 | `src/dispatcher/distributed_dispatch.py:build_cmd_stage2()` |

## 下一步

切完之后 → [Stage 4 镜头分类](04_shot_classify.md)。
如果视频有硬字幕需要去除 → [Stage 3 字幕去除](03_subtitle_remove.md)（可选）。
