# 流水线阶段总览

本项目把"任意格式电影文件 → 可供视频生成模型训练的标注数据"拆成 **5 个阶段**。每一阶段都是独立的 worker，通过共享文件锁队列 (`src/common/task_queue.py`) 协调，可以单独跑、中断续跑、多机并行。

```
┌──────────┐   ┌──────────┐   ┌─────────────┐   ┌──────────────────┐
│ Stage 1  │──>│ Stage 2  │──>│ Stage 4     │──>│ Stage 5          │
│ 转码     │   │ 镜头切分 │   │ 分类+过滤   │   │ VLM 云端标注     │
└──────────┘   └──────────┘   └─────────────┘   └──────────────────┘
                     │
                     └────> [可选] Stage 3 去字幕

数据流：
  *.mkv/mp4/... ──Stage 1──> output/*.mp4 (统一规格)
                 ──Stage 2──> output/clips/<movie>/shot_NNN.mp4
   (可选)        ──Stage 3──> output/clean/<movie>/shot_NNN.mp4  (去硬字幕)
                 ──Stage 4──> output/manifest/<movie>.jsonl (分类 + 画质 + 抖动)
   (本地到云)    ──Stage 5──> output/labels/<movie>/<shot_stem>.json (Qwen3-VL 标注)
```

## 各阶段详细文档

| 阶段 | 文档 | 代码入口 | 运行位置 |
|------|------|----------|----------|
| [1] 转码 | [01_transcode.md](01_transcode.md) | `src/workers/process_videos.py` | 本地 3 机 |
| [2] 镜头切分 | [02_scene_split.md](02_scene_split.md) | `src/workers/scene_split.py` | 本地 3 机 |
| [3] 字幕去除（可选） | [03_subtitle_remove.md](03_subtitle_remove.md) | `src/workers/subtitle_remove.py` | 本地 3 机 |
| [4] 镜头分类 | [04_shot_classify.md](04_shot_classify.md) | `src/workers/shot_classify.py` | 本地 3 机 |
| [5] 云端标注 | [05_labeling.md](05_labeling.md) | `src/runpod/pod_runner.py` | Runpod H100 Pod |

## 调度器入口

分布式调度器 `src/dispatcher/distributed_dispatch.py`（根目录 shim: `distributed_dispatch.py`）统一管理 Stage 1 / 2 / 3 / 4 在本地 3 机的运行。Stage 5 是**独立工作流**，不经过 dispatcher，由 `scripts/runpod/*.sh` 驱动。

- `--stage all` 默认跑 **1 → 2 → 4**（跳过 3）
- `--stage 3` 或 `--stage 4` 显式指定单阶段
- `--stage 5` **不存在**；Stage 5 看 `docs/RUNPOD_MANUAL.md` 和 `05_labeling.md`

## 各阶段间的数据契约

**Stage 1 → Stage 2**
- Stage 1 输出：`<output_dir>/*.mp4`（单个大文件，每部电影一份）
- Stage 2 输入：同目录 `*.mp4`

**Stage 2 → Stage 4**（或 Stage 2 → Stage 3 → Stage 4）
- Stage 2 输出：`<output_dir>/clips/<movie_stem>/shot_NNN.mp4`（按镜头切好的小片段）
- Stage 3（可选）输出：`<output_dir>/clean/<movie_stem>/shot_NNN.mp4`（底部字幕被擦除）
- Stage 4 输入：默认读 `<output_dir>/clips/` 下的 `shot_*.mp4`

**Stage 4 → Stage 5**
- Stage 4 输出：`<output_dir>/manifest/<movie_stem>.jsonl`（每个 shot 一行 JSON，字段见 [04_shot_classify.md](04_shot_classify.md)）
- v3 manifest 增加：`num_faces`（MediaPipe FaceDetector, conf≥0.5）、`largest_face_ratio`、`quality_metrics.camera_motion`（Farneback 光流）、`issues` 包含 `camera_shake`
- Stage 5 输入：同 manifest JSONL；`src/runpod/upload.py` 按 `shot_category` 和 `quality_ok` 过滤，rsync 到 Pod

**Stage 5 输出**
- `output/labels/<movie_stem>/<shot_stem>.json`（每个 shot 一份 Qwen3-VL-32B 产出的完整 `ShotLabel` 结构）
- 对外交付前用 `docs/labelingStandards/external_delivery_v1/scripts/validate_body_analysis.py` 复核，`errors=0` 才算合格

## 关键规范文档（不属于代码）

- `docs/labelingStandards/external_delivery_v1/docs/README_external.md` — 对外交付方的操作手册
- `docs/labelingStandards/external_delivery_v1/docs/json_schema_integrated.md` — 标注 JSON 的权威 schema
- `docs/labelingStandards/external_delivery_v1/docs/motion_taxonomy.yaml` — 动作词汇表（67 条动词 × 6 大类）
- `docs/labelingStandards/external_delivery_v1/docs/motion_synonyms.yaml` — 同义词归一 + 镜头术语黑名单
- `docs/labelingStandards/external_delivery_v1/docs/vlm_prompts/examples/*.json` — 9 份 few-shot 示例标注

## 运行环境速查

| 机器 | 角色 | conda env | 关键路径 |
|------|------|-----------|----------|
| RTX4090_local (主控+worker) | dispatcher + Stage 1/2/3/4 worker | `vsr` | `~/video_pipeline` |
| A6000 (remote worker) | Stage 1/2/3/4 worker | `vsr` | `~/video_pipeline` |
| A8000 (remote worker) | Stage 1/2/3/4 worker | `vsr` | `~/video_pipeline` |
| Runpod H100 Pod (临时) | Stage 5 标注 | 自建 venv `/opt/labeling-env` | `/workspace/labeling` |

详细硬件和共享盘挂载见 `docs/README.md` 的"环境准备"章节。

## 也要读的跨阶段文档

- [`docs/README.md`](../README.md) — 项目总览、目录结构、环境准备
- [`docs/RUNPOD_MANUAL.md`](../RUNPOD_MANUAL.md) — Runpod 账号注册到 Pod 销毁的完整手册（Stage 5 必读）
