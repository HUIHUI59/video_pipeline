# Stage 4 — 镜头分类（Shot Classify / Prefilter）

## 作用

对 Stage 2 输出的每个 `shot_*.mp4` 做四件事：

1. **检测人和脸**：人体 YOLOv8-Large + 人脸 **MediaPipe Tasks FaceDetector**（conf≥0.5，
   full-range；不可用时回落到 YOLOv8-face → OpenCV Haar）。所有检测跨 5 帧取 max。
2. **分类 shot_category**：`single / dominant / multi / wide / landscape`，按脸框面积比 + 人数判定
3. **画质评估**：计算亮度、对比度、清晰度，打 `quality_ok` 标记太黑/太亮/低对比/模糊的 clip
4. **镜头抖动评估**：中央 5 帧 Farneback 密集光流，平均位移 > 阈值打 `camera_shake`（issue）

输出：每部电影一个 JSONL（**prefilter manifest**），每行一个 shot 的分类 + 画质 + 抖动结果。
这个 manifest 就是 Stage 5 云端标注 + `filter_clips.py` 对外交付的**输入清单**。

## 版本历史

- **v3 (2026-04-21)**：MediaPipe 替代 Haar 成为 tier-0 face detector；新增 Farneback 光流
  `camera_motion` 字段 + `camera_shake` issue；模型文件自动下载到 `~/.cache/mediapipe-models/`。
- **v2**：加入 `num_faces`、`largest_face_ratio`、`quality_ok`、`quality_metrics`、
  `largest_subject_bbox`、`largest_subject_vertical_center`。
- **v1**：初始版本，YOLOv8 人体检测 + 规则分类。

## 代码入口

- 根目录 shim：`shot_classify.py`（可执行）
- 真正实现：`src/workers/shot_classify.py`
- 被 dispatcher 作为 Stage 4 调用时的命令构造：`src/dispatcher/distributed_dispatch.py:build_cmd_stage4`

## 输入输出

```
output_dir/clips/                      output_dir/manifest/
├── MovieA/                    ──>     ├── MovieA.jsonl
│   ├── shot_0001.mp4                  ├── MovieB.jsonl
│   └── ...                            └── ...
└── MovieB/
    └── ...
```

每部电影一个 `.jsonl`，每行一个 shot 的 JSON 记录。

## Manifest 字段完整参考表

### 一行 manifest 条目长这样

```json
{
  "shot_id": "MovieA/shot_007",
  "source_movie": "MovieA",
  "path": "clips/MovieA/shot_007.mp4",
  "num_people": 1,
  "num_faces": 1,
  "shot_category": "single",
  "duration_sec": 2.5,
  "width": 1920,
  "height": 804,
  "fps": 24.0,
  "largest_subject_ratio": 0.42,
  "largest_face_ratio": 0.18,
  "classifier_confidence": 0.88,
  "classified_at": 1713312000.0,
  "quality_ok": true,
  "quality_metrics": {
    "mean_brightness": 85.3,
    "brightness_std": 42.1,
    "sharpness": 123.5,
    "camera_motion": 0.162,
    "issues": []
  }
}
```

### 字段说明

| 字段 | 类型 | 范围 / 枚举 | 必填 | v1/v2 | 下游消费者 | 含义 |
|------|------|-----|------|-------|-----------|------|
| `shot_id` | str | `<movie_stem>/<shot_stem>` | ✓ | v1 | upload, pod_runner | 本次标注工作流的唯一标识；Stage 5 输出 JSON 会用 `<shot_stem>.json` 存盘 |
| `source_movie` | str | movie_stem | ✓ | v1 | upload, pod_runner | 源电影目录名（Stage 1 输出的 mp4 文件名去扩展名） |
| `path` | str | 相对或绝对 | ✓ | v1 | upload (本地→Pod 映射) | 指向 `shot_*.mp4` 的路径，一般 `clips/<movie>/shot_NNN.mp4` |
| `num_people` | int | ≥ 0 | ✓ | v1 | pod_runner (prompt) | **跨 5 帧取 max** 的 YOLO 人体检测计数 |
| `num_faces` | int \| null | ≥ 0 | opt | v2 | pod_runner (prompt hint) | **跨 5 帧取 max** 的人脸计数；MediaPipe Tasks FaceDetector（conf≥0.5）为首选，退回 YOLO / Haar（旧 manifest 是 Haar） |
| `shot_category` | str | `single`\|`dominant`\|`multi`\|`wide`\|`landscape` | ✓ | v1 | upload (filter), pod_runner (采 4 或 8 帧) | **核心分类结果**。规则见下文 |
| `duration_sec` | float | ≥ 0 | ✓ | v1 | pod_runner (prompt) | clip 时长（秒） |
| `width` | int | ≥ 0 | ✓ | v1 | pod_runner (prompt) | clip 宽度（像素，基本是 1920） |
| `height` | int | ≥ 0 | ✓ | v1 | pod_runner (prompt) | clip 高度（像素，21:9 是 804，16:9 是 1080） |
| `fps` | float | ≥ 0 | ✓ | v1 | pod_runner (prompt) | 帧率（基本是 24.0） |
| `largest_subject_ratio` | float | [0, 1] | ✓ | v1 | diagnostic | 最大**人体框**面积占帧比例；只是诊断信息 |
| `largest_face_ratio` | float \| null | [0, 1] | opt | v2 | pod_runner (prompt hint) | 最大**脸框**面积占帧比例；分类时 ≥0.15 判 single，≤0.03 判 wide |
| `classifier_confidence` | float | [0, 1] | ✓ | v1 | diagnostic | 跨 5 帧人/脸计数一致性得分（越高越稳定） |
| `classified_at` | float | Unix epoch 秒 | ✓ | v1 | diagnostic | 分类完成的 wall-clock 时间戳 |
| `quality_ok` | bool \| null | | opt | v2 | upload (默认过滤 `False`) | **画质评价总结**。`false` = 有一条或多条 issue；`null` = 旧 manifest 没此字段 |
| `quality_metrics.mean_brightness` | float | [0, 255] | opt | v2 | diagnostic | 跨 5 帧灰度均值的均值 |
| `quality_metrics.brightness_std` | float | [0, 255] | opt | v2 | diagnostic | 跨 5 帧灰度标准差的均值（对比度代理） |
| `quality_metrics.sharpness` | float | [0, ∞) | opt | v2 | diagnostic | 跨 5 帧 Laplacian 方差的均值（清晰度代理） |
| `quality_metrics.camera_motion` | float \| null | [0, ∞) | opt | v3 | filter_clips (抖动过滤) | 中央 5 帧 Farneback 光流平均位移（px/frame on 480-wide gray）；>6.0 判 `camera_shake` |
| `quality_metrics.issues` | list[str] | `too_dark`\|`too_bright`\|`low_contrast`\|`blurry`\|`camera_shake` | opt | v2/v3 | 人工排错 | 具体触发了哪些 issue；`camera_shake` 需要 camera_motion 字段 |

- **v1** = 从 Stage 4 诞生第一天起就有的字段，总是存在
- **v2** = 后续加的字段，在旧 manifest 里可能不存在；upload 做了向后兼容（字段是 `None` 视为 v1 不过滤）
- **v3** = MediaPipe + 光流升级（2026-04-21）后加的字段；`camera_motion` 和 `camera_shake` issue

## `shot_category` 分类规则

基于跨帧 max 的 `num_persons`、`num_faces`、`largest_face_ratio`：

| 条件 | 判定 |
|------|------|
| `num_persons == 0` | `landscape` |
| `num_persons > 0, num_faces == 0` | `wide`（有人但没脸：背对镜头或太远） |
| `num_faces == 1, face_ratio ≥ 0.15` | `single`（single close-up） |
| `num_faces == 1, face_ratio ≤ 0.03` | `wide`（单人远景） |
| `num_faces == 1, 中间` | `single`（保守归） |
| `num_faces ∈ {2, 3}, 最大脸 > 2.5×平均脸` | `dominant`（主角 + 配角） |
| `num_faces ∈ {2, 3}, 均衡` | `multi`（两三人均等） |
| `num_faces ≥ 4` | `multi`（人多就归 multi，crowd 级别留给 VLM 判定） |

阈值可通过 CLI 调：`--single-face-ratio 0.15`、`--wide-face-ratio 0.03`。

## 画质阈值

跨 5 帧的均值落到以下范围就打 issue：

| Issue | 判据 | spec 模式阈值 | strict 模式阈值 |
|-------|------|---------------|-----------------|
| `too_dark` | 亮度均值 < 阈值 | 12 | 25 |
| `too_bright` | 亮度均值 > 阈值 | 242 | 230 |
| `low_contrast` | 亮度标准差 < 阈值 | 5 | 15 |
| `blurry` | Laplacian 方差 < 阈值 | 15 | 50 |
| `camera_shake` | 中央 5 帧 Farneback 光流平均位移 > 阈值（px/frame, 480-宽灰图） | 6.0 | 6.0 |

只要命中任意一条 → `quality_ok = False`。upload.py 默认跳过所有 `quality_ok=False` 的 shot，避免浪费 H100 去标注废片。

阈值优先级：CLI flag (`--brightness-min` / `--camera-motion-max` 等) > `--quality-config` YAML > `--quality-mode spec` > 模块常量（strict）。

## 运行方式

### 1) 单机本地模式

```bash
python shot_classify.py <clips_root> <output_dir>

# 例
python shot_classify.py \
  /mnt/movies/Films/output/clips \
  /mnt/movies/Films/output \
  --workers 1
# manifest 落在 /mnt/movies/Films/output/manifest/<movie>.jsonl
```

### 2) 分布式（3 机协作）

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/forCloudKor \
  --output-dir /mnt/movies/Films/forCloudKorOutput \
  --stage 4
```

### 3) `--stage all` 自动链路

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/Baiduyun \
  --output-dir /mnt/movies/Films/output \
  --git-pull
# 等价于 --stage all，跑 1 → 2 → 4
```

## 参数速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `input_dir` (必需) | — | clips 根目录（Stage 2 输出） |
| `output_dir` (必需) | — | output_dir（manifest 落在其下 `manifest/`） |
| `--workers N` | 1 | 同机并发（YOLO 串行 GPU，1 就够） |
| `--model` | `yolov8l.pt` | 人体检测模型；首次自动下载 |
| `--face-model` | `yolov8n-face.pt` | 人脸模型路径（没装就退化到 OpenCV Haar） |
| `--person-conf` | 0.35 | 人体检测置信度阈值 |
| `--face-conf` | 0.30 | 人脸检测置信度阈值（仅 YOLO face 路径用） |
| `--single-face-ratio` | 0.15 | 最大脸框 ≥ 该值判 `single` |
| `--wide-face-ratio` | 0.03 | 最大脸框 ≤ 该值判 `wide` |
| `--sample-frames N` | 5 | 每个 clip 采多少帧（默认 15%-85% 均匀分布） |
| `--queue-dir` | — | 共享队列目录（队列模式） |
| `--worker-id` | `hostname` | 本机标识 |
| `--pid-file` | — | PID 落地路径 |
| `--log-file` | `shot_classify.log` | 日志路径 |

## 队列名

Stage 4 用 `classify_queue`。

## 脸检测后端

代码按优先级尝试：

0. **MediaPipe Tasks FaceDetector**（`blaze_face_full_range.tflite`, `min_detection_confidence=0.5`）
   —— **默认**，精度最高、假阳性少（画/雕像/反光基本不误判）
1. **本地已下载的 YOLO face 模型**（`--face-model` 指定路径存在时）—— 次选
2. **OpenCV Haar Cascade**（`opencv-python` 自带）—— 最后兜底
3. **都没有** → 降级为**只用人体检测**，所有有人的镜头都归 `wide`

worker 启动时会在日志里打印 `脸检测：使用 MediaPipe Tasks FaceDetector (...)`（或 YOLO / Haar），
**没这行就说明降级了**。

### MediaPipe 模型文件

`blaze_face_full_range.tflite` 约 1.1 MB，首次使用自动下载到：

```
$HOME/.cache/mediapipe-models/blaze_face_full_range.tflite
```

可通过环境变量 `MEDIAPIPE_FACE_MODEL_PATH` 指定别的路径（例如共享目录避免每台机器重复下）。

### MediaPipe 对 num_faces 的影响

跟老的 OpenCV Haar Cascade 相比：
- MediaPipe 的假阳性显著减少（约 10% "landscape + num_faces≥1" 的噪声去掉）
- `largest_face_ratio` 分布整体右移（bbox 更贴实脸，往往稍大 / 更稳定）
- `num_faces` 会更保守（侧脸被漏掉的比 Haar 略多，但置信阈值 0.5 可调）

## 光流 / 抖动检测（`camera_motion`）

对每个 clip 从中间位置连续采 5 帧，相邻帧对做 OpenCV Farneback 密集光流（下采样到 480 宽以省时），
相邻对的平均 L2 位移取 max 作为 `camera_motion`（单位：px/frame）。默认阈值 6.0：
高于该值加 `"camera_shake"` 到 issues，`quality_ok=False`。`--skip-motion-detect` 可跳过。

## 性能参考

- MediaPipe + YOLO + Farneback + 5 帧 + 画质 ≈ **0.7–1.5 s/clip**（稳态，冷启动约 7–11s）
- 208k clips / 3 机并行 ≈ **12–24 小时**

## 验证 manifest

```bash
# 统计各类别数量
cat /mnt/movies/Films/output/manifest/*.jsonl | python -c "
import sys, json
from collections import Counter
c = Counter()
q = Counter()
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    try: e=json.loads(line)
    except: continue
    c[e.get('shot_category','?')]+=1
    q[str(e.get('quality_ok'))]+=1
print('shot_category:', c.most_common())
print('quality_ok:', q.most_common())
"
```

健康值参考（典型剧情片）：
- `single`: 30-40%
- `dominant`: 5-10%
- `multi`: 25-35%
- `wide`: 15-25%
- `landscape`: 5-15%
- `quality_ok=True`: 80-95%

**如果全是 `landscape/wide`**：脸检测根本没跑起来（`yolov8n-face.pt` 没下载，Haar 也失败）。看启动日志的那一行"脸检测：使用 X"。

## 常见问题

**Q: manifest 里某行是空行 / 不完整的 JSON？**
A: 之前 SMB 并发写竞争留下的损伤。`upload.py` 读 manifest 时会 `[WARN] ...校验失败，跳过`，不影响正常行。修完下次 Stage 4 重跑就干净。

**Q: `classifier_confidence` 一般多少算高？**
A: 0.8+ 表示跨 5 帧计数非常稳定；0.5-0.8 表示人物进出画面；< 0.5 一般是快速切换镜头或强遮挡。这个字段 Stage 5 不用，只做人工复盘参考。

**Q: `largest_subject_ratio` 和 `largest_face_ratio` 差很多正常吗？**
A: 正常。人体框包括整个身体，脸只是头部一小块。典型 close-up：subject_ratio 0.6-0.9，face_ratio 0.15-0.30。wide 镜头：subject_ratio 0.1-0.3，face_ratio 0.01-0.05。

**Q: 我想让画质阈值严一点（多筛掉一些差的）？**
A: 改 `src/workers/shot_classify.py` 顶部的四个 `QUALITY_*` 常量，push 代码，重跑 Stage 4。注意**放宽没代价，收紧会扔掉真数据**。

**Q: Stage 4 跑完后我的 manifest 字段数量不对？**
A: 老 manifest（v1）字段少 4 个（`num_faces`、`largest_face_ratio`、`quality_ok`、`quality_metrics`）。Pydantic 的 `ManifestEntry` 把它们设为 `Optional`，所以**新旧混用不报错**。但你用 v1 manifest 走 Stage 5 时，upload 的画质过滤会失效（`quality_ok=None` 不过滤，全部上传）。建议 Stage 4 重跑得到 v2 manifest 再上云。

## 相关代码位置

| 功能 | 文件:行 |
|------|---------|
| 主入口 | `src/workers/shot_classify.py:main()` |
| 分类核心 | `src/workers/shot_classify.py:classify_one()` |
| YOLO 人体检测 | `src/workers/shot_classify.py:_detect_boxes()` |
| Haar / YOLO face | `src/workers/shot_classify.py:get_face_detector()` / `detect_faces()` |
| 画质计算 | `src/workers/shot_classify.py:_compute_quality()` |
| Pydantic 模型 | `src/runpod/schemas.py:ManifestEntry` + `ManifestQuality` |
| 画质阈值常量 | `src/workers/shot_classify.py:QUALITY_MIN_BRIGHTNESS` 等 |
| dispatcher 调用 | `src/dispatcher/distributed_dispatch.py:build_cmd_stage4()` |

## 下一步

Manifest 生成完 → [Stage 5 云端标注](05_labeling.md)。
