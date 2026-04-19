# Stage 5 — 云端标注（Labeling via Runpod + Qwen3-VL-32B）

## 作用

把 Stage 4 产出的 manifest 作为输入清单，**上云（Runpod H100 Pod）**跑 **Qwen3-VL-32B-Instruct + vLLM guided_json**，对每个镜头产出一份严格符合 `external_delivery_v1` schema 的标注 JSON（面部表情 + 身体动作 + 场景上下文）。

跟前四个阶段的本质区别：
- **不走 dispatcher**（dispatcher 是为本地三机设计的）
- 运行在**临时租来的 Runpod Pod** 上
- 由 `scripts/runpod/*.sh` 驱动：push → run → pull → Terminate

## 必读：Runpod 使用手册

> 📖 **[docs/RUNPOD_MANUAL.md](../RUNPOD_MANUAL.md)** — 第一次用 Runpod 必读。涵盖注册、SSH、Network Volume（**关键省钱步骤**）、Pod 选型、Stop vs Terminate、费用估算、12 类常见坑排错。
> 本文档只讲 Stage 5 **代码侧**的工作流；Pod 管理看那边。

## 代码入口

| 端 | 文件 | 作用 |
|---|---|---|
| 本地 | `src/runpod/upload.py` | 读 manifest + 筛选 → rsync clips 到 Pod |
| 本地 | `src/runpod/download.py` | rsync Pod 的 output/*.json 回本地 + schema 校验 |
| 本地 | `src/runpod/schemas.py` | Pydantic v2 模型，镜像 delivery_v1 的 `ShotLabel` / `ManifestEntry` |
| Pod 内 | `src/runpod/pod_runner.py` | 加载模型 + 读 manifest + 跑 VLM + 后处理 + 写 JSON |
| Pod 内 | `tools/pod_setup.sh` | Pod 第一次启动时装 vLLM + 下载 Qwen3-VL-32B（67GB） |
| 驱动脚本 | `scripts/runpod/01_push.sh` | 推代码 + clips + manifest + delivery_v1 bundle 到 Pod |
| 驱动脚本 | `scripts/runpod/02_run.sh` | SSH 进 Pod，跑 setup + pod_runner |
| 驱动脚本 | `scripts/runpod/03_pull.sh` | 拉 Pod 的 output 回本地 + Pydantic 校验 |
| 驱动脚本 | `scripts/runpod/run_all.sh` | 一键 push → run → pull |

## 输入输出

**输入**：Stage 4 产出的 manifest JSONL
```
output/manifest/<movie>.jsonl    # 每行一个 ManifestEntry
```

**输出**：每个 shot 一份标注 JSON
```
output/labels/<movie>/<shot_stem>.json
# 结构符合 docs/labelingStandards/external_delivery_v1/docs/json_schema_integrated.md
```

每份 JSON 顶层字段（详细见规范文档）：
- `shot_id`, `source_movie` — 跟 manifest 对齐
- `shot_context.{shot_type, shot_emotion_summary, shot_motion_summary, scene_context}` — 镜头整体
- `persons[]` — 每个可见人物一条：`face_analysis`（9 类情绪 + 4 种 caption + 14 个 blendshape hint）+ `body_analysis`（动作分类 + 4 种 motion caption + upper_body_detail）
- `interaction` — 多人互动关系
- `quality_flags` — VLM 对 clip 画质的评估
- `usability_score` — face / motion 可用度评分
- `meta` — 模型名、版本、用了几帧、推理耗时

## 完整工作流（高层次）

```
本地                                    Runpod Pod
┌──────────────────────────┐           ┌──────────────────────────────┐
│ 1. Stage 4 manifest ready│           │                              │
│ 2. configs/runpod.yaml   │           │                              │
│    填 Pod host/port/key  │           │                              │
├──────────────────────────┤  rsync    │                              │
│ 3. 01_push.sh            │ ────────> │ /workspace/labeling/         │
│    - 按 filters 筛 shot  │           │   clips/  manifest.jsonl     │
│    - push clips+代码     │           │   src/runpod/  tools/        │
│    - push delivery_v1/   │           │   delivery_v1/               │
├──────────────────────────┤   ssh     │                              │
│ 4. 02_run.sh             │ ────────> │ 4a. 首次 tools/pod_setup.sh  │
│    - ssh 进 Pod          │           │     装 vLLM + 下 67GB 模型   │
│    - tail 日志           │           │ 4b. nohup pod_runner.py      │
│                          │           │     每个 shot:               │
│                          │           │     → 采 4 或 8 帧           │
│                          │           │     → vLLM guided_json       │
│                          │           │     → normalize_tags         │
│                          │           │     → ShotLabel.validate     │
│                          │           │     → ShotValidator.validate │
│                          │           │     → 写 output/<m>/<s>.json │
│                          │           │     → 追加 .checkpoint.jsonl │
├──────────────────────────┤  rsync    │                              │
│ 5. 03_pull.sh            │ <──────── │ output/<movie>/*.json        │
│    - 拉 JSON 回本地       │           │                              │
│    - Pydantic 校验        │           │                              │
│                          │           │                              │
│ 6. 本地批量复核           │           │                              │
│    跑官方 validator       │           │                              │
│                          │           │                              │
│ 7. Runpod 网页 Terminate │           │                              │
│    Pod（只留 Network Vol）│           │                              │
└──────────────────────────┘           └──────────────────────────────┘
```

## 配置：`configs/runpod.yaml`

从模板拷：
```bash
cp configs/runpod.yaml.example configs/runpod.yaml
vim configs/runpod.yaml
```

关键字段：

| 字段 | 说明 |
|---|---|
| `pod.{host, port, user, ssh_key}` | 每次 Deploy 新 Pod 都要刷 host/port |
| `paths.local_clips_root` | Stage 2 的 clips 根目录 |
| `paths.local_manifest_dir` | Stage 4 的 manifest 目录 |
| `paths.local_labels_root` | Stage 5 结果本地落地位置 |
| `paths.pod_workspace` | Pod 里的工作目录，一般 `/workspace/labeling`（挂在 Network Volume 上） |
| `model.name` | 32B：`Qwen/Qwen3-VL-32B-Instruct`；122B：`Qwen/Qwen3.5-122B-A10B-Instruct-AWQ` |
| `model.precision` | `bf16` (32B 推荐) / `fp8` (省显存) / `awq` / `awq_marlin` / `gptq-int4` (量化模型) |
| `model.quantization` | 量化模型必填，传给 vLLM（如 `awq_marlin`）；bf16/fp8 留 `null` |
| `model.limit_mm_per_prompt.image` | 单次 chat 最多图数，32B 可 16，122B 建议 8 |
| `model.tensor_parallel_size` | `1` = 单卡；多卡 TP（122B 有时要 2）设 `>1` |
| `filters.shot_categories` | `[single, dominant]` 只标人物特写；`[]` = 全部除 landscape |
| `filters.movies` | `[]` = 全部；或 `[MovieA, MovieB]` |
| `filters.max_shots` | `null` = 全量；数字可限量调试 |
| `sampling.*` | temperature 0.2、top_p 0.9、max_tokens_round1/round2 10240、repetition_penalty 1.05 |

## 模型与 GPU 选型

项目同时支持两种 VLM；命令行 `--config` 切换：

```bash
# 32B 默认
python -m src.runpod.upload --config configs/runpod.yaml
python -m src.runpod.pod_runner --config runpod.yaml

# 122B AWQ 主力
python -m src.runpod.upload --config configs/runpod.122b.yaml
python -m src.runpod.pod_runner --config runpod.122b.yaml
```

### GPU 租用矩阵（Runpod 2026-04 行情参考）

| 模型 | 权重 | 最低 GPU | 推荐 GPU | 备注 |
|---|---|---|---|---|
| **Qwen3-VL-32B BF16** (默认) | ~64 GB | 1× H100 80GB | 1× H200 141GB | `max_model_len=16384` 留 ~12 GB KV |
| **Qwen3-VL-32B FP8** | ~35 GB | 1× L40S 48GB | 1× H100 80GB | 质量接近 BF16，显存对半 |
| **Qwen3.5-122B-A10B AWQ-INT4** | ~68 GB | 1× H100 80GB（紧） | 1× H200 141GB **或** 2× A100 80GB TP=2 | MoE 总 122B 驻留；active 10B，单 token 比 dense 32B 更快 |
| **Qwen3.5-122B-A10B FP8** | ~122 GB | 2× H100 80GB TP=2 | 2× H200 141GB | 不推荐；AWQ 已够 |
| **Qwen3.5-122B-A10B BF16** | ~234 GB | 3× H100 80GB TP=3 | 4× H100 80GB | 仅作参考 |

**122B AWQ 在 80GB H100 上注意事项**：
- `max_model_len=16384` + `limit_mm_per_prompt.image=8` 是上限，想要 16 图必须上 H200 或 TP=2
- `gpu_memory_utilization=0.92` 给 KV cache 留约 6 GB（80×0.92 − 68）
- MoE 256 experts 全驻留，所以权重按 122B 计算；但每 token 只激活约 10B，吞吐接近 dense 10B

### 租到 Pod 后快速验证显存

```bash
# 加载模型后立刻退出，打印实际显存占用。不花推理钱。
ssh pod "python src/runpod/pod_runner.py --config runpod.122b.yaml --dry-run-model-load"
```

输出类似 `加载完成后显存 72.34 GB`，可立即判断"是否挤得下 KV"。

## 运行步骤

### 首次（包括装 vLLM + 下模型，约 30-40 分钟）

```bash
# 1. 确保 Stage 4 的 manifest 已生成
ls /mnt/movies/Films/output/manifest/
# 应该看到 <movie>.jsonl

# 2. 配 Pod 凭据
cp configs/runpod.yaml.example configs/runpod.yaml
# 填 host/port/ssh_key

# 3. 小批试跑
bash scripts/runpod/01_push.sh configs/runpod.yaml --max-shots 5
bash scripts/runpod/02_run.sh
# 前 30-40 分钟是装环境 + 下模型；之后 1-2s/shot

# 4. 看 5 个 shot 都合格了
bash scripts/runpod/03_pull.sh
ls /mnt/movies/Films/output/labels/<movie>/
cat /mnt/movies/Films/output/labels/<movie>/shot_0001.json | head -40

# 5. 全量跑（模型已在 Network Volume，启动快）
bash scripts/runpod/run_all.sh
```

### 后续批次（模型已缓存）

```bash
# 1. Runpod 网页 Deploy 新 Pod（挂同一 Network Volume）
# 2. 更新 configs/runpod.yaml 的 host/port
# 3. 一键跑
bash scripts/runpod/run_all.sh
# 4. Runpod 网页 Terminate
```

## 对齐 external_delivery_v1 的后处理链

每个 shot 在 Pod 里跑完 VLM 后依次走 **4 层校验**：

1. **JSON 解析** — VLM 输出用 `guided_json=ShotLabel.model_json_schema()` 解码，已经保证 JSON 合法
2. **normalize_tags** — `docs/labelingStandards/external_delivery_v1/scripts/normalize_tags.py::TagNormalizer`
   - 动词归一：`walks → walking`，`sat → sitting`
   - 同义词归一：`strolling → walking`，`grabbing → grasping`
   - 删除 action 字段里泄漏的镜头术语（`close-up, wide, handheld, ...` 等 30+ 条）
   - intensity / tone 归轴（例：从 action_primary 移到 action_quality.intensity）
3. **Pydantic 结构校验** — `src/runpod/schemas.py::ShotLabel.model_validate(obj)` 确保类型 / 字段 / 枚举符合
4. **业务校验（16 项）** — `validate_body_analysis.py::ShotValidator.validate(obj)`
   - 镜头术语泄漏、action_primary 不在 taxonomy、null 规则不一致、caption 字数超纲、跨字段矛盾、
     顶层 interaction 自洽（solo → contact=none）、跨人 interaction 对称性…… 合计 16 类
   - `errors=0` 才合格；warnings 允许（只记日志）

任何一步失败都把 raw 输出扔进 `output/_failed/<slug>.raw.txt` + `<slug>.errors.json`，**不**追加到 `.checkpoint.jsonl`——下次重启会再试。

## 交付前本地批量复核（最后一道关）

`03_pull.sh` 拉完结果后，用官方 validator 再过一遍：

```bash
python docs/labelingStandards/external_delivery_v1/scripts/validate_body_analysis.py \
  --input /mnt/movies/Films/output/labels/<movie>/ \
  --taxonomy docs/labelingStandards/external_delivery_v1/docs/motion_taxonomy.yaml \
  --synonyms docs/labelingStandards/external_delivery_v1/docs/motion_synonyms.yaml \
  --output /mnt/movies/Films/output/labels/<movie>/validation_report.json
```

合格标准：`errors = 0`。`validation_report.json` 会列出所有 warnings 和 infos（分布统计），适合人工抽样复核。

## 断点续传

- Pod 内 `pod_runner.py` 每写完一个 shot 追加 `<shot_id>` 到 `/workspace/labeling/output/.checkpoint.jsonl`
- 重启 `pod_runner.py`（或重新 `02_run.sh`）时自动跳过 checkpoint 里的 shot
- 合格的 shot 在 `output/<movie>/<shot_stem>.json`；失败的原始 raw 在 `output/_failed/<slug>.raw.txt`

## 成本估算（参考）

| 项 | 估算 |
|---|---|
| H100 80GB Community On-Demand | ~$2.49/hr |
| 首次 setup + 下模型 | ~30 min ≈ $1.25 |
| 每 shot 推理（VLM + 校验） | ~2-3 s |
| 250,000 shots 推理 | ~140 hrs ≈ **$350** |
| Network Volume 200GB | **$14 / 月**（模型一次下载终身用） |

优化策略详见 `docs/RUNPOD_MANUAL.md § 10`。

## 参数速查（upload.py）

| CLI | 说明 |
|---|---|
| `--config configs/runpod.yaml` | 必需 |
| `--shot-category single,dominant` | 覆盖 filters.shot_categories |
| `--movies MovieA,MovieB` | 覆盖 filters.movies |
| `--max-shots 20` | 覆盖 filters.max_shots（小批调试） |
| `--include-bad-quality` | 即使 quality_ok=False 也上传 |
| `--dry-run` | 只打印 rsync 命令不实际执行 |

## 常见问题

**Q: 0 条符合筛选条件？**
A: 看 Stage 4 manifest 的 `shot_category` 分布。如果全是 `landscape/wide` 说明 Stage 4 的脸检测没跑起来（见 [04_shot_classify.md](04_shot_classify.md) 的"脸检测后端"一节）。

**Q: vLLM OOM？**
A: `configs/runpod.yaml` 的 `model.precision: bf16` 改 `fp8`，省一半显存。

**Q: Pod 断开怎么办？**
A: Spot Pod 随时可能被收回，On-Demand 很稳。收回后 Deploy 新 Pod，挂同一 Network Volume，`02_run.sh` 会从 checkpoint 续跑。

**Q: 大规模上传 125GB clips 要一晚上？**
A: Runpod S3-compat 中转方案：见 `docs/RUNPOD_MANUAL.md § 11.5`。

**Q: 想看 Pod 日志但 tail 退出了？**
A: Pod 里日志在 `/workspace/labeling/output/pod_runner.log`（tail 只是本地镜像，退出不影响 nohup 进程）。
```bash
ssh -i ~/.ssh/id_ed25519 -p <port> root@<host> 'tail -100 /workspace/labeling/output/pod_runner.log'
```

## 相关代码位置

| 功能 | 文件:行 |
|------|---------|
| Pod 主循环 | `src/runpod/pod_runner.py:main()` |
| 采帧 | `src/runpod/pod_runner.py:_sample_frames()` |
| prompt 构造（待集成 build_vlm_prompt） | `src/runpod/pod_runner.py:_user_prompt()` |
| Pydantic schema | `src/runpod/schemas.py:ShotLabel` |
| 本地 push | `src/runpod/upload.py:main()` |
| 本地 pull | `src/runpod/download.py:main()` |
| Pod 环境安装 | `tools/pod_setup.sh` |
| 交付规范 | `docs/labelingStandards/external_delivery_v1/docs/README_external.md` |
| 官方 prompt 构造 | `docs/labelingStandards/external_delivery_v1/scripts/build_vlm_prompt.py` |
| 官方后处理 | `docs/labelingStandards/external_delivery_v1/scripts/normalize_tags.py` |
| 官方校验 | `docs/labelingStandards/external_delivery_v1/scripts/validate_body_analysis.py` |
