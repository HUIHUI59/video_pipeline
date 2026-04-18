# Stage 5 VLM 输出 JSON 截断

- **记录日期**：2026-04-18
- **状态**：已选方案 E（两轮标注），待实现
- **相关文件**：`src/runpod/pod_runner.py`、`src/runpod/schemas.py`、`configs/runpod.yaml`
- **相关 spec**：`docs/labelingStandards/external_delivery_v1/`

## 现象

Stage 5 在 Runpod H100 80G Pod 上跑 Qwen3-VL-32B-Instruct 标注时，部分 shot（尤其 2 人以上）输出 JSON 被 `max_tokens` 截断：

```
[ERR parse] .../shot_0035: Expecting ',' delimiter: line 871 column 8 (char 34247)
```

`char 34247` 约 8500 tokens，撞到 `max_tokens=8192` 后半截没了 → `json.loads` 失败 → 整个 shot 归入 `_failed/`。

## 根因

**不是 GPU 过载**（H100 80G 满载是正常工作状态），**是 token 预算不够**：

| 项 | Token | 说明 |
|---|---|---|
| `max_model_len` 上限 | 16384 | 再大 KV cache 就爆 80GB（加模型权重 67GB 刚好占满） |
| Input: system(800) + user(1600) + 8 帧 × ~350 vision tokens | ~5200 | 固定开销，压缩余地小 |
| Output 预算 = 16384 − 5200 | **~11200** | 硬上限 |
| 实测 2 人 shot 输出 | **8500-15000+** | **经常超** |

**为什么输出这么长**——`external_delivery_v1` schema 本身就丰富：

- 每 person：`face_analysis`（15+ 字段）+ `body_analysis`（14+ 字段）+ `alternative_captions`（4 版描述）
- `observable_blendshape_hints` 一个对象 15 子字段
- `facial_components` / `facial_attributes` / `upper_body_detail` 各 7-8 子字段
- 自由 string 无 `maxLength` → 模型爱写多长写多长
- 两人 × 2

vLLM 的 `structured_outputs` 只保证**语法合法**，**不会在 max_tokens 用完前强制闭合 `{}`**，所以直接断尾。

## 方案对比

| 方案 | 改动量 | 效果 | 标准符合性 |
|---|---|---|---|
| A. `json-repair` 兜底 | 小（加 pip + 2 行 try/except） | 救 70-90% 截断 shot | 不动 schema ✓ |
| B. prompt 硬要求简洁 | 小（追加一段 system prompt） | 减 30-50% token，不保证听话 | 不动 schema ✓ |
| C. schema 加 maxLength | 中（改 delivery_v1 schema） | 模型被硬卡住 | **需和标注方沟通 ✗** |
| D. schema 瘦身 | 大（砍字段） | 根治 | **违反 delivery 标准 ✗** |
| **E. 两轮标注** | 中 | 每轮 output <5K token 稳过 | ✓ 最严谨 |

## 当前选择：E（两轮标注）

### 理由

- **最严谨**：每轮 output 在合理预算内，不依赖 json-repair 这种事后打补丁的启发式。
- **输出质量更好**：模型一次聚焦一个子任务（先 body / 再 face），每个子任务 prompt 更具体、few-shot 更对口，理论上质量 > 一次性全出。
- **标准不动**：和 `external_delivery_v1` 完全对齐，标注方侧不用改。
- **代价可接受**：推理成本约翻倍，但 H100 一小时几刀可控；换来可靠性、可复现性、可调试性更优。

### 设计草稿（待实现时细化）

- **Round 1**：输出 `shot_context` + `persons[].{person_index, spatial_position, body_analysis}` + `interaction` + `quality_flags` + `usability_score` + `meta`。
  - 去掉 `face_analysis`、`alternative_captions`
  - 每 shot 输出预算 ≈ 3000-4500 tokens
- **Round 2**：只对 Round 1 成功的 shot，输出 `persons[].face_analysis`（按 `person_index` 对齐回去）。
  - Prompt 专注 face，用 crop 过的脸部 patch 而不是整帧（减 vision token）
  - 每 shot 输出预算 ≈ 2500-4000 tokens
- **合并**：pod_runner 或本地 post-process 脚本把两份 JSON 合并成完整 delivery_v1 格式。
- **校验**：合并后跑 delivery_v1 的 `validate_body_analysis.py`，不符合 `errors=0` 的进 `_failed/`。

### 涉及改动

- `src/runpod/pod_runner.py`：拆成 `--round 1|2` 参数，对应不同的 system_prompt 和 output schema。
- 新 `src/runpod/merge_rounds.py`：合并两轮输出。
- `configs/runpod.yaml`：`sampling.max_tokens` 按 round 分别配置。
- 可能需要新 `docs/labelingStandards/external_delivery_v1/scripts/build_vlm_prompt.py` 的调用参数选 body-only / face-only。

### 已否决方案的复测时机

A、B 先留着作为 quick-fix，**如果 E 上完还有极端长 shot 截断**，再叠加 A 做最终兜底。

## 目前的临时措施

- `max_tokens` = 10240（`max_model_len=16384` − input 5200 留余量）
- `max_model_len` = 16384（KV cache 刚好塞进 80G）
- structured_outputs 启用（保证语法合法）
- 截断的 shot 归入 `output/_failed/` 不污染合格数据

这能把截断率从 100% 压到可能 20-40%，但不是真正解决，只是凑合跑着。

## 待办

- [ ] 设计 Round 1 / Round 2 的 output schema 子集定义
- [ ] 和 delivery_v1 维护者确认是否允许把 `face_analysis` 作为"可选二次输出"
- [ ] 实现 `pod_runner.py --round` 参数
- [ ] 实现 `merge_rounds.py`
- [ ] 跑一个完整 demo（比如 20 个 shot）对比 one-shot vs two-round 的质量 + 成本
