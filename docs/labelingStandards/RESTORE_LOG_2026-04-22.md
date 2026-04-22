# `external_delivery_v1/` Restore Log — 2026-04-22

## What

`docs/labelingStandards/external_delivery_v1/` 整个目录已**从 `external_delivery_v1.zip` 重新解压**，
覆盖之前两个 commit (`d9441cc` 和 `2084a44`) 对其中 3 个脚本所做的修改。

按用户要求：spec 目录是 **read-only baseline**，任何"合规性"修改都应该发生在
`src/runpod/`，不能改 spec 本身。

## 验证

| 项 | 验证结果 |
|---|---|
| 文件总数 | 21 ✓ (与 zip 一致) |
| `*:Zone.Identifier` Windows 元数据 | 0 ✓ (清理完毕) |
| 16 个 docs/yaml/json 文件 | 与 zip byte-identical ✓ |
| `python -m src.runpod.schemas` | 9 pass / 0 fail ✓ |

## 被回退的 3 个脚本

恢复前的版本备份在 `/home/leo4090/spec_pre_restore_backup_2026-04-22/`（不进 git）。

### 1. `scripts/build_vlm_prompt.py` — 23 行 diff

**改动 #1 (perf)**：加 `@functools.lru_cache(maxsize=8)` 缓存 YAML 加载

| 项 | 详情 |
|---|---|
| 类型 | 性能优化 |
| spec 是否需要 | ✗ 不需要（spec 关心规则不关心实现） |
| 是否影响输出 | ✗ 不影响 |
| 是否 port 到 src/runpod/ | ✓ 需要 — pod_runner 在 batch 推理时连续调用，加缓存避免重复 YAML 解析 |
| Phase C 任务编号 | C9（性能注释类） |

**改动 #2 (behavior)**：删除 `forbidden_terms[:30]` 的截断，注入完整列表

| 项 | 详情 |
|---|---|
| 类型 | 行为变更 — VLM 看到的 forbidden camera term 数量从 30 变成全部 |
| spec 文档（§5.3）说 | 修改版注释引用："VLM 必须看到所有被禁止的 camera 术语" |
| zip 版（恢复后）行为 | 截断到前 30 个 |
| 是否 port 到 src/runpod/ | ✓ pod_runner 调 `build_system_prompt` 后**自己注入完整列表**，不修改 spec 函数行为 |
| Phase C 任务编号 | C5（prompt 增强） |

### 2. `scripts/normalize_tags.py` — 95 行 diff

**改动**：新增 `_strip_camera_terms_text()` 工具方法 + `normalize_shot()` 把 camera_terms_forbidden
扩散到 8 类自由文本字段（之前只作用于 `body_analysis.action_primary`）

被扩展的字段：
- `face_analysis.expression_caption`
- `face_analysis.alternative_captions.{direct,literary,direction,situational}`
- `body_analysis.{motion_caption, gesture_detail}`
- `body_analysis.alternative_captions.{direct,literary,direction,situational}`
- `body_analysis.upper_body_detail.{head, neck, shoulders, arms, hands, torso}`

| 项 | 详情 |
|---|---|
| 类型 | 行为扩展 — normalize_tags 从只改 action_primary 变成清理所有自由文本 |
| spec 文档（§5.3）说 | 修改版注释引用："camera 术语不应污染 captions/summary 等" |
| zip 版（恢复后）行为 | 仅清理 `body_analysis.action_primary` |
| 风险 | 直接改自由文本是有损操作，正则边界要保守（修改版用 `(?<![A-Za-z0-9])` word boundary） |
| 是否 port 到 src/runpod/ | ✓ — `post_normalize.py` 加 `strip_camera_terms_in_captions(label)` 函数，作为 spec normalize_tags 之后的额外步骤运行 |
| Phase C 任务编号 | C3（流水线整合） |

### 3. `scripts/validate_body_analysis.py` — 88 行 diff

**改动 #1**：新增 CHECK 15 — `_check_top_interaction_consistency`
顶层 `interaction.count='solo'` 时 `interaction.contact` 必须是 `'none'`，违反则 ERROR。

**改动 #2**：新增 CHECK 16 — `_check_interaction_symmetry`
`persons[].body_analysis.interaction.interacts_with_person_index` 应当对称引用，违反则 WARNING。

| 项 | 详情 |
|---|---|
| 类型 | 业务规则扩展 — 14 → 16 checks |
| spec 文档说 | 修改版注释引用 "§ 7.2 / § 5.4"，未必对应 zip 版 spec 的明确条款 |
| zip 版（恢复后）行为 | 14 个 CHECK，无 solo+contact 一致性 / 对称性检查 |
| 是否 port 到 src/runpod/ | ✓ — pod_runner 在 spec ShotValidator 之后追加 2 个本地检查；不改 spec validator |
| Phase C 任务编号 | C2（流水线 + interaction enum 兼并） |

## 流向（Phase C 路线）

```
spec scripts (zip baseline, read-only)
       │
       │  调用 / import
       ▼
src/runpod/
   ├── pod_runner.parse_and_validate(raw)
   │     1. spec normalize_tags.normalize_shot(raw)        ← zip 行为
   │     2. post_normalize.fix_all(raw)                    ← pod 端兜底
   │        ├── enforce_altcap_null_consistency            ← 新加 (Phase C6)
   │        ├── strip_camera_terms_in_captions             ← port from removed (C3)
   │        └── 既有 fix_interaction / fix_emotion / ...
   │     3. ShotLabel.model_validate(raw)
   │     4. spec ShotValidator.validate(obj)               ← zip 14 CHECKs
   │     5. _pod_extra_checks(obj)                         ← port CHECK 15/16 (C2)
   │
   └── prompt 构造（C5/C7/C8）
         ├── 调 spec build_system_prompt(...)
         ├── 追加 forbidden_terms 完整列表（不截断 30）
         ├── 追加 blendshape 5-class enum
         └── R1 / R2 / R3 round-specific prompt 用 spec 常量构造
```

## 后续执行

1. 提交本日志 + 恢复后的 spec 目录
2. 按 plan Phase C1-C12 改 `src/runpod/`
3. 每个 C 项 commit 时引用本日志的"Phase C 任务编号"列
