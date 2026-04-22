# Stage 5 Pod Control UI — P0 设计文档

**Date:** 2026-04-22
**Status:** Draft, pending user sign-off before implementation
**Owner:** HUIHUI59
**Module:** `src/pod_control/` (new, separate from `src/review/`)

## 1. 目标与边界

为 Stage 5（Qwen3-VL delivery_v1 推理）提供一个本地 Web 控制面板，**替换当前纯
shell 驱动的工作流**：`00_push_code / 01_push / 02_run / 03_pull / 99_kill`。

**P0 定位：launcher + monitor。** UI 只负责数据准备、SSH 参数管理、触发现有
`scripts/runpod/run_all.sh` 与 `99_kill.sh`、实时 tail 日志。上传 / 推理 / 下载逻辑
**仍由现有 shell 脚本执行**，不改 runpod 流水线。

**工作流顺序（关键）：**

```
Prepare (offline)        Rent pod            Run                    Monitor
    │                        │                 │                       │
    ├─ 选 movie               ├─ 配 SSH profile ├─ pick batch + pod    ├─ tail log (2s)
    ├─ 套 filter              ├─ test connect   ├─ subprocess spawn    ├─ checkpoint count
    ├─ 抽样预览               │                 │                       ├─ kill button
    └─ save batch (.json)    │                 │                       │
                             ▼                 ▼                       ▼
                        data/pod_control/pods.yaml    data/pod_control/state.json
                        data/pod_control/batches/*.json
                                                data/pod_control/runs/<id>/
```

**数据准备 vs pod 连接解耦**：可以在还没租服务器的时候把 N 个 batch 都准备好
（filter + preview 确认），pod 一上线直接 launch。

## 2. 非目标 (YAGNI)

- ❌ per-shot checkbox 选择（P0 用 category + quality filter 已足够）
- ❌ 预生成缩略图（浏览器 HTML5 `<video>` poster frame 就行）
- ❌ runpod.yaml 参数的 UI form 编辑（P0 里就手改 yaml）
- ❌ multi-pod 并行（P0 单 run slot）
- ❌ WebSocket 日志（2 秒 polling 够 MVP 用）
- ❌ 多用户 / auth（localhost-only 工具）
- ❌ 重写 upload.py / download.py 为纯 Python（P0 继续 shell 脚本）
- ❌ 与 `src/review/` 合并（不同关注点：review 是后 QA，pod_control 是前执行）

## 3. P0 页面

1. **Prepare** (offline, 不需 SSH)
   - 读 `manifest/*.jsonl` 列出 movie + 分类统计
   - filter 表单：categories, skip_bad_quality, skip_landscape, max_shots
   - paginated 预览列表（20/page），每行内嵌 HTML5 `<video>`
   - "Random sample 10" 按钮重置 seed 抽 10 条
   - Save batch → `data/pod_control/batches/<name>.json`
   - Batch 列表可 view / delete（只允 delete status=ready 的）

2. **Pods** (租 pod 之后)
   - 列 `data/pod_control/pods.yaml` 里的 SSH profile
   - Add / Edit / Delete
   - **Test connect** → `ssh -o ... pod exit`，显示 ok/latency/err

3. **Run**
   - 下拉选 batch + pod + (可选) preset yaml path
   - **Launch** → subprocess spawn `run_all.sh`；写 PID/state；batch.status → running
   - 单 run slot：active_run 存在时 Launch 被拒（409）

4. **Monitor** (active_run 存在时常显)
   - Log tail：`ssh pod 'tail -c +OFFSET pod_runner.log'`，2s 轮询
   - Checkpoint counter：10s 轮询解析 `.checkpoint.jsonl`
   - **Kill**：`ssh pod 'bash 99_kill.sh'` + local `os.killpg`

## 4. 架构

### 模块布局

```
src/pod_control/
├── __init__.py
├── __main__.py         # python -m src.pod_control --port 8765 --data-root DIR
├── api.py              # FastAPI app + routes
├── store.py            # 文件系统持久化 (JSON/YAML, fcntl lock)
├── filter.py           # Stage 4 manifest filtering — 复用 src/runpod/upload.py
├── ssh.py              # subprocess-ssh 封装（test connect + tail log）
├── runner.py           # 本地 subprocess orchestration (spawn run_all.sh, PID, kill)
└── static/
    ├── index.html
    ├── styles.css
    └── app.js          # plain ES modules，无构建
```

### 数据目录

```
data/pod_control/                   # 通过 --data-root 指定，默认项目 data/pod_control
├── pods.yaml                       # SSH profile 列表
├── batches/<name>.json             # 每个 batch 一个文件
├── state.json                      # active_run（单 slot） + 历史 run 列表
└── runs/<run_id>/
    ├── stdout.log                  # 本地 subprocess 捕获
    └── pod_tail.log                # pod 端 log 的本地缓存（带 offset）
```

### 分层责任

| 模块 | 只做 | 不做 |
|---|---|---|
| `api.py` | HTTP 路由 + 请求校验 + 调 store/filter/ssh/runner | 不直接读写文件 / 起子进程 |
| `store.py` | 唯一能写盘的模块（fcntl lock）| 不处理业务逻辑 |
| `filter.py` | 导入 `src.runpod.upload._filter_entries` 做筛选 | 不新写 filter semantics |
| `ssh.py` | `subprocess.run(["ssh", ...])` 包装 | 不用 paramiko/asyncssh |
| `runner.py` | `subprocess.Popen(..., preexec_fn=os.setsid)` + PID 跟踪 + `os.killpg` | 不写持久状态（走 store） |

### 复用现有代码

- `src/runpod/upload.py::_filter_entries` + `_iter_manifest_lines` — filter 逻辑单一真值
- `src/runpod/upload.py::_ssh_opts` — SSH 连接参数约定（StrictHostKeyChecking=accept-new + known_hosts）
- `src/review/api.py` — FastAPI + StaticFiles mount 模板
- `src/review/store.py` — 文件追加写持久化模板
- `scripts/runpod/run_all.sh` / `99_kill.sh` — 推理 / kill 不重实现

### 并发模型

- **单 run slot**：`state.json.active_run` Optional[RunRecord]
- fcntl lock 写 `state.json`（run 生命周期转换时）
- 读端点无锁
- Run 子进程：`os.setsid()` 做 process group leader → kill 时能一锅端 ffmpeg / ssh / python 子孙进程

## 5. API + 数据模型

### REST endpoints (`/api/*`)

```
GET    /api/movies                          # [{movie, total_shots, by_category, quality_ok_count}]
GET    /api/movies/{movie}/preview?…        # filter + page + sample_seed → {shots:[…], total, page}

POST   /api/batches                         # {movie, filter_params, name} → Batch
GET    /api/batches                         # list
GET    /api/batches/{name}                  # one
DELETE /api/batches/{name}                  # 仅 status=ready

GET    /api/pods
POST   /api/pods
PUT    /api/pods/{name}
DELETE /api/pods/{name}
POST   /api/pods/{name}/test                # {ok, latency_ms, msg}

POST   /api/runs                            # {batch_name, pod_name, preset_path?} → RunRecord
GET    /api/runs/active                     # RunRecord | null
GET    /api/runs                            # history (last 20)
GET    /api/runs/{id}/tail?offset=N         # {text, next_offset, checkpoint:{done,failed,pending}}
POST   /api/runs/{id}/kill

GET    /                                    # index.html
GET    /static/*                            # static files
GET    /clips/{movie}/{shot}.mp4            # 代理 Stage 2 clips（preview 用）
```

### Pydantic 模型

```python
class FilterParams(BaseModel):
    categories: list[str] = ["single", "dominant", "multi"]
    skip_bad_quality: bool = True
    skip_landscape: bool = True
    max_shots: int | None = None

class Batch(BaseModel):
    name: str                    # slug, unique
    movie: str
    filter_params: FilterParams
    shot_count: int
    status: Literal["ready", "running", "done", "failed"]
    created_at: float
    last_run_id: str | None = None

class PodProfile(BaseModel):
    name: str
    host: str
    user: str
    ssh_key: str
    port: int = 22
    workspace: str
    last_test_ok: bool | None = None
    last_test_at: float | None = None

class RunRecord(BaseModel):
    id: str
    batch_name: str
    pod_name: str
    preset_path: str | None
    started_at: float
    ended_at: float | None
    status: Literal["running", "done", "failed", "killed"]
    pid: int | None
    pod_log_offset: int = 0
    exit_code: int | None

class ActiveState(BaseModel):
    active_run: RunRecord | None
    history: list[RunRecord] = []
```

### 错误响应

统一 envelope：`{"error": {"code": "…", "message": "…"}}`

Code 列表：`invalid_filter`, `movie_not_found`, `batch_exists`, `batch_not_found`,
`batch_not_ready`, `pod_unreachable`, `pod_not_found`, `run_already_active`,
`run_not_found`, `internal_error`.

## 6. 测试计划

| 层 | 内容 | 工具 |
|---|---|---|
| `store.py` | batches/pods/state round-trip + fcntl 并发 | pytest + tmp_path |
| `filter.py` | 合成 manifest → filter → 计数 / 分布 | pytest |
| `api.py` | 每条 route happy-path + 错误码 | fastapi TestClient |
| `ssh.py` | mock `subprocess.run` 返回码 / 输出 | pytest + unittest.mock |
| `runner.py` | mock `subprocess.Popen` + `os.killpg`，验证 PID 跟踪 | pytest + mock |
| E2E | 真实 pod | 租 pod 后手测 |

覆盖率目标：`store / filter / api` ≥ 80%。`ssh / runner` 以 smoke 为主（都是 subprocess wrapper，真机最可靠）。

## 7. 实施里程碑 (commit-worthy)

每个里程碑都应包含对应测试、都可独立跑、都能 commit：

1. **M1 Scaffold**: 模块目录 + `__main__` + 空 FastAPI serve index.html + pytest skeleton
2. **M2 Store + Filter**: `store.py` + `filter.py` + 单元测试 (no UI)
3. **M3 Prepare page**: movies/preview/batches endpoints + 前端（movie picker / filter form / preview 列表 + save batch）
4. **M4 Pods page**: pod CRUD + test-connect + 前端
5. **M5 Run page**: launch / active-run lock / kill endpoint + 前端
6. **M6 Monitor panel**: tail endpoint + 2s polling + checkpoint counter + run history

M6 完成 = P0 完成，可以租 pod 跑 end-to-end smoke。

## 8. 不改的东西（显式承诺）

- `docs/labelingStandards/external_delivery_v1/` 仍是 read-only baseline
- `src/runpod/` 流水线（pod_runner / upload / download）不改
- `scripts/runpod/*.sh` 不改
- `src/review/` 不合并不动
- `configs/runpod.yaml` 的格式不变

## 9. 风险 + 预案

1. **subprocess run_all.sh 挂起 / 僵尸** — 用 `os.setsid` + `os.killpg`，同 `process_videos.py` 范式
2. **pod 断连时 tail 打不到** — tail 端点应能返回 `{text: "", next_offset: N, pod_unreachable: true}` 而非 500
3. **state.json 竞争写** — fcntl lock + 原子 rename（`*.tmp` → `*`）
4. **preview 视频 seek 跨 VLAN 慢** — `<video preload="metadata">` 只加 poster；实际播放才 load
5. **filter.py 与 upload.py 语义不一致** — 通过直接 import 消除；若 upload.py filter 签名将来变，这里会编译时炸出来

---

**签核状态：** 待用户确认，然后进入 M1 scaffold。
