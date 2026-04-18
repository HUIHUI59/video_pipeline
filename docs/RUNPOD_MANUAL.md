# Runpod 使用手册（Stage 5 云端标注）

> 本手册假设你**第一次**用 Runpod，完整覆盖：注册 → 密钥 → 租 Pod → 推数据 → 跑推理 → 拉结果 → 关机 → 计费。
> 每一步都有"为什么"和"别踩坑"说明。目标：**最低费用跑完 250k 镜头标注**。
>
> 配合使用：`configs/runpod.yaml`、`src/runpod/*.py`、`scripts/runpod/*.sh`、`tools/pod_setup.sh`

---

## 目录

- [0. TL;DR 省钱要点](#0-tldr-省钱要点)
- [1. 核心概念（30 秒看懂计费）](#1-核心概念30-秒看懂计费)
- [2. 账号准备](#2-账号准备)
- [3. 添加 SSH 公钥](#3-添加-ssh-公钥)
- [4. 创建 Network Volume（**关键省钱步骤**）](#4-创建-network-volume关键省钱步骤)
- [5. 租 Pod（选 H100 80GB）](#5-租-pod选-h100-80gb)
- [6. 连接 Pod（SSH）](#6-连接-podssh)
- [7. 首次环境部署（pod_setup.sh）](#7-首次环境部署pod_setupsh)
- [8. 日常工作流（push → run → pull）](#8-日常工作流push--run--pull)
- [9. 关机 vs 终止 Pod（**别搞错！**）](#9-关机-vs-终止-pod别搞错)
- [10. 费用估算 & 省钱清单](#10-费用估算--省钱清单)
- [11. 常见坑 & 故障排查](#11-常见坑--故障排查)

---

## 0. TL;DR 省钱要点

只看这 8 条就能避开 80% 的坑：

1. **用 Network Volume** 存模型权重（$0.07/GB/月），不要每次开 Pod 都重下 67GB 模型。
2. **用 Community Cloud** 而不是 Secure Cloud（便宜 15%~20%），批量任务够用。
3. Pod 用完**点 Terminate（销毁），不是 Stop（暂停）**。Stop 状态下的 Pod 还在扣"存储费"（容器盘），不干活也烧钱。模型已经在 Network Volume 上，重新 Terminate 再开不丢数据。
4. **开 Pod 前先准备好要推的数据**（Stage 4 manifest 已完成 + clips 已分镜头）。开了 Pod 立刻开始推数据 + 跑，不要让 H100 空转。
5. **用 runpod.yaml 里的 `filters.shot_categories` 和 `filters.max_shots` 先小批量试跑**（`max_shots: 20`），验证流程通顺再全量跑。
6. **跑完一批 Terminate**。下次再开新 Pod，Network Volume 里的模型和代码都在，重建环境 ~3 分钟就能继续。
7. 睡觉前、吃饭前**检查 Pod 是不是还在跑**。Runpod 不会因为你忘记就自动停——**忘关一台 H100 一晚上 ≈ $60**。
8. 先 On-Demand 跑通全流程，**再考虑 Spot**（便宜 40-50%，随机被收回，配合 checkpoint 断点续跑可承受）。

---

## 1. 核心概念（30 秒看懂计费）

### Pod（容器实例）
你租的就是这个——一台带 GPU 的 Linux 容器。**只要状态是 Running，就按秒收钱。**

### 两种 Pod 类型

| | Community Cloud | Secure Cloud |
|---|---|---|
| 定价 | 便宜（H100 ~$2.49/hr） | 贵 15-25%（H100 ~$2.99/hr） |
| 可靠性 | 宿主机可能偶尔下线 | SLA 保证 |
| 网络 | 普通 | 更快 |
| **推荐** | **我们这个场景用这个** | 企业用 |

### 两种硬盘

| 类型 | 位置 | 持久性 | 计费 |
|---|---|---|---|
| **Container Disk**（容器盘） | Pod 自己的 `/` | Pod 一 Terminate 就**全丢** | 免费（含在 Pod 里） |
| **Network Volume**（网盘） | 挂成 `/workspace` | **独立于 Pod，Pod 销毁后还在** | $0.07/GB/月 |

**关键**：模型权重 67GB 放 Network Volume，**一次下载终身使用**。

### 两种租赁模式

| | On-Demand | Spot/Interruptible |
|---|---|---|
| 价格 | 基准价 | 便宜 40-60% |
| 会被打断 | ❌ | ✅ 随时可能被收回 |
| **推荐** | **先用这个** | 跑通后再切，配合 checkpoint 续跑 |

### Pod 生命周期状态（**记住 Stop ≠ Terminate**）

```
Creating → Running → Stopped → Running → ... → Terminated (永久销毁)
             ↑                    ↑
          按 GPU 秒数计费        不计 GPU 费
                               但仍按容器盘和 Network Volume 计费
```

- **Stop**：GPU 释放，但容器磁盘保留（还在收"存储费" ~$0.1/hr）。重新 Start 快（几十秒）。
- **Terminate**：整台 Pod 连带容器盘**物理删除**。Network Volume 不受影响。重开要 ~2 分钟。

**我们的策略**：**永远 Terminate，不 Stop**。模型在 Network Volume，Terminate 不丢；代码从本地 rsync 上去也秒推。

---

## 2. 账号准备

### 2.1 注册
https://runpod.io → Sign Up。推荐 Google 登录快一点。

### 2.2 充值（Credits）
Runpod 是预付费：先充钱，跑的时候扣。

- 右上角账号 → Billing → Add Credits
- 建议先充 **$20-50** 试试，跑通再加
- 支持信用卡 / PayPal
- **坑**：充的钱**不能退**。第一次别充太多。

### 2.3 信用卡自动续费（可选）
Setting → Automatic Payments → 打开。余额低到 $10 自动充。**第一次跑完全部流程前保持手动**——容易在忘记关 Pod 时被持续扣费。

---

## 3. 添加 SSH 公钥

这是**唯一一次**手动配置，之后所有 SSH 登录都靠这个。

### 3.1 本地生成密钥（如果没有）

```bash
# 在你的主控机（RTX4090_local 或任意 Linux 开发机）
ls -la ~/.ssh/id_ed25519.pub
# 如果已存在就跳到 3.2

ssh-keygen -t ed25519 -C "runpod" -f ~/.ssh/id_ed25519
# 一路回车，不设 passphrase（设了的话脚本自动化会一直问密码）
```

### 3.2 拷贝公钥内容

```bash
cat ~/.ssh/id_ed25519.pub
# 输出类似：ssh-ed25519 AAAAC3Nz... runpod
```

### 3.3 粘贴到 Runpod

Runpod 网页 → 右上角头像 → **Settings** → 左侧 **SSH Public Keys** → **Add Public Key**

- Name：随便，比如 `home_main`
- Public Key：粘贴完整一行 `ssh-ed25519 AAAA...`
- Save

**坑 1**：Runpod 的 SSH key 是**全账户共享**的，以后开任何 Pod 都自动带这个 key，不用每次配。

**坑 2**：Pod 默认用户是 `root`，不是你 Linux 系统用户名。ssh 要用 `root@<ip>`。

---

## 4. 创建 Network Volume（**关键省钱步骤**）

这是省钱最多的一步。不做的话每次开 Pod 都要重下 67GB 模型（每次 ~$1-2 + 20 分钟）。

### 4.1 创建

Runpod 网页 → 左侧菜单 → **Storage** → **+ New Network Volume**

| 字段 | 填什么 |
|---|---|
| Name | `labeling-vol` |
| Data Center | 选跟你打算租 H100 的机房**同一个**（比如 `US-CA-2` 或 `EU-RO-1`）。**必须同机房**。先看哪个机房 H100 便宜（§5），再按那个选 |
| Size | **200 GB**（67GB 模型 + clips 缓存 + 输出余量）。以后可以扩容不能缩 |

创建后列出：`labeling-vol  200 GB  US-CA-2  ~$14/month`

- 200GB × $0.07/GB/月 = **~$14/月**
- 当你不租 Pod 时，这个 $14 还是在扣——但相比每次开 Pod 重下 67GB（~$1 GPU + 带宽），几天就回本

### 4.2 Network Volume 的寿命

- 不需要时 → 网页点 **Delete**（删前先 `03_pull.sh` 把结果拉回本地）
- 跑完几百部电影的最终批次后就删掉省 $14/月

---

## 5. 租 Pod（选 H100 80GB）

### 5.1 打开 Deploy 页面

左侧菜单 → **Deploy** → 顶部切到 **GPU Pods**（不要 Serverless）

### 5.2 筛选

- **GPU type**：`H100 80GB PCIe`（或 HBM3 更快，价差几毛）。**不要选 40GB、MI300X** 这些——我们用 CUDA + vLLM，H100 是最合适的
- **Cloud type**：**Community Cloud**
- **Region/Data Center**：跟 Network Volume **同一个机房**

列出可选机器：
```
US-CA-2   1× H100 80GB PCIe  $2.49/hr  On-Demand  │ Deploy
US-CA-2   1× H100 80GB PCIe  $1.59/hr  Spot       │ Deploy
```

**第一次先选 On-Demand**，跑通全流程再考虑 Spot。

### 5.3 Configure（部署参数——逐字段抄这个）

点 Deploy，弹出配置表：

| 字段 | 填什么 | 说明 |
|---|---|---|
| **Template** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`（或任意带 CUDA 12.4+ 的官方 PyTorch 模板） | 省得自己装 CUDA |
| **GPU Count** | 1 | 32B 模型单卡够 |
| **Container Disk** | **40 GB**（默认 20 可能不够） | 系统 + pip cache |
| **Volume Disk (Network Volume)** | 选上面创建的 `labeling-vol`，Mount Path `/workspace` | **关键！** 不挂就是烧钱 |
| **Ports** | `22/tcp` 勾上（SSH）。HTTP ports 全不开 | 只要 SSH 够 |
| **Environment Variables** | 空 | |
| **Start Script / Docker Command** | 留默认（空） | |

然后点 **Deploy On-Demand**。

### 5.4 等待 Running

Runpod 网页 → **Pods**：

```
Status    Running ✅        Uptime  00:00:42
GPU       1× H100 80GB      Memory  80/80 GB
Cost      $2.49/hr          Accumulated  $0.03
```

看到 `Running ✅` 就开始计费。点 Pod 名字进详情，找 **Connect** 按钮。

---

## 6. 连接 Pod（SSH）

### 6.1 找到 SSH 信息

Pod 详情页 → **Connect** 弹窗 → 选 **SSH** tab → 看到类似：

```
SSH over exposed TCP:
  ssh root@100.64.1.23 -p 22100 -i ~/.ssh/id_ed25519
```

**注意**：Runpod 两种 SSH：
- **SSH over exposed TCP**（直连，推荐）：端口 ~20000+
- **Direct SSH via web terminal**：网页里连，不稳定，别用

抄 `host`、`port` 到 `configs/runpod.yaml`。

### 6.2 本地配置 runpod.yaml

```bash
cd ~/video_pipeline
cp configs/runpod.yaml.example configs/runpod.yaml
vim configs/runpod.yaml
```

改这几行：
```yaml
pod:
  host: 100.64.1.23          # 从 Runpod Connect 弹窗抄
  port: 22100                # 从 Runpod Connect 弹窗抄
  user: root                 # 保持 root
  ssh_key: ~/.ssh/id_ed25519 # 你生成的私钥
```

### 6.3 测一下连接

```bash
ssh -i ~/.ssh/id_ed25519 -p 22100 -o StrictHostKeyChecking=no root@100.64.1.23 'echo hello from pod'
# 应该输出：hello from pod
```

出错？看 §11.1。

---

## 7. 首次环境部署（pod_setup.sh）

这步装 vLLM + PyTorch + 下载 67GB Qwen3-VL 模型。**首次约 25-40 分钟**，之后再开新 Pod 只要 ~3 分钟（模型在 Network Volume 缓存了）。

### 7.1 本地代码先推上去

```bash
# 确保 configs/runpod.yaml 的 host/port 填好
bash scripts/runpod/01_push.sh
```

脚本会 rsync：
- `src/runpod/schemas.py`
- `src/runpod/pod_runner.py`
- `tools/pod_setup.sh`
- `configs/runpod.yaml`
- 以及经过 Stage 4 筛选（`filters.shot_categories` + `quality_ok=True`）的 manifest + clips

**坑**：第一次 push 前确保本地 Stage 4 至少跑了几个 clip 产生 manifest。否则脚本报"没有符合条件的 shot"。

**先用 max-shots 测试**：
```bash
bash scripts/runpod/01_push.sh configs/runpod.yaml --max-shots 5
```

### 7.2 跑 pod_setup + runner

```bash
bash scripts/runpod/02_run.sh
```

自动执行：
1. SSH 进 Pod
2. 首次自动跑 `bash tools/pod_setup.sh`（通过 `.pod_setup_done` 标志只跑一次）
3. `pod_setup.sh` 做的事：
   - `apt install ffmpeg git rsync`
   - 建 Python venv `/opt/labeling-env`
   - `pip install vllm>=0.6.3 torch>=2.4 decord pillow pydantic>=2 pyyaml rich`
   - `huggingface-cli download Qwen/Qwen3-VL-32B-Instruct` 下载 67GB 到 `/workspace/models/qwen3-vl-32b/`（Network Volume 上）
4. setup 完成后自动 nohup 启动 `pod_runner.py`
5. 本地 terminal 实时 tail `pod_runner.log`

首次输出：
```
══ pod_setup.sh @ <hostname> 2026-04-17T... ══
apt-get update...
python3 -m venv /opt/labeling-env
Installing collected packages: vllm, torch...
Fetching 20 files: 100%|████| 20/20 [18:23<00:00]
══ pod_setup 完成 ══
启动 pod_runner（nohup）...
tail -f output/pod_runner.log ...
2026-04-17 12:34:56  INFO  加载模型 Qwen/Qwen3-VL-32B-Instruct (dtype=bfloat16) ...
2026-04-17 12:36:10  INFO  模型加载完成 (74.2s)
2026-04-17 12:36:11  INFO  本次要跑: 5
2026-04-17 12:36:14  INFO  进度 ok=1 bad=0  最新 MovieA/shot_001 (2134ms)
```

**Ctrl+C 只退出本地 tail，Pod 里的 pod_runner 继续 nohup 跑**——随时可以再 ssh 上去 `tail -f output/pod_runner.log`。

### 7.3 后续（非首次）

- `.pod_setup_done` 已在 → `02_run.sh` 直接跳过 setup
- 模型在 `/workspace/models/qwen3-vl-32b/`（Network Volume） → vLLM 直接加载，**启动 ~60 秒**

---

## 8. 日常工作流（push → run → pull）

一次完整的云端标注：

### 8.1 开 Pod（~2 分钟）
按 §5 Deploy 新 Pod，挂 `labeling-vol`，拿 host+port，填进 `configs/runpod.yaml`。

### 8.2 推数据 + 代码
```bash
bash scripts/runpod/01_push.sh
```
默认 push 所有符合 `filters.shot_categories=[single, dominant]` 且 `quality_ok=True` 的镜头。

**首次推 250k clips × 平均 500KB ≈ 125GB**——上传慢的话看 §11.5。

### 8.3 触发 Pod 跑推理
```bash
bash scripts/runpod/02_run.sh
```
Ctrl+C 退出本地 tail，Pod 继续跑。

### 8.4 看进度（可选）
```bash
ssh -i ~/.ssh/id_ed25519 -p <port> root@<host> \
  'tail -100 /workspace/labeling/output/pod_runner.log'
```
Pod 内：
```bash
wc -l /workspace/labeling/output/.checkpoint.jsonl   # 当前完成数
ls /workspace/labeling/output/*/*.json | wc -l       # 已生成 JSON 数
```

### 8.5 拉结果
```bash
bash scripts/runpod/03_pull.sh
```
自动 rsync `/workspace/labeling/output/` → 本地 `paths.local_labels_root`，用 `schemas.py` 校验每份 JSON。

### 8.6 一键跑全部
```bash
bash scripts/runpod/run_all.sh   # = 01_push → 02_run → 03_pull
```

### 8.7 跑完销毁 Pod
Runpod 网页 → Pods → 你的 Pod → **Terminate**（不是 Stop！）。
模型在 Network Volume 还在，下次重开 Pod 挂同一个 volume 立刻可用。

---

## 9. 关机 vs 终止 Pod（**别搞错！**）

| 操作 | GPU 计费 | 容器盘计费 | Network Volume | 重新启动 |
|---|---|---|---|---|
| **Stop** | ❌ 停收 | ✅ 继续收（~$0.1/hr） | ✅ 一直收 | ~30s |
| **Terminate** | ❌ 停收 | ❌ 销毁释放 | ✅ 一直收 | ~2 min |

**除非你**预计 < 30 分钟回来继续跑，**否则永远 Terminate**。我们批处理场景完全没影响（模型 + 代码 + 输出都在 Network Volume）。

**忘关坑**：吃饭/睡觉前**必须显式 Terminate**。Runpod 不会因没活动就帮你停。半夜忘关 = **$60**。手机装 Runpod app，睡前扫一眼状态。

---

## 10. 费用估算 & 省钱清单

### 10.1 全流程费用估算

假设 250,000 镜头，H100 Community Cloud On-Demand $2.49/hr：

| 项 | 耗时 | 费用 |
|---|---|---|
| 首次 pod_setup + 模型下载 | 30 min | $1.25 |
| 每个镜头 VLM 推理 ~2s | 250k × 2s = 139 hr | **$346** |
| 拉取结果（GPU 空转几十秒）| ≤ 5 min | $0.21 |
| Network Volume（1 个月） | - | $14 |
| **合计（一次性跑完）** | ~140 hr | **~$361** |

分批跑（每次 5 部电影）：
- 每次 setup（Network Volume 读，不重下）+ 模型加载 ≈ 3 min
- 分 10 批额外开销 ≈ 10 × 3 min × $2.49/hr ≈ **$1.25**（可忽略）

### 10.2 省钱清单

1. ✅ **Network Volume 存模型**：省 $1-2/次 × N 次 = 几十美金
2. ✅ **Community Cloud**：省 15-20%
3. ✅ **只在推理时开 Pod**：不要开着去干别的
4. ✅ **小批试跑再全量**：先 `max_shots: 20` 验证
5. ✅ **准备好再开 Pod**：clips + manifest 先备齐
6. ⚙️ **考虑 Spot**：跑通 On-Demand 后切 Spot，便宜 40-50%
7. ⚙️ **Batch 提示（未来优化）**：pod_runner 目前一次一个 shot，batch=4-8 能再省 30%
8. ✅ **关机前拉结果**：每次 push 回本地，不要堆 Pod 上

### 10.3 Spot Pod 用法

Spot = **随时可能被收回**的低价 Pod。便宜 40-50%，被收回时当前 shot 中断，下次从 checkpoint 续跑（代码本来就支持）。

切换：Deploy 时把 **On-Demand** 改 **Spot**。其他步骤不变。

**注意**：Spot 被收回时 Runpod 会通知（邮件/网页），但**不自动重建**。你需要再 Deploy 一次。适合"有耐心分 2-3 天跑完"的场景。

---

## 11. 常见坑 & 故障排查

### 11.1 SSH 连不上

```
ssh: connect to host 100.64.1.23 port 22100: Connection refused
```

**可能原因**：
- Pod 还在 Provisioning → 等 1-2 分钟再试
- 抄错 host/port（**每次开新 Pod 都会变**）
- 企业防火墙挡 20000+ 高端口

**Permission denied (publickey)**：
- `~/.ssh/id_ed25519.pub` 的内容必须完全匹配 Runpod Settings 里的 SSH Public Key
- 用 `ssh -vv -i ~/.ssh/id_ed25519 -p 22100 root@<host>` 看详细握手

### 11.2 `pod_setup.sh` 失败

**`apt-get: E: Unable to locate package ...`**
- 某些老 Pod 镜像 apt 源没配全。重开一个带新版 Ubuntu 22.04 PyTorch 模板的 Pod

**`huggingface-cli download` 卡住 / 失败**
- HF 流控。等 10 分钟再试（`pod_setup.sh` 幂等，重跑没事）
- 或 ssh 进 Pod 手动：
```bash
source /opt/labeling-env/bin/activate
export HF_HUB_DOWNLOAD_TIMEOUT=300
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct \
  --local-dir /workspace/models/qwen3-vl-32b \
  --local-dir-use-symlinks False \
  --resume-download
```

**磁盘满**
- Container Disk 默认 20GB，模型应该去 Network Volume（`/workspace/models/`）。若路径写到 `/root/` 会满。检查 `configs/runpod.yaml` 的 `paths.pod_workspace: /workspace/labeling` 是否正确

### 11.3 vLLM OOM (CUDA Out of Memory)

Qwen3-VL-32B BF16 需要 ~66.7GB。H100 80GB 够，但默认配置偏激进可能 OOM。

```
CUDA out of memory. Tried to allocate 2.00 GiB
```

**改 configs/runpod.yaml**：
```yaml
model:
  precision: fp8   # bf16 → fp8，省一半显存 (~35GB)
```
然后重推代码 + 重启 pod_runner：
```bash
bash scripts/runpod/01_push.sh
ssh -i ~/.ssh/id_ed25519 -p <port> root@<host> \
  'pkill -f pod_runner; cd /workspace/labeling && nohup python src/runpod/pod_runner.py --config runpod.yaml >> output/pod_runner.stdout 2>&1 &'
```

### 11.4 Pod 自动关机 / 被收回

**On-Demand Pod 被关**：罕见，Community Cloud 宿主机维护。重新 Deploy，挂同一 Network Volume，`02_run.sh` 自动续跑（`.checkpoint.jsonl` 在 volume 上）。

**Spot Pod 被收回**：正常。收到通知时 Pod 还能跑 2 分钟。然后：
1. Runpod 网页 Deploy 新 Spot
2. 挂同一 Network Volume
3. 刷新 `configs/runpod.yaml` 的 host/port
4. `bash scripts/runpod/02_run.sh` —— 看到 `.checkpoint.jsonl` 已完成的 shot_id 自动跳过

### 11.5 上传 125GB clips 太慢（可选：Runpod S3 中转）

家里上传 10MB/s × 125GB = ~28 小时。Pod 必须开着接收 = **~$70** 浪费。

**Runpod S3-compatible storage**（Pod 从 S3 走内网拉，非常快）：

1. Runpod 网页 → Settings → API Keys → 生成 S3 key
2. 本地装 `aws-cli` 或 `rclone`
3. Endpoint：`https://s3api-<region>.runpod.io`
4. 上传：
```bash
aws s3 cp /mnt/movies/Films/output/clips/ \
  s3://<bucket>/clips/ --recursive \
  --endpoint-url https://s3api-<region>.runpod.io
```
5. Pod 内从 S3 拉（走内网，~1GB/s）

**权衡**：多 S3 存储费 ~$0.02/GB/月 × 125GB = $2.5/月，省下 28 小时 Pod 费 $70。适合 **上传量巨大**。小规模（<50GB）直接本地 rsync 就行。

### 11.6 Pod 磁盘用完

容器盘 40GB 默认。日志狂飙或 pip 装太多可能满。

Pod 内：
```bash
df -h
# 满的话：
du -sh /tmp/* /var/log/* ~/.cache/* 2>/dev/null | sort -h | tail
rm -rf ~/.cache/pip
```

**预防**：Container Disk 选 **40 GB** 而不是默认 20。多 20GB = $0.0015/hr = 每天 $0.036，可忽略。

### 11.7 查当前烧钱多少

Runpod 网页 → Billing → Accumulated Usage。实时更新。
Pods 详情页看单个 Pod 累计用了多少小时。

### 11.8 Pod 内 pod_runner 挂了怎么办

ssh 进 Pod 看 `output/pod_runner.stdout` 和 `output/pod_runner.log` 最后几行：

```bash
ssh -i ~/.ssh/id_ed25519 -p <port> root@<host>
cd /workspace/labeling
tail -50 output/pod_runner.stdout
tail -50 output/pod_runner.log
```

常见错误：
- `torch.cuda.OutOfMemoryError` → §11.3 换 fp8
- `FileNotFoundError: clips/xxx.mp4` → manifest 里的文件上传时没推上来，重跑 `01_push.sh`
- `pydantic.ValidationError` → VLM 生成的 JSON 不合规，检查 prompt 或 guided_decoding 是否正常

手动重启：
```bash
pkill -f pod_runner
cd /workspace/labeling
nohup python src/runpod/pod_runner.py --config runpod.yaml \
  >> output/pod_runner.stdout 2>&1 &
tail -f output/pod_runner.log
```

---

## 附录 A：quick-start 速查表

全流程命令清单（已完成 §1-§5）：

```bash
# 1. 开 Pod（网页，挂 Network Volume，拿 host+port）

# 2. 本地
vim configs/runpod.yaml   # 填 host/port

# 3. 推数据 + 代码
bash scripts/runpod/01_push.sh

# 4. 触发 pod_setup + pod_runner（首次 30 min，之后 1 min）
bash scripts/runpod/02_run.sh

# 5. 定期 tail 看进度
ssh -i ~/.ssh/id_ed25519 -p <port> root@<host> \
  'tail -50 /workspace/labeling/output/pod_runner.log'

# 6. 跑完拉结果
bash scripts/runpod/03_pull.sh

# 7. 检查本地
ls /mnt/movies/Films/output/labels/<movie>/ | head

# 8. Runpod 网页 → Terminate Pod（别忘！）
```

## 附录 B：配置文件字段速查

`configs/runpod.yaml`：

| 字段 | 必填？ | 说明 |
|---|---|---|
| `pod.host` | ✅ | 每次 Deploy 新 Pod 都要改 |
| `pod.port` | ✅ | 每次 Deploy 新 Pod 都要改 |
| `pod.user` | ✅ | 固定 `root` |
| `pod.ssh_key` | ✅ | 你的私钥路径 |
| `paths.pod_workspace` | ✅ | `/workspace/labeling`（跟 Network Volume mount point 对上） |
| `paths.local_clips_root` | ✅ | Stage 2 输出的 clips 目录 |
| `paths.local_labels_root` | ✅ | Stage 5 JSON 结果落地位置 |
| `paths.local_manifest_dir` | ✅ | Stage 4 manifest 目录 |
| `model.name` | ✅ | `Qwen/Qwen3-VL-32B-Instruct` |
| `model.precision` | ✅ | `bf16`（推荐）或 `fp8`（省显存） |
| `filters.shot_categories` | ✅ | `[single, dominant]` 只标人物特写；`[]` = 全部 |
| `filters.movies` | ✅ | `[]` = 全部；或 `[MovieA, MovieB]` |
| `filters.max_shots` | ✅ | `null` = 全量；给数字可限量调试 |
| `sampling.*` | ✅ | 默认就行 |

`configs/runpod.yaml` 在 `.gitignore` 里，包含真实 Pod 凭据，**不要 commit**。

## 附录 C：费用快速参考（2026 年 4 月价）

| 项 | 价格 |
|---|---|
| H100 80GB On-Demand（Community） | $2.49/hr |
| H100 80GB Spot（Community） | $1.59/hr |
| H100 80GB On-Demand（Secure） | $2.99/hr |
| Network Volume | $0.07/GB/月 |
| Container Disk | 含在 Pod 里 |
| S3-compat 存储（可选） | $0.02/GB/月 |
| 流量（入/出） | 免费 |

实际价格可能随时变动，以 Runpod Deploy 页显示为准。
