# 视频批量预处理 Pipeline

三台 GPU 服务器协作，将任意格式任意分辨率的电影文件加工成统一规格 → 按镜头切分 → 预分类（人物 / 风景 / 人数）→（可选）字幕擦除 → 上云 VLM 标注，为视频生成模型（Wan2.2 / LTX-2.3）的 LoRA + ControlNet 训练准备素材。

**流水线**：
```
Stage 1 转码 → Stage 2 切分 → Stage 4 分类 → Stage 5 Runpod 标注（手动）
                                    ↑
                          Stage 3 字幕去除（可选，默认不跑）
```

`--stage all` 默认跑 Stage 1 + 2 + 4；Stage 3 / Stage 5 按需触发。

> 📖 **每个阶段的完整说明 + 操作手册**：[docs/stages/](stages/README.md)
> - [Stage 1 转码](stages/01_transcode.md) · [Stage 2 镜头切分](stages/02_scene_split.md)
> - [Stage 3 字幕去除（可选）](stages/03_subtitle_remove.md)
> - [Stage 4 镜头分类](stages/04_shot_classify.md)（**含 manifest 字段参考表**）
> - [Stage 5 云端标注](stages/05_labeling.md) + [Runpod 使用手册](RUNPOD_MANUAL.md)

---

## 目录结构

```
video_pipeline/
├── src/
│   ├── common/              共享模块
│   │   └── task_queue.py    文件锁队列
│   ├── workers/             本地 GPU worker
│   │   ├── process_videos.py    Stage 1 转码
│   │   ├── scene_split.py       Stage 2 镜头切分
│   │   ├── subtitle_remove.py   Stage 3 字幕去除（可选模块，默认不跑）
│   │   └── shot_classify.py     Stage 4 镜头分类（YOLOv8-person + 规则）
│   ├── dispatcher/
│   │   └── distributed_dispatch.py  分布式调度器
│   └── runpod/              Stage 5 Runpod 云端标注（独立工作流）
│       ├── schemas.py       Pydantic v2 schema（对齐 docs/labelingStandards/）
│       ├── pod_runner.py    Pod 内 vLLM + Qwen3-VL-32B 推理（guided_json）
│       ├── upload.py        本地 → Pod：rsync 筛后 clips + manifest
│       └── download.py      Pod → 本地：rsync + schema 校验
├── configs/
│   ├── servers.yaml{,.example}    三台本地 GPU 机器配置
│   └── runpod.yaml{,.example}     Runpod Pod SSH + 推理配置
├── docs/
│   ├── README.md                  本文件
│   └── labelingStandards/         标注严格规范（schema + examples + taxonomy）
├── scripts/
│   ├── startAllServerFirst.sh / startAllServer.sh / startServer.sh / stopAllServer.sh / stopServer.sh / checkStatus.sh
│   └── runpod/
│       ├── 01_push.sh       推送 clips+manifest 到 Pod
│       ├── 02_run.sh        SSH 进 Pod 跑推理（首次自动装环境）
│       ├── 03_pull.sh       拉回 JSON + 校验
│       └── run_all.sh       一键 push→run→pull
├── tools/
│   ├── setup_vsr_env.sh     Stage 3 VSR 环境（仅需去字幕时装）
│   ├── patch_vsr.sh         VSR 补丁
│   ├── pod_setup.sh         Pod 内一次性装 vLLM + 下载 Qwen3-VL-32B
│   └── deploy_ssh.sh        SSH 互信部署
├── process_videos.py        ← 根目录薄封装
├── scene_split.py           ← 根目录薄封装
├── subtitle_remove.py       ← 根目录薄封装（Stage 3 可选）
├── shot_classify.py         ← 根目录薄封装
├── distributed_dispatch.py  ← 根目录薄封装
├── CLAUDE.md
├── requirements.txt
└── .gitignore
```

所有命令都通过根目录的薄封装脚本运行，`configs/servers.yaml` 是 `--servers` 的默认值，无需显式指定。

---

## 输出规格

| 阶段 | 输出 |
|---|---|
| Stage 1 转码 | 1920×auto / 24fps / H.264(NVENC 优先，libx264 回退) / AAC 128k / MP4 faststart。动态码率（4K→8M、1080p→3.5M、720p→2M），单文件≤2GB。命名 `前6字_uuid8.mp4` |
| Stage 2 切分 | `clips_root/{movie_stem}/shot_NNN.mp4`，首尾各去掉 10 个镜头 |
| Stage 3 去字幕 | `clean_root/{movie_stem}/shot_NNN.mp4`，底部 25% 区域硬字幕擦除（VSR） |

---

## 环境准备

### 各机器都需要

```bash
sudo apt install -y ffmpeg openssh-server cifs-utils
ffmpeg -encoders | grep h264_nvenc        # 确认 NVENC 可用
conda activate movietest
pip install -r requirements.txt           # rich, pyyaml, scenedetect[opencv]
```

### SSH 免密（在主控机执行）

```bash
ssh-keygen -t ed25519
ssh-copy-id -p 2222 <用户名>@<A6000_IP>
ssh-copy-id -p 2222 <用户名>@<A8000_IP>
```

### 挂载共享盘（各机器挂到同一路径，队列文件才能共享）

```bash
sudo mkdir -p /mnt/movies
echo 'username=你的用户名' | sudo tee /etc/smbcredentials
echo 'password=你的密码' | sudo tee -a /etc/smbcredentials
sudo chmod 600 /etc/smbcredentials
echo '//<NAS_IP>/movies /mnt/movies cifs credentials=/etc/smbcredentials,uid=1000,gid=1000,vers=3.0,nofail,_netdev 0 0' \
  | sudo tee -a /etc/fstab
sudo mount -a
```

### `configs/servers.yaml` 配置

复制模板填入真实 IP / 用户名：
```bash
cp configs/servers.yaml.example configs/servers.yaml
vim configs/servers.yaml
```
`remote_script: ~/video_pipeline/process_videos.py` 指向**远端根目录薄封装**，保持这个路径不用改。

---

## 每个功能的启动命令

所有命令都从项目根目录执行。`configs/servers.yaml` 是 `--servers` 的默认值，无需显式传。

### 1. 首次启动（git pull + 部署脚本）

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/Baiduyun \
  --output-dir /mnt/movies/Films/output \
  --git-pull --deploy
```
或：
```bash
bash scripts/startAllServerFirst.sh
```

### 2. 日常启动（默认 Stage 1 → 2 → 4）

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/Baiduyun \
  --output-dir /mnt/movies/Films/output \
  --git-pull
```
`--stage all` 默认跑 `1,2,4`。Stage 3（去字幕）不在默认链路里，按需单独触发。

### 3. 只跑指定 Stage

```bash
# 只跑 Stage 2 镜头切分
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/forCloudKor \
  --output-dir /mnt/movies/Films/forCloudKorOutput \
  --stage 2

# 只跑 Stage 4 镜头分类（YOLOv8-person + 规则，写 manifest）
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/forCloudKor \
  --output-dir /mnt/movies/Films/forCloudKorOutput \
  --stage 4
# 产出：<output>/manifest/<movie>.jsonl，每行一个 shot 的分类结果

# 只跑 Stage 3 字幕去除（仅当视频有硬字幕时才跑；首次用前先 bash tools/setup_vsr_env.sh）
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/forCloudKor \
  --output-dir /mnt/movies/Films/forCloudKorOutput \
  --stage 3
```
`scripts/startAllServer.sh` 是 Stage 2 的快捷入口。

### 4. 只在指定机器启动

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/Baiduyun \
  --output-dir /mnt/movies/Films/output \
  --target RTX4090_local
# 多台机器：--target A6000,A8000
```
`scripts/startServer.sh` 是 `--target RTX4090_local` 的快捷入口。

### 5. 查状态

```bash
python distributed_dispatch.py \
  --output-dir /mnt/movies/Films/output --status
```
或 `bash scripts/checkStatus.sh`。输出：各机器运行状态 + 队列进度（done / pending / claimed / error）。

### 6. 停止

```bash
# 停所有机器
python distributed_dispatch.py --stop
# 停指定机器
python distributed_dispatch.py --stop --target RTX4090_local
```
或 `bash scripts/stopAllServer.sh` / `bash scripts/stopServer.sh`。

### 7. 连通性检测

```bash
python distributed_dispatch.py --check
```

### 8. 预览任务（不执行）

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/Baiduyun \
  --output-dir /mnt/movies/Films/output \
  --dry-run
```

### 9. 强制重新处理全部文件

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films/Baiduyun \
  --output-dir /mnt/movies/Films/output \
  --force
```

### 10. 单机本地模式（不用调度器）

```bash
python process_videos.py ./input ./output
python process_videos.py ./input ./output --workers 1 --no-nvenc   # 强制 CPU
```

### 11. 手动加入队列（单台机器补位）

```bash
conda activate movietest
cd ~/video_pipeline
python process_videos.py /mnt/movies/Films /mnt/movies/Films/output \
  --queue-dir /mnt/movies/Films/output/.queue \
  --worker-id 我的机器名
```

---

## Stage 5：Runpod 云端标注（手动工作流，不走 dispatcher）

> 📖 **详细手册**：[docs/RUNPOD_MANUAL.md](./RUNPOD_MANUAL.md) — 第一次用 Runpod 必读，包含注册、SSH、Network Volume、Pod 选型、费用优化、常见坑和排错。下面只是快速摘要。

用 Runpod 租 H100 Pod 跑 **Qwen3-VL-32B-Instruct + vLLM guided_json**，严格输出 `docs/labelingStandards/json_schema_integrated.md` 规定的 schema。

**前置条件**：Stage 4 已产出 `<output_dir>/manifest/<movie>.jsonl`（每个镜头一行 shot_id / shot_category / num_people 等）。

**一次性准备**：
1. 在 Runpod 控制台开一个带 SSH 的 H100 Pod（≥ 80GB 显存、≥ 200GB 磁盘），把你的 SSH 公钥加到 Runpod 账号。
2. 复制配置模板并填入 Pod 的 IP / 端口 / SSH key：
   ```bash
   cp configs/runpod.yaml.example configs/runpod.yaml
   vim configs/runpod.yaml
   ```
3. （可选）在 `configs/runpod.yaml` 的 `filters.shot_categories` 里限定要标注的镜头类型（默认 `[single, dominant]`）。

**一键运行**（push 本地数据 → Pod 跑推理 → pull 结果）：
```bash
bash scripts/runpod/run_all.sh
```

**分步运行**（调试或分阶段执行）：
```bash
bash scripts/runpod/01_push.sh    # 只推数据到 Pod（含 src/runpod/ 代码 + pod_setup.sh）
bash scripts/runpod/02_run.sh     # SSH 进 Pod：首次装 vLLM+下载模型；nohup 跑 pod_runner；tail 日志
bash scripts/runpod/03_pull.sh    # 把 Pod 上 output/*.json 拉回本地，用 schemas.py 校验
```

**dry-run 校验**（不连 Pod）：
```bash
# 校验 manifest 与 schemas.py 对齐
python -m src.runpod.upload --config configs/runpod.yaml --dry-run

# 用 docs/labelingStandards/examples/*.json 自测 schema
python -m src.runpod.schemas
```

**断点续跑**：Pod 内 `pod_runner.py` 每写完一个 shot 就追加到 `output/.checkpoint.jsonl`，下次重启自动跳过已完成的 shot_id。单个 shot 校验失败时原始模型输出保存在 `output/_failed/<shot_id>.raw.txt` 便于排查。

**停机**：在 Runpod 网页手动 stop Pod（SSH 脚本不负责计费管理）。

---

## 中断与恢复

- **Ctrl+C**：主控退出，远端 worker 继续后台运行。重新执行原命令即可附加回日志监控。
- **广播 kill 后恢复**：`--stop` → 直接重跑原命令。队列会把上次 claimed（被中断）的任务重置为 pending，done 的不重跑。
- **网络/断电**：超过 10 分钟心跳超时的 claimed 任务自动释放为 pending，其他 worker 自动接手。
- **加/减机器**：修改 `configs/servers.yaml`（注释 / 增行），重跑原命令即可。

---

## 代码更新

```bash
git add -A && git commit -m "..." && git push
# 下次启动加 --git-pull 自动同步所有机器
```

---

## kill 机制

`src/workers/process_videos.py` 等在模块顶部调用 `os.setsid()` 让自身成为进程组 leader（PGID == PID）。通过根目录 shim `import` 时即触发，所有 ffmpeg 子进程继承同一进程组。

dispatcher 用 `kill -TERM/-KILL -- -$PGID`，**一条命令覆盖 Python 进程 + 所有 ffmpeg 子进程**，CPU 和 GPU 立即释放。

应急手动 kill：
```bash
kill -KILL -- -$(cat ~/pipeline_<机器名>.pid)
# 或暴力
pkill -KILL -f process_videos.py && pkill -KILL -f ffmpeg
```

---

## 常见问题

**Q：某机器显示"队列文件不可访问"被跳过？**  
A：该机器的 `--output-dir` 路径不是共享存储或共享盘未挂载。在该机器 `ls /mnt/movies/Films/output/.queue/` 如果不可见，参考"挂载共享盘"章节。

**Q：kill 后 CPU 还在跑？**  
A：确认 v5.2+。旧版 ffmpeg 曾用 `start_new_session=True` 成了孤儿，新版 ffmpeg 在同一进程组，`kill -- -$PGID` 一次搞定。

**Q：WSL2 看不到 Windows 网络映射盘？**  
A：WSL2 不会自动挂载 Windows 映射盘，要用 `sudo mount -t cifs` 直接挂载 SMB。

**Q：SSH 连接超时？**  
A：WSL2 IP 每次重启都变。在 Windows 11 的 `.wslconfig` 里加 `[wsl2]` / `networkingMode=mirrored` 即可固定 IP。

**Q：GPU 利用率低？**  
A：先看 NAS/共享盘读速是否瓶颈（多机同读同源）。用 `nvidia-smi` 看 NVENC 是否工作，对比 `--no-nvenc` 跑 CPU 编码。
