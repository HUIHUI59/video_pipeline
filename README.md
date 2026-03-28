# 视频批量预处理 Pipeline

三台 GPU 服务器协作，将任意格式、任意分辨率的电影文件批量转码为统一规格，为后续分镜标注和视频模型训练做准备。

---

## 文件结构

```
video_pipeline/
├── process_videos.py       # 单机处理主程序（支持队列模式）
├── distributed_dispatch.py # 多机分布式调度器（主控运行）
├── task_queue.py           # 共享任务队列（基于文件锁）
├── servers.yaml            # 服务器配置（本地维护，不提交真实 IP）
├── requirements.txt        # Python 依赖
└── README.md
```

---

## 输出规格

| 参数 | 值 |
|------|------|
| 分辨率 | 宽度 1920px，高度按原始比例自适应（21:9 不会出现黑边）|
| 帧率 | 24fps |
| 视频编码 | H.264（NVENC GPU 优先，自动回退 libx264）|
| 音频 | AAC 128kbps 双声道 |
| 容器 | MP4（faststart）|
| 码率 | 动态：4K→8Mbps，1080p→3.5Mbps，720p→2Mbps，单文件≤2GB |
| 命名 | `原名前6字_uuid8.mp4` |

---

## 环境准备

### 每台机器都需要执行

```bash
# 系统依赖
sudo apt install -y ffmpeg openssh-server

# 验证 NVENC
ffmpeg -encoders | grep h264_nvenc

# Python 依赖（在 conda 环境里）
conda activate movietest
pip install rich pyyaml
```

### SSH 免密登录（在主控机的 WSL2 里执行）

```bash
ssh-keygen -t ed25519
ssh-copy-id -p 2222 <用户名>@<A6000_IP>
ssh-copy-id -p 2222 <用户名>@<A8000_IP>
```

### WSL2 挂载网络共享盘

每台需要访问共享素材的机器都要挂载到**同一路径**（例如 `/mnt/movies`），队列文件才能被所有机器读取。

```bash
sudo apt install -y cifs-utils
sudo mkdir -p /mnt/movies

# 写入凭据
sudo tee /etc/smbcredentials << EOF
username=你的Windows用户名
password=你的Windows密码
EOF
sudo chmod 600 /etc/smbcredentials

# 写入 fstab 实现开机自动挂载（IP 填 NAS/共享主机的 Windows 局域网 IP）
echo '//<NAS_IP>/movies /mnt/movies cifs credentials=/etc/smbcredentials,uid=1000,gid=1000,vers=3.0,nofail,_netdev 0 0' | sudo tee -a /etc/fstab
sudo mount -a
```

> **关键**：所有机器的 `--output-dir` 必须指向同一个物理位置，队列文件（`.queue/.pipeline_queue.json`）才能在多机之间共享。若某台机器无法访问队列文件，启动时会打印警告并跳过该机器。

### Git 初始化（首次）

```bash
# 在主控机上
cd ~/video_pipeline
git init
git add .
git commit -m "init"
git remote add origin https://github.com/<你的账号>/video_pipeline.git
git push -u origin main

# 在其他机器上各执行一次
git clone https://github.com/<你的账号>/video_pipeline.git ~/video_pipeline
```

---

## servers.yaml 配置

复制 `servers.yaml` 模板，按实际环境填写 IP、用户名、路径：

```yaml
servers:
  - name: GPU_SERVER_1         # 机器标识，随意命名
    host: 192.168.x.x          # WSL2 IP 或 Windows 局域网 IP（端口转发）
    port: 2222                 # SSH 端口（WSL2 端口转发通常用 2222）
    user: <用户名>              # WSL2 Linux 用户名
    ssh_key: ~/.ssh/id_ed25519
    gpus: [0]
    weight: 1.3                # 算力权重，越高抢到的任务越多
    conda_env: movietest
    remote_script: ~/video_pipeline/process_videos.py
    git_repo: ~/video_pipeline

  - name: GPU_SERVER_2
    host: 192.168.x.x
    port: 2222
    user: <用户名>
    ssh_key: ~/.ssh/id_ed25519
    gpus: [0]
    weight: 1.5
    conda_env: movietest
    remote_script: ~/video_pipeline/process_videos.py
    git_repo: ~/video_pipeline

  - name: LOCAL_GPU            # 主控机自身，不走 SSH
    host: localhost
    port: 22
    user: <用户名>
    ssh_key: ~/.ssh/id_ed25519
    gpus: [0]
    weight: 1.0
    conda_env: movietest
    remote_script: ~/video_pipeline/process_videos.py
    git_repo: ~/video_pipeline
    # 白天不用主控机时，注释掉这整个块即可
```

**WSL2 IP 稳定化**（推荐）：在 Windows 11 的 `%USERPROFILE%\.wslconfig` 里加：
```ini
[wsl2]
networkingMode=mirrored
```
开启后 WSL2 直接使用 Windows 网卡 IP，IP 不再每次重启变化。

---

## 快速开始

### 首次运行

```bash
cd ~/video_pipeline
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml \
  --git-pull \
  --deploy
```

- `--git-pull`：启动前同步所有机器到最新代码
- `--deploy`：首次必须加，将脚本上传到各远端机器；后续改用 `--git-pull`

### 日常运行

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml \
  --git-pull
```

### 预览任务（不实际执行）

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml \
  --dry-run
```

---

## 运维命令

### 查看状态和进度

```bash
# 查看所有机器是否在运行 + 队列完成进度
python distributed_dispatch.py \
  --servers servers.yaml \
  --output-dir /mnt/movies/Films/output \
  --status
```

输出示例：
```
服务器          状态        PID
GPU_SERVER_1   运行中 ▶    4123
GPU_SERVER_2   已停止 ■    -
LOCAL_GPU      运行中 ▶    4912

队列: total=297  done=85 (28.6%)  pending=209  claimed=3  error=0
```

运行期间调度器主控也会**每 30 秒自动打印一次进度**，无需手动查询。

### 停止服务

```bash
# 停止所有机器
python distributed_dispatch.py --servers servers.yaml --stop

# 只停某一台（不影响其他机器继续处理）
python distributed_dispatch.py --servers servers.yaml --stop --target GPU_SERVER_1

# 停止多台
python distributed_dispatch.py --servers servers.yaml --stop --target GPU_SERVER_1,LOCAL_GPU
```

停止后会确认进程是否真正终止（`✅ 已终止` 或 `FAILED: 仍在运行`），不再静默失败。

### 在指定机器上启动

```bash
# 只在 GPU_SERVER_2 上启动（其他机器已在运行时使用）
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml \
  --target GPU_SERVER_2
```

### 连通性检测

```bash
python distributed_dispatch.py --servers servers.yaml --check
```

---

## 中断与恢复

### Ctrl+C：主控退出，远端继续

按 `Ctrl+C`，主控直接退出并打印三条命令，**远端 worker 继续在后台处理**。

```
⚠  主控已退出，worker 继续运行。
  查看进度  →  --status --output-dir <路径>
  停止全部  →  --stop
  停止单台  →  --stop --target <机器名>
  继续监控  →  重新运行原命令（已运行的 worker 不会重复启动）
```

重新运行原命令后，调度器会检测到 worker 已在运行，直接附加到日志监控，**不会重复启动**。

### 广播 kill 后恢复

```bash
# 停止所有机器
python distributed_dispatch.py --servers servers.yaml --stop

# 直接重新运行，自动续传
# init_queue 会把上次 claimed（被中断）的任务重置为 pending，done 的不重跑
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml
```

启动时会显示 **续传模式**：
```
输入目录视频总数 : 297
续传模式 : done=85 (28.6%)  pending=212  error=0  (已重置 3 个中断任务)
```

### 白天关掉主控机，晚上重新加入

在 `servers.yaml` 里注释掉 `LOCAL_GPU` 块，只用远端机器。晚上取消注释后重新运行日常命令，队列里剩余的任务自动分配给主控机。

### 某台机器意外断电或网络中断

该机器正在处理的任务（`claimed` 状态）会在 **10 分钟**后因心跳超时自动释放为 `pending`，其他机器的 worker 会自动接手，无需任何手动操作。

### 单台机器手动加入队列处理

```bash
# 在任意机器的 WSL2 里直接运行
conda activate movietest
cd ~/video_pipeline
python process_videos.py \
  /mnt/movies/Films \
  /mnt/movies/Films/output \
  --queue-dir /mnt/movies/Films/output/.queue \
  --worker-id 我的机器名
```

### 强制重新处理所有文件

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml \
  --force
```

---

## 单机模式（不用调度器）

```bash
# 本机处理当前目录下所有视频
python process_videos.py ./input ./output

# 限制并发数
python process_videos.py ./input ./output --workers 1

# 强制 CPU 编码
python process_videos.py ./input ./output --no-nvenc
```

---

## 代码更新流程

```bash
# 1. 在主控机上修改代码
# 2. 推送到 git
git add -A && git commit -m "描述改动" && git push

# 3. 下次启动时加 --git-pull 自动同步所有机器
python distributed_dispatch.py ... --git-pull
```

---

## kill 机制说明

`process_videos.py` 启动时立即调用 `os.setsid()`，使自身成为进程组 leader（PGID == PID）。所有 ffmpeg 子进程继承同一进程组。

调度器执行 `kill -TERM/-KILL -- -$PGID`，**一条命令覆盖 Python 进程 + 所有 ffmpeg 子进程**，CPU 和 GPU 会立即释放。

如需手动 kill（应急）：

```bash
# 在对应机器的 WSL2 里
kill -KILL -- -$(cat ~/pipeline_<机器名>.pid)
# 或者暴力方式
pkill -KILL -f process_videos.py && pkill -KILL -f ffmpeg
```

---

## 常见问题

**Q: 某台机器显示"队列文件不可访问"然后被跳过？**
A: 该机器上的 `--output-dir` 路径不是共享存储，或共享盘未挂载。确认在该机器的 WSL2 里执行 `ls /mnt/movies/Films/output/.queue/` 能看到队列文件，若不行则参考"WSL2 挂载网络共享盘"章节。

**Q: kill 后 CPU 还在跑？**
A: 确认是否用了 v5.2+ 版本。旧版 ffmpeg 用 `start_new_session=True` 成了孤儿进程。新版 ffmpeg 在同一进程组，`kill -- -$PGID` 一次搞定。

**Q: WSL2 看不到网络共享盘（Z: 等映射盘）？**
A: WSL2 不自动挂载 Windows 网络映射盘。用 `sudo mount -t cifs` 直接挂载 SMB，详见环境准备部分。

**Q: SSH 连接超时？**
A: WSL2 IP 每次重启都会变。推荐在 Windows 11 的 `.wslconfig` 里设置 `networkingMode=mirrored`，让 WSL2 直接使用 Windows 的局域网 IP（固定不变）。

**Q: 某台机器 GPU 利用率低？**
A: 优先检查 NAS/共享盘读速是否成为瓶颈（多机同时读同一来源）。也可用 `nvidia-smi` 确认 NVENC 是否正常工作，或在该机器上单独运行 `python process_videos.py --no-nvenc` 对比 CPU 编码速度。
