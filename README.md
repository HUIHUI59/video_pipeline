# 视频批量预处理 Pipeline

三台 GPU 服务器（RTX 4090 / A6000 / A8000）协作，将任意格式、任意分辨率的电影文件批量转码为统一规格，为后续分镜标注和视频模型训练做准备。

---

## 文件结构

```
video_pipeline/
├── process_videos.py       # 单机处理主程序（支持队列模式）
├── distributed_dispatch.py # 多机分布式调度器（主控运行）
├── task_queue.py           # 共享任务队列（基于文件锁）
├── servers.yaml            # 服务器配置
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
| 命名 | `原名前6字_uuid8.mp4`，如 `极品老妈_a3f2c1b4.mp4` |

---

## 环境准备

### 每台机器都需要执行

```bash
# 系统依赖
sudo apt install -y ffmpeg openssh-server

# 验证 NVENC
ffmpeg -encoders | grep h264_nvenc

# Python 依赖（在 movietest conda 环境里）
conda activate movietest
pip install rich pyyaml
```

### SSH 免密登录（在 4090 主控的 WSL2 里）

```bash
ssh-keygen -t ed25519
ssh-copy-id -p 2222 leolee@192.168.50.102   # A6000
ssh-copy-id -p 2222 leolee@192.168.50.237   # A8000
```

### WSL2 挂载网络共享盘（A6000 和 4090）

```bash
sudo apt install -y cifs-utils
sudo mkdir -p /mnt/movies

# 写入凭据
sudo tee /etc/smbcredentials << EOF
username=你的Windows用户名
password=你的Windows密码
EOF
sudo chmod 600 /etc/smbcredentials

# 写入 fstab 实现开机自动挂载
echo '//192.168.50.237/movies /mnt/movies cifs credentials=/etc/smbcredentials,uid=1000,gid=1000,vers=3.0,nofail,_netdev 0 0' | sudo tee -a /etc/fstab
sudo mount -a
```

### Git 初始化（首次）

```bash
# 在 4090 上
cd ~/video_pipeline
git init
git add .
git commit -m "init"
git remote add origin https://github.com/你/video_pipeline.git
git push -u origin main

# 在 A6000、A8000 上各执行一次
git clone https://github.com/你/video_pipeline.git ~/video_pipeline
```

---

## servers.yaml 配置

```yaml
servers:
  - name: A6000
    host: 192.168.50.102
    port: 2222               # WSL2 端口转发后的端口
    user: leolee
    ssh_key: ~/.ssh/id_ed25519
    gpus: [0]
    weight: 1.3
    conda_env: movietest
    remote_script: ~/video_pipeline/process_videos.py
    git_repo: ~/video_pipeline   # git pull 时的仓库目录

  - name: A8000
    host: 192.168.50.237
    port: 2222
    user: leolee
    ssh_key: ~/.ssh/id_ed25519
    gpus: [0]
    weight: 1.5
    conda_env: movietest
    remote_script: ~/video_pipeline/process_videos.py
    git_repo: ~/video_pipeline

  - name: RTX4090_local
    host: localhost           # 主控机自身，不走 SSH
    port: 22
    user: leo4090
    ssh_key: ~/.ssh/id_ed25519
    gpus: [0]
    weight: 1.0
    conda_env: movietest
    remote_script: ~/video_pipeline/process_videos.py
    git_repo: ~/video_pipeline
```

**白天只用 A6000 + A8000**：注释掉 `RTX4090_local` 整个块即可，无需其他改动。

---

## 快速开始

### 首次运行（部署代码 + 启动）

```bash
# 在 4090 主控的 WSL2 里
cd ~/video_pipeline
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml \
  --git-pull \   # 先同步所有机器到最新代码
  --deploy       # 部署脚本到各机器（首次必须加，后续用 --git-pull 替代）
```

### 日常运行

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml \
  --git-pull
```

### 预览任务分配（不实际执行）

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml \
  --dry-run
```

### 检测连通性

```bash
python distributed_dispatch.py --servers servers.yaml --check
```

---

## 中断与恢复

### 情况一：主控退出，远端继续跑

按 `Ctrl+C` 后选 **1**，主控退出，A6000 / A8000 在后台继续处理。

**查看远端进度**：

```bash
# 在 4090 上 SSH 进去看日志
ssh -p 2222 leolee@192.168.50.102 "tail -f ~/pipeline_A6000.log.stdout"

# 或者直接查看队列状态（在任意机器上）
python3 -c "
import sys; sys.path.insert(0,'~/video_pipeline')
from task_queue import TaskQueue
q = TaskQueue('/mnt/movies/Films/output/.queue')
print(q.stats())
"
```

**重新连接主控监控**（不重启任务）：

```bash
# 主控重新运行，--dry-run 不会重置队列，只会显示当前状态
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml
# 此时队列已存在，init_queue 只重置 claimed→pending，done 保留
# 各机器会自动继续抢剩余任务
```

### 情况二：广播 kill（Ctrl+C 选 2，或 --stop）

```bash
# 随时终止所有机器
python distributed_dispatch.py --servers servers.yaml --stop
```

**恢复**：直接重新运行日常运行命令，`init_queue` 会自动把上次 `claimed`（被中断）的任务重置为 `pending`，`done` 的不重跑。

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml
```

### 情况三：白天关掉 4090，晚上重新加入

白天：在 `servers.yaml` 里注释掉 `RTX4090_local`，只用 A6000 + A8000。

晚上加回 4090：取消注释 `RTX4090_local`，重新运行日常命令。队列里剩余的任务会自动分配给 4090。

### 情况四：某台机器意外断电/网络中断

该机器正在处理的任务（`claimed` 状态）会在 **10 分钟**后因心跳超时自动释放为 `pending`，其他机器的 worker 会自动接手，无需任何手动操作。

### 情况五：单台机器手动加入队列处理

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

---

## 强制重新处理所有文件

```bash
python distributed_dispatch.py \
  --input-dir /mnt/movies/Films \
  --output-dir /mnt/movies/Films/output \
  --servers servers.yaml \
  --force   # 清空队列，包括已完成的文件
```

---

## 单机模式（不用调度器）

```bash
# 本机处理当前目录下所有视频
python process_videos.py ./input ./output

# 限制并发数（白天不影响正常使用）
python process_videos.py ./input ./output --workers 1

# 强制 CPU 编码
python process_videos.py ./input ./output --no-nvenc
```

---

## 代码更新流程

```bash
# 1. 在 4090 上修改代码
vim process_videos.py

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
kill -KILL -- -$(cat ~/pipeline_A6000.pid)   # 用 PGID kill 整组
# 或者暴力方式
pkill -KILL -f process_videos.py && pkill -KILL -f ffmpeg
```

---

## 常见问题

**Q: A8000 启动时显示 `pending=0` 直接退出？**  
A: 队列被其他机器抢完了。重新运行调度器命令，`init_queue` 会把 `claimed` 重置为 `pending`，A8000 就能抢到任务。

**Q: kill 后 CPU 还在跑？**  
A: 确认是否用了新版代码（v5.2+）。旧版 ffmpeg 用 `start_new_session=True` 成了孤儿进程。新版 ffmpeg 在同一进程组，PGID kill 一次搞定。

**Q: WSL2 看不到网络共享盘 Z:？**  
A: WSL2 不自动挂载 Windows 网络映射盘。用 `sudo mount -t cifs` 直接挂载 SMB，详见环境准备部分。

**Q: SSH 连接超时？**  
A: WSL2 IP 每次重启都会变。推荐在 Windows 11 的 `.wslconfig` 里设置 `networkingMode=mirrored`，让 WSL2 直接使用 Windows 的局域网 IP。