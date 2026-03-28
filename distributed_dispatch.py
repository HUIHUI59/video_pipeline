#!/usr/bin/env python3
"""
distributed_dispatch.py  v5.1
════════════════════════════════════════════════════════════════
修复：
  ① 本地机器启动方式：改用 subprocess.Popen 直接后台启动，
     不走 SSH / bash -lc，彻底解决 nohup & 在非交互 shell 无效问题
  ② kill：改用 PGID（进程组）kill，覆盖 ffmpeg 子进程
     process_videos 用 os.setsid() 让自己成为进程组 leader，
     kill_server 读 PID 后用 kill -TERM/-KILL -- -PGID 杀整组
  ③ 命名：原名前6字 + uuid8（无序号）
  ④ git：--git-pull 在远端执行 git pull 更新代码
════════════════════════════════════════════════════════════════
"""

import os, sys, json, shutil, signal, logging, argparse
import subprocess, threading, time
from pathlib import Path
from dataclasses import dataclass, field

import yaml
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)
log     = logging.getLogger("dispatch")
console = Console()

VIDEO_EXTENSIONS = {
    ".mp4",".mkv",".avi",".mov",".wmv",".flv",
    ".m4v",".ts",".mts",".m2ts",".webm",".rmvb",
    ".rm",".mpeg",".mpg",".vob",".3gp",
}
CONDA_SEARCH = [
    "~/miniconda3/bin/conda","~/anaconda3/bin/conda",
    "/opt/conda/bin/conda","/usr/local/conda/bin/conda",
]


# ══════════════════════════════════════════════════════════════
# Server
# ══════════════════════════════════════════════════════════════

@dataclass
class Server:
    name:str; host:str; port:int; user:str; ssh_key:str
    gpus:list; weight:float; conda_env:str; remote_script:str
    git_repo:str = ""

    @property
    def is_local(self) -> bool:
        return self.host in ("localhost", "127.0.0.1")

    def ssh_opts(self) -> list[str]:
        return [
            "-i", os.path.expanduser(self.ssh_key),
            "-p", str(self.port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
        ]

    def run(self, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
        """同步执行（用于探测、检查、kill 等短命令）"""
        if self.is_local:
            full = ["bash", "-lc", cmd]
        else:
            full = ["ssh"] + self.ssh_opts() + [f"{self.user}@{self.host}", cmd]
        return subprocess.run(full, capture_output=True, text=True, timeout=timeout)

    def launch_bg(self, cmd: str, log_stdout: str) -> bool:
        """
        ① 真正的后台启动，本地和远端行为统一：
           - 本地：直接 Popen(["bash","-lc", cmd])，不等待
           - 远端：ssh + nohup + &（ssh 自身不等待，因为远端 nohup 立即返回）
        返回是否成功启动（只检查启动本身，不等待任务完成）。
        """
        if self.is_local:
            # 本地：直接 Popen 后台运行，父进程不等待
            # 用 os.setsid 让子进程成为进程组 leader，便于后续 kill
            expanded = os.path.expanduser(log_stdout)
            Path(expanded).parent.mkdir(parents=True, exist_ok=True)
            log_f = open(expanded, "w")
            try:
                proc = subprocess.Popen(
                    ["bash", "-lc", cmd],
                    stdout=log_f, stderr=subprocess.STDOUT,
                    start_new_session=True,   # 新进程组，与主控解耦
                    close_fds=True,
                )
                log.info(f"[{self.name}] 本地后台启动 PID={proc.pid} ✅")
                return True
            except Exception as e:
                log.error(f"[{self.name}] 本地启动失败: {e}")
                return False
            finally:
                log_f.close()
        else:
            # 远端：ssh 执行 nohup，ssh 连接完成后立即返回
            remote_cmd = f"nohup bash -c '{cmd}' > {log_stdout} 2>&1 &"
            full = ["ssh"] + self.ssh_opts() + [f"{self.user}@{self.host}", remote_cmd]
            r = subprocess.run(full, capture_output=True, text=True, timeout=15)
            ok = r.returncode in (0, 1)
            if ok:
                log.info(f"[{self.name}] 远端后台启动 ✅")
            else:
                log.error(f"[{self.name}] 远端启动失败 rc={r.returncode}: {r.stderr[:200]}")
            return ok

    def upload(self, local_path: str, remote_path: str):
        if self.is_local:
            dst = os.path.expanduser(remote_path)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            if Path(local_path).resolve() == Path(dst).resolve():
                return
            shutil.copy2(local_path, dst)
        else:
            subprocess.run(
                ["scp", f"-P{self.port}"] + self.ssh_opts()[:-2] +
                [local_path, f"{self.user}@{self.host}:{remote_path}"],
                capture_output=True, check=False,
            )


def load_servers(yaml_path: str) -> list[Server]:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    raw = data.get("servers", data) if isinstance(data, dict) else data
    return [Server(
        name=s.get("name", s["host"]), host=s["host"],
        port=s.get("port", 22), user=s.get("user", "ubuntu"),
        ssh_key=s.get("ssh_key", "~/.ssh/id_ed25519"),
        gpus=s.get("gpus", [0]), weight=float(s.get("weight", 1.0)),
        conda_env=s.get("conda_env", "base"),
        remote_script=s.get("remote_script", "~/video_pipeline/process_videos.py"),
        git_repo=s.get("git_repo", ""),
    ) for s in raw]


# ══════════════════════════════════════════════════════════════
# conda / python 路径
# ══════════════════════════════════════════════════════════════

def find_python(server: Server) -> str:
    r = server.run("which conda 2>/dev/null || echo ''")
    conda = r.stdout.strip()
    if not conda:
        search = " || ".join(f"[ -f {p} ] && echo {p}" for p in CONDA_SEARCH)
        r = server.run(f"bash -c '{search}' 2>/dev/null | head -1")
        conda = r.stdout.strip()
    if not conda:
        log.error(f"[{server.name}] 找不到 conda")
        return ""

    conda_dir = str(Path(conda).parent.parent)
    py = (f"{conda_dir}/bin/python" if server.conda_env == "base"
          else f"{conda_dir}/envs/{server.conda_env}/bin/python")

    r = server.run(f"[ -f {py} ] && echo ok || echo missing")
    if "missing" in r.stdout:
        r2 = server.run(f"{conda} run -n {server.conda_env} which python 2>/dev/null")
        py = r2.stdout.strip()

    if not py:
        log.error(f"[{server.name}] 找不到 env '{server.conda_env}' 的 python")
        return ""
    log.info(f"[{server.name}] python → {py}")
    return py


# ══════════════════════════════════════════════════════════════
# ④ git 版本同步
# ══════════════════════════════════════════════════════════════

def git_pull(server: Server) -> bool:
    repo = server.git_repo or str(Path(server.remote_script).parent)
    cmd  = (
        f"cd {repo} && "
        f"git fetch origin 2>&1 && "
        f"git reset --hard origin/$(git rev-parse --abbrev-ref HEAD) 2>&1 && "
        f"git log -1 --oneline 2>&1"
    )
    try:
        r = server.run(cmd, timeout=60)
        if r.returncode == 0:
            rev = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
            log.info(f"[{server.name}] git pull ✅  HEAD={rev}")
            return True
        log.error(f"[{server.name}] git pull 失败:\n{r.stdout}\n{r.stderr}")
        return False
    except Exception as e:
        log.error(f"[{server.name}] git pull 异常: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════

def scan_videos(root: str) -> list[str]:
    found = []
    for d, _, fns in os.walk(root):
        for fn in sorted(fns):
            if Path(fn).suffix.lower() in VIDEO_EXTENSIONS:
                found.append(os.path.join(d, fn))
    return found


def check_server(server: Server) -> bool:
    try:
        r = server.run("echo ok")
        return r.returncode == 0 and "ok" in r.stdout
    except Exception as e:
        log.warning(f"[{server.name}] 连接失败: {e}")
        return False


def deploy_script(server: Server, local_script: str):
    remote_dir = str(Path(server.remote_script).parent)
    server.run(f"mkdir -p {remote_dir}")
    server.upload(local_script, server.remote_script)
    log.info(f"[{server.name}] 脚本部署 ✅")


# ══════════════════════════════════════════════════════════════
# ② kill（PGID 进程组，覆盖 Python + ffmpeg）
# ══════════════════════════════════════════════════════════════

def kill_server(server: Server, pid_file: str):
    """
    读 PID 文件 → 获取 PGID → kill 整个进程组。
    process_videos 启动时调用 os.setsid() 成为进程组 leader，
    PGID == PID，kill -- -PGID 可以覆盖所有 ffmpeg 子进程。
    """
    kill_cmd = f"""
set -e
if [ -f {pid_file} ]; then
  PID=$(cat {pid_file})
  # PGID = 进程组 ID（process_videos 用 os.setsid 让 PGID==PID）
  PGID=$(ps -o pgid= -p $PID 2>/dev/null | tr -d ' ' || echo $PID)
  echo "SIGTERM → PGID=$PGID (PID=$PID)"
  kill -TERM -- -$PGID 2>/dev/null || true
  sleep 3
  # 确认是否还活着
  if ps -p $PID > /dev/null 2>&1; then
    echo "SIGKILL → PGID=$PGID"
    kill -KILL -- -$PGID 2>/dev/null || true
  fi
  rm -f {pid_file}
  echo "done"
else
  # 无 PID 文件：全局 pkill 兜底（覆盖 ffmpeg）
  echo "no pid file, fallback pkill"
  pkill -TERM -f process_videos.py 2>/dev/null || true
  sleep 3
  pkill -KILL -f process_videos.py 2>/dev/null || true
  pkill -KILL -f ffmpeg 2>/dev/null || true
  echo "fallback done"
fi
"""
    try:
        r = server.run(kill_cmd.strip(), timeout=20)
        console.print(f"  [{server.name}] {r.stdout.strip() or 'done'}")
        if r.stderr.strip():
            log.debug(f"[{server.name}] kill stderr: {r.stderr.strip()}")
    except Exception as e:
        console.print(f"  [{server.name}] kill 异常: {e}")


def stop_all(servers: list[Server], pid_files: dict[str, str]):
    console.rule("[bold red]⛔  广播终止[/bold red]")
    ts = [
        threading.Thread(
            target=kill_server,
            args=(s, pid_files.get(s.name, f"~/pipeline_{s.name}.pid")),
            daemon=True,
        )
        for s in servers
    ]
    for t in ts: t.start()
    for t in ts: t.join(timeout=25)
    console.print("[bold green]所有 kill 命令已发送 ✅[/bold green]")
    console.print("下次加 [bold]--resume[/bold] 继续（队列状态已保留）。")


# ══════════════════════════════════════════════════════════════
# 启动命令
# ══════════════════════════════════════════════════════════════

def build_worker_cmd(
    server: Server, python_path: str,
    input_dir: str, output_dir: str,
    queue_dir: str, pid_file: str,
    workers: int, log_path: str,
) -> str:
    """
    构建 process_videos.py 的启动命令字符串。
    注意：不含 nohup/& —— 由 launch_bg() 负责后台化。
    """
    gpu_ids    = ",".join(str(g) for g in server.gpus)
    script_dir = str(Path(server.remote_script).parent)
    return (
        f"PYTHONPATH={script_dir}:$PYTHONPATH "
        f"{python_path} {server.remote_script} "
        f"{input_dir} {output_dir} "
        f"--gpu-ids {gpu_ids} "
        f"--workers {workers} "
        f"--log-file {log_path} "
        f"--queue-dir {queue_dir} "
        f"--worker-id {server.name} "
        f"--pid-file {pid_file}"
    )


# ══════════════════════════════════════════════════════════════
# 日志 tail & 等待
# ══════════════════════════════════════════════════════════════

def tail_log(server: Server, log_stdout: str, stop_event: threading.Event):
    if server.is_local:
        full = ["tail", "-f", os.path.expanduser(log_stdout)]
    else:
        full = (["ssh"] + server.ssh_opts() +
                [f"{server.user}@{server.host}", f"tail -f {log_stdout}"])
    try:
        proc = subprocess.Popen(
            full, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        while not stop_event.is_set():
            line = proc.stdout.readline()
            if line:
                console.print(f"  [dim][{server.name}][/dim] {line.rstrip()}")
            else:
                time.sleep(0.3)
        proc.terminate()
    except Exception:
        pass


def wait_done(server: Server, pid_file: str, poll: int = 10):
    """等待 PID 文件消失（process_videos 退出时删除它）"""
    while True:
        try:
            r = server.run(f"[ -f {pid_file} ] && echo running || echo done", timeout=10)
            if "done" in r.stdout:
                return
        except Exception:
            pass
        time.sleep(poll)


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="分布式调度器 v5.1")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--servers",  required=True)
    parser.add_argument("--deploy",   action="store_true", help="部署脚本到各机器")
    parser.add_argument("--git-pull", action="store_true", help="部署前先在远端 git pull")
    parser.add_argument("--force",    action="store_true", help="重置队列，重新处理所有文件")
    parser.add_argument("--workers",  type=int, default=0, help="每台并发数（0=自动 NVENC→3）")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--check",    action="store_true", help="仅检测连通性")
    parser.add_argument("--stop",     action="store_true", help="广播终止所有服务器")
    args = parser.parse_args()

    servers = load_servers(args.servers)

    # ── 连通性 ──────────────────────────────────────────────────
    console.rule("[bold cyan]🎬  分布式调度器 v5.1[/bold cyan]")
    alive = []
    for s in servers:
        ok   = check_server(s)
        icon = "✅" if ok else "❌"
        console.print(
            f"  {icon} [bold]{s.name}[/bold]  {s.user}@{s.host}:{s.port}"
            f"  GPU={s.gpus}  w={s.weight}  env={s.conda_env}"
            + (f"  git={s.git_repo}" if s.git_repo else "")
        )
        if ok:
            alive.append(s)

    pid_files = {s.name: f"~/pipeline_{s.name}.pid" for s in alive}

    # ── --stop ──────────────────────────────────────────────────
    if args.stop:
        stop_all(alive, pid_files)
        return

    if args.check:
        return

    if not alive:
        console.print("[red]没有可用服务器！[/red]")
        sys.exit(1)

    if not args.input_dir or not args.output_dir:
        console.print("[red]需要 --input-dir 和 --output-dir[/red]")
        sys.exit(1)

    # ── ④ git pull ──────────────────────────────────────────────
    if args.git_pull:
        console.print("\n[bold]Git 版本同步...[/bold]")
        for s in alive:
            git_pull(s)

    # ── python 路径 ──────────────────────────────────────────────
    console.print("\n[bold]探测 python 路径...[/bold]")
    py_paths: dict[str, str] = {}
    for s in alive:
        p = find_python(s)
        if p:
            py_paths[s.name] = p
        else:
            console.print(f"  [red][{s.name}] 跳过[/red]")
    alive = [s for s in alive if s.name in py_paths]
    if not alive:
        console.print("[red]所有服务器都找不到 python，退出。[/red]")
        sys.exit(1)

    # ── 初始化队列 ──────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from task_queue import TaskQueue

    console.print(f"\n[bold]扫描: {args.input_dir}[/bold]")
    all_videos = scan_videos(args.input_dir)
    if not all_videos:
        console.print("[red]未找到视频文件！[/red]")
        sys.exit(1)

    queue_dir = str(Path(args.output_dir) / ".queue")
    q = TaskQueue(queue_dir=queue_dir, worker_id="dispatcher")
    q.init_queue(src_files=all_videos, start_idx=1, force=args.force)
    stats = q.stats()
    total = stats.get("pending", 0) + stats.get("claimed", 0) + stats.get("done", 0)
    console.print(
        f"  队列: total={total}  "
        f"pending={stats.get('pending',0)}  "
        f"done={stats.get('done',0)}  "
        f"error={stats.get('error',0)}"
    )
    if stats.get("pending", 0) + stats.get("claimed", 0) == 0:
        console.print("[green]队列中无待处理任务，全部完成！[/green]")
        return

    # ── 显示服务器列表 ──────────────────────────────────────────
    tbl = Table(title="参与服务器（动态抢队列）", box=None)
    tbl.add_column("服务器", style="cyan")
    tbl.add_column("GPU",    style="yellow")
    tbl.add_column("并发",   style="bold white")
    tbl.add_column("权重",   style="dim")
    for s in alive:
        tbl.add_row(s.name, str(s.gpus), str(args.workers or "auto(3)"), str(s.weight))
    console.print(tbl)
    console.print(
        "  [dim]动态抢任务：任意机器随时加入/退出，"
        "掉线机器任务 10min 后自动释放给其他机器[/dim]"
    )

    if args.dry_run:
        return

    # ── 部署脚本（含 task_queue.py）────────────────────────────
    if args.deploy:
        console.print("\n[bold]部署脚本...[/bold]")
        local_pv = str(Path(__file__).parent / "process_videos.py")
        local_tq = str(Path(__file__).parent / "task_queue.py")
        for s in alive:
            deploy_script(s, local_pv)
            remote_tq = str(Path(s.remote_script).parent / "task_queue.py")
            s.upload(local_tq, remote_tq)
            log.info(f"[{s.name}] task_queue.py ✅")

    # ── ① 启动各机器（本地用 Popen，远端用 SSH+nohup）──────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    stop_event  = threading.Event()
    remote_jobs = []   # (server, log_stdout, pid_file)

    for s in alive:
        log_path   = f"~/pipeline_{s.name}.log"
        log_stdout = log_path + ".stdout"
        pid_file   = pid_files[s.name]

        cmd = build_worker_cmd(
            server      = s,
            python_path = py_paths[s.name],
            input_dir   = args.input_dir,
            output_dir  = args.output_dir,
            queue_dir   = queue_dir,
            pid_file    = pid_file,
            workers     = args.workers,
            log_path    = log_path,
        )
        console.print(f"\n[bold][{s.name}][/bold]:\n  [dim]{cmd}[/dim]")

        ok = s.launch_bg(cmd, log_stdout)
        if ok:
            remote_jobs.append((s, log_stdout, pid_file))
            threading.Thread(
                target=tail_log,
                args=(s, log_stdout, stop_event),
                daemon=True,
            ).start()

    if not remote_jobs:
        console.print("[red]没有服务器成功启动！[/red]")
        sys.exit(1)

    # ── Ctrl+C ──────────────────────────────────────────────────
    def _sigint(sig, frame):
        console.print(
            "\n[yellow]⚠  Ctrl+C\n"
            "   [1] 仅退出主控（远端继续跑）\n"
            "   [2] 广播终止所有服务器（ffmpeg 一并停止）\n"
            "   输入 1 或 2：[/yellow]",
            end="",
        )
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            choice = input().strip()
        except Exception:
            choice = "1"
        stop_event.set()
        if choice == "2":
            stop_all(alive, pid_files)
        else:
            console.print(
                "[yellow]主控退出，远端继续运行。\n"
                f"终止命令: python distributed_dispatch.py"
                f" --servers {args.servers} --stop[/yellow]"
            )
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    # ── 等待完成 ────────────────────────────────────────────────
    console.print(
        "\n[bold]等待各服务器完成...[/bold]\n"
        "  Ctrl+C → 选择退出/终止\n"
        f"  随时终止: python distributed_dispatch.py --servers {args.servers} --stop"
    )
    wait_ts = []
    for s, _, pf in remote_jobs:
        t = threading.Thread(
            target=lambda sv=s, p=pf: (
                wait_done(sv, p),
                log.info(f"[{sv.name}] ✅ 全部完成"),
            ),
            daemon=True,
        )
        t.start()
        wait_ts.append(t)

    for t in wait_ts:
        t.join()

    stop_event.set()
    console.rule("[bold green]所有服务器处理完成 🎉[/bold green]")


if __name__ == "__main__":
    main()