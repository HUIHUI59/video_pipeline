#!/usr/bin/env python3
"""
distributed_dispatch.py  v5.2
════════════════════════════════════════════════════════════════
kill 方案（问题2修复）：
  process_videos 在模块顶部调用 os.setsid()，自身成为进程组 leader
  PGID == PID，所有 ffmpeg 子进程（未用 start_new_session）也在同组
  kill_server 执行：kill -TERM/-KILL -- -$PGID
  → 一条命令覆盖 Python + 所有 ffmpeg，CPU 立即降下来

恢复说明（问题3）：
  - 主控退出（Ctrl+C 选1）：远端继续跑，不影响
  - 广播 kill（Ctrl+C 选2 / --stop）：队列状态保留（done 的不重跑）
    恢复：python distributed_dispatch.py ... 即可（init_queue 自动重置
    上次 claimed 为 pending）
  - 单台机器重新加入：在该机器上直接运行 process_videos.py --queue-dir ...
════════════════════════════════════════════════════════════════
"""

import os, sys, json, shutil, signal, logging, argparse
import subprocess, threading, time
from pathlib import Path
from dataclasses import dataclass

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
    def is_local(self): return self.host in ("localhost","127.0.0.1")

    def ssh_opts(self):
        return ["-i",os.path.expanduser(self.ssh_key),
                "-p",str(self.port),
                "-o","StrictHostKeyChecking=no",
                "-o","ConnectTimeout=10",
                "-o","BatchMode=yes"]

    def run(self, cmd:str, timeout:int=30) -> subprocess.CompletedProcess:
        """同步执行（探测/kill 等短命令用）"""
        if self.is_local:
            full = ["bash","-lc", cmd]
        else:
            full = ["ssh"]+self.ssh_opts()+[f"{self.user}@{self.host}", cmd]
        return subprocess.run(full, capture_output=True, text=True, timeout=timeout)

    def launch_bg(self, cmd:str, log_stdout:str) -> bool:
        """
        后台启动 process_videos.py。
        本地：Popen(start_new_session=False) — 进程在同一组，但主控退出不影响它
              （因为 process_videos 自己调用了 os.setsid()）
        远端：SSH + nohup &
        """
        if self.is_local:
            log_p = os.path.expanduser(log_stdout)
            Path(log_p).parent.mkdir(parents=True, exist_ok=True)
            lf = open(log_p, "w")
            try:
                proc = subprocess.Popen(
                    ["bash","-lc", cmd],
                    stdout=lf, stderr=subprocess.STDOUT,
                    close_fds=True,
                    # 不设 start_new_session：process_videos 自己 setsid()
                )
                log.info(f"[{self.name}] 本地后台启动 PID={proc.pid} ✅")
                return True
            except Exception as e:
                log.error(f"[{self.name}] 本地启动失败: {e}"); return False
            finally:
                lf.close()
        else:
            remote = f"nohup bash -c '{cmd}' > {log_stdout} 2>&1 &"
            full   = ["ssh"]+self.ssh_opts()+[f"{self.user}@{self.host}", remote]
            r = subprocess.run(full, capture_output=True, text=True, timeout=15)
            ok = r.returncode in (0,1)
            if ok: log.info(f"[{self.name}] 远端后台启动 ✅")
            else:  log.error(f"[{self.name}] 远端启动失败 rc={r.returncode}: {r.stderr[:200]}")
            return ok

    def upload(self, local:str, remote:str):
        if self.is_local:
            dst = os.path.expanduser(remote)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            if Path(local).resolve() == Path(dst).resolve(): return
            shutil.copy2(local, dst)
        else:
            subprocess.run(
                ["scp",f"-P{self.port}"]+self.ssh_opts()[:-2]+
                [local, f"{self.user}@{self.host}:{remote}"],
                capture_output=True, check=False)


def load_servers(yaml_path:str) -> list[Server]:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    raw = data.get("servers", data) if isinstance(data,dict) else data
    return [Server(
        name=s.get("name",s["host"]), host=s["host"],
        port=s.get("port",22), user=s.get("user","ubuntu"),
        ssh_key=s.get("ssh_key","~/.ssh/id_ed25519"),
        gpus=s.get("gpus",[0]), weight=float(s.get("weight",1.0)),
        conda_env=s.get("conda_env","base"),
        remote_script=s.get("remote_script","~/video_pipeline/process_videos.py"),
        git_repo=s.get("git_repo",""),
    ) for s in raw]

# ══════════════════════════════════════════════════════════════
# conda / python 路径
# ══════════════════════════════════════════════════════════════

def find_python(server:Server) -> str:
    r = server.run("which conda 2>/dev/null || echo ''")
    conda = r.stdout.strip()
    if not conda:
        search = " || ".join(f"[ -f {p} ] && echo {p}" for p in CONDA_SEARCH)
        r = server.run(f"bash -c '{search}' 2>/dev/null | head -1")
        conda = r.stdout.strip()
    if not conda:
        log.error(f"[{server.name}] 找不到 conda"); return ""
    conda_dir = str(Path(conda).parent.parent)
    py = (f"{conda_dir}/bin/python" if server.conda_env=="base"
          else f"{conda_dir}/envs/{server.conda_env}/bin/python")
    r = server.run(f"[ -f {py} ] && echo ok || echo missing")
    if "missing" in r.stdout:
        r2 = server.run(f"{conda} run -n {server.conda_env} which python 2>/dev/null")
        py = r2.stdout.strip()
    if not py:
        log.error(f"[{server.name}] 找不到 '{server.conda_env}' 的 python"); return ""
    log.info(f"[{server.name}] python → {py}")
    return py

# ══════════════════════════════════════════════════════════════
# git pull
# ══════════════════════════════════════════════════════════════

def git_pull(server:Server) -> bool:
    repo = server.git_repo or str(Path(server.remote_script).parent)
    cmd  = (f"cd {repo} && git fetch origin 2>&1 && "
            f"git reset --hard origin/$(git rev-parse --abbrev-ref HEAD) 2>&1 && "
            f"git log -1 --oneline 2>&1")
    try:
        r = server.run(cmd, timeout=60)
        if r.returncode == 0:
            rev = (r.stdout.strip().splitlines() or ["?"])[-1]
            log.info(f"[{server.name}] git pull ✅  HEAD={rev}"); return True
        log.error(f"[{server.name}] git pull 失败:\n{r.stdout}\n{r.stderr}"); return False
    except Exception as e:
        log.error(f"[{server.name}] git pull 异常: {e}"); return False

# ══════════════════════════════════════════════════════════════
# kill（PGID，覆盖 Python + ffmpeg）
# ══════════════════════════════════════════════════════════════

def kill_server(server:Server, pid_file:str):
    """
    读 PID 文件 → PGID == PID（process_videos 已 setsid）
    kill -TERM -- -PGID → 整个进程组（Python + 所有 ffmpeg）
    等待 3s → kill -KILL -- -PGID 兜底 → 验证进程已死
    """
    cmd = f"""
if [ -f {pid_file} ]; then
  PID=$(cat {pid_file})
  if ! kill -0 $PID 2>/dev/null; then
    echo "already stopped (stale pid=$PID)"
    rm -f {pid_file}
  else
    echo "TERM → PGID=$PID"
    kill -TERM -- -$PID 2>/dev/null || true
    sleep 3
    if kill -0 $PID 2>/dev/null; then
      echo "KILL → PGID=$PID"
      kill -KILL -- -$PID 2>/dev/null || true
      sleep 1
    fi
    if kill -0 $PID 2>/dev/null; then
      echo "FAILED: PID=$PID 仍在运行"
    else
      echo "✅ 已终止 (pid=$PID)"
    fi
    rm -f {pid_file}
  fi
else
  if pgrep -f process_videos.py > /dev/null 2>&1; then
    echo "no pid file, pkill fallback"
    pkill -TERM -f process_videos.py 2>/dev/null || true
    sleep 3
    pkill -KILL -f process_videos.py 2>/dev/null || true
    pkill -KILL -f ffmpeg 2>/dev/null || true
    echo "fallback done"
  else
    echo "already stopped (no process found)"
  fi
fi
"""
    try:
        r = server.run(cmd.strip(), timeout=25)
        console.print(f"  [{server.name}] {r.stdout.strip() or 'done'}")
        if r.returncode != 0:
            log.debug(f"[{server.name}] kill stderr: {r.stderr.strip()}")
    except Exception as e:
        console.print(f"  [{server.name}] kill 异常: {e}")

def stop_all(servers:list[Server], pid_files:dict):
    console.rule("[bold red]⛔  广播终止[/bold red]")
    ts = [threading.Thread(target=kill_server,
                           args=(s, pid_files.get(s.name, f"~/pipeline_{s.name}.pid")),
                           daemon=True) for s in servers]
    for t in ts: t.start()
    for t in ts: t.join(timeout=30)
    console.print("[bold green]kill 命令执行完毕[/bold green]")
    console.print("队列状态已保留。重新运行命令即可续传（无需 --resume 参数）。")

def is_worker_alive(server:Server, pid_file:str) -> tuple[bool, str]:
    """检查 worker 是否在运行。返回 (alive, pid_or_reason)"""
    cmd = (f"if [ -f {pid_file} ]; then "
           f"PID=$(cat {pid_file}); "
           f"if kill -0 $PID 2>/dev/null; then echo \"alive:$PID\"; "
           f"else echo \"dead:stale_pid\"; fi; "
           f"else echo \"dead:no_pidfile\"; fi")
    try:
        r = server.run(cmd, timeout=10)
        out = r.stdout.strip()
        if out.startswith("alive:"):
            return True, out.split(":", 1)[1]
        return False, out
    except Exception as e:
        return False, str(e)

def status_all(servers:list[Server], pid_files:dict, queue_dir:str=""):
    console.rule("[bold cyan]📊  系统状态[/bold cyan]")
    tbl = Table(box=None)
    tbl.add_column("服务器", style="cyan"); tbl.add_column("状态", style="bold")
    tbl.add_column("PID"); tbl.add_column("PID文件")
    for s in servers:
        pf = pid_files.get(s.name, f"~/pipeline_{s.name}.pid")
        alive, info = is_worker_alive(s, pf)
        status = "[green]运行中 ▶[/green]" if alive else "[red]已停止 ■[/red]"
        pid    = info if alive else "-"
        tbl.add_row(s.name, status, pid, pf)
    console.print(tbl)
    if queue_dir:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from task_queue import TaskQueue
            q = TaskQueue(queue_dir=os.path.expanduser(queue_dir), worker_id="status")
            s = q.stats()
            total = sum(s.values())
            done  = s.get("done", 0)
            console.print(
                f"\n  队列: total={total}  "
                f"[green]done={done}[/green]  "
                f"[yellow]pending={s.get('pending',0)}[/yellow]  "
                f"[blue]claimed={s.get('claimed',0)}[/blue]  "
                f"[red]error={s.get('error',0)}[/red]"
                + (f"  ({done/total*100:.1f}%)" if total else ""))
        except Exception as e:
            console.print(f"  [red]无法读取队列: {e}[/red]")
    else:
        console.print("  [dim](提供 --output-dir 可显示队列进度)[/dim]")

# ══════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════

def scan_videos(root:str) -> list[str]:
    found = []
    for d,_,fns in os.walk(root):
        for fn in sorted(fns):
            if Path(fn).suffix.lower() in VIDEO_EXTENSIONS:
                found.append(os.path.join(d,fn))
    return found

def check_server(server:Server) -> bool:
    try:
        r = server.run("echo ok")
        return r.returncode==0 and "ok" in r.stdout
    except Exception as e:
        log.warning(f"[{server.name}] 连接失败: {e}"); return False

def deploy_script(server:Server, local:str):
    server.run(f"mkdir -p {Path(server.remote_script).parent}")
    server.upload(local, server.remote_script)
    log.info(f"[{server.name}] 脚本部署 ✅")

def build_cmd(server:Server, python:str, input_dir:str, output_dir:str,
              queue_dir:str, pid_file:str, workers:int, log_path:str) -> str:
    gpus = ",".join(str(g) for g in server.gpus)
    sdir = str(Path(server.remote_script).parent)
    return (f"PYTHONPATH={sdir}:$PYTHONPATH "
            f"{python} {server.remote_script} "
            f"{input_dir} {output_dir} "
            f"--gpu-ids {gpus} --workers {workers} "
            f"--log-file {log_path} "
            f"--queue-dir {queue_dir} "
            f"--worker-id {server.name} "
            f"--pid-file {pid_file}")

def tail_log(server:Server, log_stdout:str, stop_ev:threading.Event):
    if server.is_local:
        full = ["tail","-f",os.path.expanduser(log_stdout)]
    else:
        full = ["ssh"]+server.ssh_opts()+[f"{server.user}@{server.host}",
                                           f"tail -f {log_stdout}"]
    try:
        proc = subprocess.Popen(full, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
        while not stop_ev.is_set():
            line = proc.stdout.readline()
            if line: console.print(f"  [dim][{server.name}][/dim] {line.rstrip()}")
            else: time.sleep(0.3)
        proc.terminate()
    except Exception: pass

def wait_done(server:Server, pid_file:str, poll:int=10):
    while True:
        try:
            r = server.run(f"[ -f {pid_file} ] && echo running || echo done", timeout=10)
            if "done" in r.stdout: return
        except Exception: pass
        time.sleep(poll)

def check_queue_access(server:Server, queue_dir:str) -> bool:
    """检测远端机器是否能看到队列 JSON 文件本身（而不只是父目录）"""
    queue_file = f"{queue_dir}/.pipeline_queue.json"
    try:
        r = server.run(f"[ -f {queue_file} ] && echo ok || echo missing", timeout=10)
        return "ok" in r.stdout
    except Exception:
        return False

def wait_for_pid(server:Server, pid_file:str, timeout:int=30) -> bool:
    """launch 后轮询 PID 文件，确认 worker 进程真正启动了"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = server.run(f"[ -f {pid_file} ] && echo ready || echo waiting", timeout=5)
            if "ready" in r.stdout: return True
        except Exception: pass
        time.sleep(3)
    return False

# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="分布式调度器 v5.2")
    parser.add_argument("--input-dir");  parser.add_argument("--output-dir")
    parser.add_argument("--servers", required=True)
    parser.add_argument("--deploy",   action="store_true", help="部署脚本到各机器")
    parser.add_argument("--git-pull", action="store_true", help="部署前 git pull")
    parser.add_argument("--force",    action="store_true", help="重置队列（全部重跑）")
    parser.add_argument("--workers",  type=int, default=0)
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--check",    action="store_true")
    parser.add_argument("--stop",     action="store_true", help="终止服务器（可配合 --target）")
    parser.add_argument("--status",   action="store_true", help="查看各服务器运行状态和队列进度")
    parser.add_argument("--target",   type=str, default="",
                        help="指定操作的机器名，逗号分隔。例: --target A6000 或 --target A6000,A8000")
    args = parser.parse_args()

    servers = load_servers(args.servers)

    # ── 连通性 ──
    console.rule("[bold cyan]🎬  分布式调度器 v5.2[/bold cyan]")
    alive = []
    for s in servers:
        ok = check_server(s)
        console.print(f"  {'✅' if ok else '❌'} [bold]{s.name}[/bold]  "
                      f"{s.user}@{s.host}:{s.port}  GPU={s.gpus}  "
                      f"w={s.weight}  env={s.conda_env}"
                      +(f"  git={s.git_repo}" if s.git_repo else ""))
        if ok: alive.append(s)

    pid_files = {s.name: f"~/pipeline_{s.name}.pid" for s in alive}

    # ── --target 过滤（用于 --stop 和启动）──
    if args.target:
        target_set  = {n.strip() for n in args.target.split(",") if n.strip()}
        not_found   = target_set - {s.name for s in alive}
        if not_found:
            console.print(f"  [yellow]⚠ 未找到或不可达: {', '.join(sorted(not_found))}[/yellow]")
        op_servers = [s for s in alive if s.name in target_set]
        if not op_servers:
            console.print("[red]--target 指定的机器均不可用[/red]"); sys.exit(1)
    else:
        op_servers = alive

    if args.stop:
        stop_all(op_servers, pid_files); return
    if args.status:
        qd = str(Path(args.output_dir) / ".queue") if args.output_dir else ""
        status_all(alive, pid_files, qd); return   # status 始终显示所有机器
    if args.check:    return
    if not alive:     console.print("[red]没有可用服务器！[/red]"); sys.exit(1)
    if not args.input_dir or not args.output_dir:
        console.print("[red]需要 --input-dir 和 --output-dir[/red]"); sys.exit(1)

    # ── git pull ──
    if args.git_pull:
        console.print("\n[bold]Git 版本同步...[/bold]")
        for s in op_servers: git_pull(s)

    # ── python 路径 ──
    console.print("\n[bold]探测 python 路径...[/bold]")
    py_paths:dict[str,str] = {}
    for s in op_servers:
        p = find_python(s)
        if p: py_paths[s.name] = p
        else: console.print(f"  [red][{s.name}] 跳过[/red]")
    op_servers = [s for s in op_servers if s.name in py_paths]
    if not op_servers:
        console.print("[red]所有服务器都找不到 python！[/red]"); sys.exit(1)

    # ── 初始化队列 ──
    sys.path.insert(0, str(Path(__file__).parent))
    from task_queue import TaskQueue

    all_videos = scan_videos(args.input_dir)
    if not all_videos:
        console.print("[red]未找到视频文件！[/red]"); sys.exit(1)

    queue_dir = str(Path(args.output_dir) / ".queue")
    q = TaskQueue(queue_dir=queue_dir, worker_id="dispatcher")
    pre = q.stats()   # 重置前的状态（用于续传提示）
    q.init_queue(src_files=all_videos, start_idx=1, force=args.force)
    s = q.stats()
    total_in_dir = len(all_videos)
    total_in_q   = sum(s.values())
    done_cnt     = s.get("done", 0)
    pct          = f"{done_cnt/total_in_q*100:.1f}%" if total_in_q else "0%"
    was_resuming = pre.get("done", 0) > 0 or pre.get("error", 0) > 0
    console.print(f"\n  输入目录视频总数 : [bold]{total_in_dir}[/bold]")
    if was_resuming:
        reset_count = pre.get("claimed", 0)
        console.print(
            f"  [yellow]续传模式[/yellow] : "
            f"[green]done={done_cnt}[/green] ({pct})  "
            f"[yellow]pending={s.get('pending',0)}[/yellow]  "
            f"[red]error={s.get('error',0)}[/red]"
            + (f"  (已重置 {reset_count} 个中断任务)" if reset_count else ""))
    else:
        console.print(f"  [green]全新开始[/green] : 共 {total_in_q} 个任务")
    if s.get("pending",0) == 0 and s.get("claimed",0) == 0:
        console.print("[green]队列无待处理任务，全部完成！[/green]"); return

    # ── 服务器表 ──
    tbl = Table(title="参与服务器（动态抢队列）", box=None)
    tbl.add_column("服务器",style="cyan"); tbl.add_column("GPU",style="yellow")
    tbl.add_column("并发",style="bold"); tbl.add_column("权重",style="dim")
    for sv in op_servers:
        tbl.add_row(sv.name, str(sv.gpus), str(args.workers or "auto(3)"), str(sv.weight))
    console.print(tbl)

    if args.dry_run: return

    # ── 部署 ──
    if args.deploy:
        console.print("\n[bold]部署脚本...[/bold]")
        local_pv = str(Path(__file__).parent / "process_videos.py")
        local_tq = str(Path(__file__).parent / "task_queue.py")
        for sv in op_servers:
            deploy_script(sv, local_pv)
            sv.upload(local_tq, str(Path(sv.remote_script).parent / "task_queue.py"))
            log.info(f"[{sv.name}] task_queue.py ✅")

    # ── 启动 ──
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    stop_ev = threading.Event()
    jobs    = []

    for sv in op_servers:
        log_path   = f"~/pipeline_{sv.name}.log"
        log_stdout = log_path + ".stdout"
        pid_file   = pid_files[sv.name]

        worker_alive, worker_pid = is_worker_alive(sv, pid_file)
        if worker_alive:
            log.info(f"[{sv.name}] Worker 已在运行 (PID={worker_pid})，附加监控日志")
            jobs.append((sv, log_stdout, pid_file))
            threading.Thread(target=tail_log,
                             args=(sv, log_stdout, stop_ev), daemon=True).start()
            continue

        if not check_queue_access(sv, queue_dir):
            console.print(
                f"\n  [bold red][{sv.name}] ⚠ 队列文件不可访问[/bold red]\n"
                f"  路径: {queue_dir}/.pipeline_queue.json\n"
                f"  该机器上 /mnt/movies/Films/output 可能是本地空目录而非 NAS 挂载点。\n"
                f"  请在 {sv.name} 上挂载共享存储后重试，跳过此机器。")
            continue

        cmd = build_cmd(sv, py_paths[sv.name], args.input_dir, args.output_dir,
                        queue_dir, pid_file, args.workers, log_path)
        console.print(f"\n[bold][{sv.name}][/bold]:\n  [dim]{cmd}[/dim]")
        if sv.launch_bg(cmd, log_stdout):
            log.info(f"[{sv.name}] 等待 worker 写入 PID 文件（最多 30s）...")
            if wait_for_pid(sv, pid_file, timeout=30):
                log.info(f"[{sv.name}] Worker 已确认启动 ✅")
                jobs.append((sv, log_stdout, pid_file))
                threading.Thread(target=tail_log,
                                 args=(sv, log_stdout, stop_ev), daemon=True).start()
            else:
                console.print(
                    f"  [bold red][{sv.name}] ❌ 30s 内未见 PID 文件，worker 可能启动失败\n"
                    f"  检查日志: {log_stdout}[/bold red]")

    if not jobs:
        console.print("[red]没有服务器启动成功！[/red]"); sys.exit(1)

    # ── Ctrl+C：直接退出主控，远端继续跑 ──
    sv_names = ",".join(s.name for s in op_servers)
    def _sigint(sig, frame):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        stop_ev.set()
        console.print(
            "\n[yellow]⚠  主控已退出，worker 继续运行。[/yellow]\n"
            f"  查看进度  →  --status --output-dir <路径>\n"
            f"  停止全部  →  --stop\n"
            f"  停止单台  →  --stop --target [bold]{sv_names.split(',')[0]}[/bold]\n"
            f"  继续监控  →  重新运行原命令（已运行的 worker 不会重复启动）")
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    console.print(
        f"\n[bold]等待完成...[/bold]  (每 30s 刷新进度)\n"
        f"  Ctrl+C → 主控退出（worker 继续）\n"
        f"  停止全部 → --stop  |  停止单台 → --stop --target {sv_names.split(',')[0]}")
    wts = []
    for sv,_,pf in jobs:
        t = threading.Thread(
            target=lambda s=sv,p=pf: (wait_done(s,p), log.info(f"[{s.name}] ✅ 完成")),
            daemon=True)
        t.start(); wts.append(t)

    while any(t.is_alive() for t in wts):
        for t in wts:
            t.join(timeout=30)
        s2    = q.stats()
        tot2  = sum(s2.values())
        done2 = s2.get("done", 0)
        pct2  = f"{done2/tot2*100:.1f}%" if tot2 else "0%"
        console.print(
            f"  [dim]{time.strftime('%H:%M:%S')}[/dim]  "
            f"进度 [bold]{pct2}[/bold]  "
            f"[green]done={done2}[/green]  "
            f"[yellow]pending={s2.get('pending',0)}[/yellow]  "
            f"[blue]claimed={s2.get('claimed',0)}[/blue]  "
            f"[red]error={s2.get('error',0)}[/red]")

    stop_ev.set()
    console.rule("[bold green]所有服务器完成 🎉[/bold green]")

if __name__ == "__main__":
    main()