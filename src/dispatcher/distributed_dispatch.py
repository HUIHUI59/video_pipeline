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

import os, sys, json, shutil, signal, logging, argparse, atexit
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

def _restore_terminal():
    """进程退出时恢复终端状态（防止 Rich 退出后终端回显消失）"""
    try:
        console.show_cursor(True)
    except Exception:
        pass
    try:
        subprocess.run(["stty", "sane"], check=False, timeout=2,
                       stdin=subprocess.DEVNULL, capture_output=True)
    except Exception:
        pass

atexit.register(_restore_terminal)

VIDEO_EXTENSIONS = {
    ".mp4",".mkv",".avi",".mov",".wmv",".flv",
    ".m4v",".ts",".mts",".m2ts",".webm",".rmvb",
    ".rm",".mpeg",".mpg",".vob",".3gp",
}
CONDA_SEARCH = [
    "~/miniconda3/bin/conda","~/anaconda3/bin/conda",
    "/opt/conda/bin/conda","/usr/local/conda/bin/conda",
]


def _known_hosts_path() -> str:
    """项目专用 known_hosts，避免污染用户 ~/.ssh/known_hosts。"""
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(ssh_dir, 0o700)
    except OSError:
        pass
    kh = ssh_dir / "video_pipeline_known_hosts"
    if not kh.exists():
        kh.touch(mode=0o600)
    return str(kh)

# ══════════════════════════════════════════════════════════════
# Server
# ══════════════════════════════════════════════════════════════

@dataclass
class Server:
    name:str; host:str; port:int; user:str; ssh_key:str
    gpus:list; weight:float; conda_env:str; remote_script:str
    git_repo:str = ""; vsr_dir:str = ""

    @property
    def is_local(self): return self.host in ("localhost","127.0.0.1")

    def ssh_opts(self):
        # StrictHostKeyChecking=accept-new：首次 TOFU 记录 host key，后续严格验
        # 证。结合项目独立 known_hosts 能抵御 MITM，而又不阻塞首次部署。
        return ["-i",os.path.expanduser(self.ssh_key),
                "-p",str(self.port),
                "-o","StrictHostKeyChecking=accept-new",
                "-o",f"UserKnownHostsFile={_known_hosts_path()}",
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
                    stdin=subprocess.DEVNULL,   # 不继承终端 stdin，防止 kill 时终端乱码
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
            remote_dir = str(Path(remote).parent.as_posix())
            subprocess.run(
                ["ssh"]+self.ssh_opts()+
                [f"{self.user}@{self.host}", f"mkdir -p {remote_dir!s}"],
                capture_output=True, check=False)
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
        vsr_dir=s.get("vsr_dir",""),
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
  if pgrep -f "process_videos.py\\|scene_split.py\\|subtitle_remove.py" > /dev/null 2>&1; then
    echo "no pid file, pkill fallback"
    pkill -TERM -f "process_videos.py\\|scene_split.py\\|subtitle_remove.py" 2>/dev/null || true
    sleep 3
    pkill -KILL -f "process_videos.py\\|scene_split.py\\|subtitle_remove.py" 2>/dev/null || true
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
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from common.task_queue import TaskQueue
            stage_labels = [
                ("pipeline_queue", "Stage1 转码"),
                ("split_queue",    "Stage2 切分"),
                ("subtitle_queue", "Stage3 字幕"),
                ("classify_queue", "Stage4 分类"),
            ]
            for qname, label in stage_labels:
                qfile = Path(os.path.expanduser(queue_dir)) / f"{qname}.json"
                if not qfile.exists(): continue
                q = TaskQueue(queue_dir=os.path.expanduser(queue_dir),
                              worker_id="status", queue_name=qname)
                s     = q.stats()
                total = sum(s.values())
                done  = s.get("done", 0)
                console.print(
                    f"\n  [bold]{label}[/bold]: total={total}  "
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

def build_cmd_stage1(server:Server, python:str, input_dir:str, output_dir:str,
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

def build_cmd_stage2(server:Server, python:str, input_dir:str, clips_dir:str,
                     queue_dir:str, pid_file:str, workers:int, log_path:str,
                     trim_shots:int=10) -> str:
    script = str(Path(server.remote_script).parent / "scene_split.py")
    sdir   = str(Path(server.remote_script).parent)
    return (f"PYTHONPATH={sdir}:$PYTHONPATH "
            f"{python} {script} "
            f"{input_dir} {clips_dir} "
            f"--workers {workers} "
            f"--trim-shots {trim_shots} "
            f"--log-file {log_path} "
            f"--queue-dir {queue_dir} "
            f"--worker-id {server.name} "
            f"--pid-file {pid_file}")

def build_cmd_stage3(server:Server, python:str, clips_dir:str, clean_dir:str,
                     queue_dir:str, pid_file:str, workers:int, log_path:str,
                     vsr_dir:str="") -> str:
    script  = str(Path(server.remote_script).parent / "subtitle_remove.py")
    sdir    = str(Path(server.remote_script).parent)
    vsr     = vsr_dir or server.vsr_dir or "~/video-subtitle-remover"
    return (f"PYTHONPATH={sdir}:$PYTHONPATH "
            f"{python} {script} "
            f"{clips_dir} {clean_dir} "
            f"--vsr-dir {vsr} "
            f"--workers {workers} "
            f"--log-file {log_path} "
            f"--queue-dir {queue_dir} "
            f"--worker-id {server.name} "
            f"--pid-file {pid_file}")

def build_cmd_stage4(server:Server, python:str, clips_dir:str, output_dir:str,
                     queue_dir:str, pid_file:str, workers:int, log_path:str,
                     face_conf:float=0.35) -> str:
    script = str(Path(server.remote_script).parent / "shot_classify.py")
    sdir   = str(Path(server.remote_script).parent)
    return (f"PYTHONPATH={sdir}:$PYTHONPATH "
            f"{python} {script} "
            f"{clips_dir} {output_dir} "
            f"--workers {workers} "
            f"--face-conf {face_conf} "
            f"--log-file {log_path} "
            f"--queue-dir {queue_dir} "
            f"--worker-id {server.name} "
            f"--pid-file {pid_file}")

# backward compat alias
build_cmd = build_cmd_stage1

def tail_log(server:Server, log_stdout:str, stop_ev:threading.Event):
    if server.is_local:
        full = ["tail","-f",os.path.expanduser(log_stdout)]
    else:
        full = ["ssh"]+server.ssh_opts()+[f"{server.user}@{server.host}",
                                           f"tail -f {log_stdout}"]
    proc = None
    try:
        proc = subprocess.Popen(full, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL,
                                text=True)
        while not stop_ev.is_set():
            line = proc.stdout.readline()
            if line: console.print(f"  [dim][{server.name}][/dim] {line.rstrip()}")
            else: time.sleep(0.3)
    except Exception: pass
    finally:
        if proc:
            proc.terminate()
            try: proc.stdout.close()
            except Exception: pass
            try: proc.wait(timeout=5)
            except Exception: proc.kill()

def wait_done(server:Server, pid_file:str, poll:int=10):
    while True:
        try:
            r = server.run(f"[ -f {pid_file} ] && echo running || echo done", timeout=10)
            if "done" in r.stdout: return
        except Exception: pass
        time.sleep(poll)

def check_queue_access(server:Server, queue_dir:str,
                       queue_name:str="pipeline_queue") -> bool:
    """检测远端机器是否能看到队列 JSON 文件本身（而不只是父目录）"""
    queue_file = f"{queue_dir}/{queue_name}.json"
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

def _stage_pid_suffix(stage:str) -> str:
    return {"1":"","2":"_split","3":"_subtitle","4":"_classify"}.get(stage,"")

def _scan_stage_input(stage:str, input_dir:str, output_dir:str) -> list[str]:
    """根据阶段返回待处理文件列表"""
    if stage == "1":
        return scan_videos(input_dir)
    elif stage == "2":
        # Stage 2 输入：Stage 1 输出目录根层级的 .mp4（不递归进 clips/clean）
        return sorted(str(p) for p in Path(output_dir).glob("*.mp4"))
    elif stage == "3":
        # Stage 3 输入：clips 目录下所有 .mp4（递归）
        clips_dir = str(Path(output_dir) / "clips")
        return sorted(str(p) for p in Path(clips_dir).rglob("*.mp4"))
    elif stage == "4":
        # Stage 4 输入：clips 目录下所有 shot_*.mp4（递归，同 Stage 3 来源）
        clips_dir = str(Path(output_dir) / "clips")
        return sorted(str(p) for p in Path(clips_dir).rglob("shot_*.mp4"))
    return []

def _run_stage(stage:str, op_servers:list, py_paths:dict, args,
               pid_files_base:dict, queue_dir:str, stop_ev, deployed:bool=False):
    """启动单个阶段的 worker 并等待完成。返回是否有 job 在跑。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common.task_queue import TaskQueue

    suffix      = _stage_pid_suffix(stage)
    pid_files   = {n: f"~/pipeline_{n}{suffix}.pid" for n in pid_files_base}
    log_suffix  = {"":"","_split":"_split","_subtitle":"_subtitle","_classify":"_classify"}[suffix]

    output_dir  = args.output_dir
    clips_dir   = str(Path(output_dir) / "clips")
    clean_dir   = str(Path(output_dir) / "clean")
    queue_names = {"1":"pipeline_queue","2":"split_queue","3":"subtitle_queue","4":"classify_queue"}
    qname       = queue_names[stage]

    # 扫描本阶段输入文件
    src_files = _scan_stage_input(stage, args.input_dir or "", output_dir)
    if not src_files:
        console.print(f"[red]Stage {stage}：未找到输入文件！[/red]"); return False

    # 初始化队列
    q = TaskQueue(queue_dir=queue_dir, worker_id="dispatcher", queue_name=qname)
    pre = q.stats()
    q.init_queue(src_files=src_files, start_idx=1, force=args.force)
    s         = q.stats()
    total_q   = sum(s.values())
    done_cnt  = s.get("done", 0)
    pct       = f"{done_cnt/total_q*100:.1f}%" if total_q else "0%"
    resuming  = pre.get("done",0) > 0 or pre.get("error",0) > 0
    console.print(f"\n  输入文件数 : [bold]{len(src_files)}[/bold]")
    if resuming:
        reset_cnt = pre.get("claimed",0)
        console.print(
            f"  [yellow]续传模式[/yellow] : "
            f"[green]done={done_cnt}[/green] ({pct})  "
            f"[yellow]pending={s.get('pending',0)}[/yellow]  "
            f"[red]error={s.get('error',0)}[/red]"
            + (f"  (已重置 {reset_cnt} 个中断任务)" if reset_cnt else ""))
    else:
        console.print(f"  [green]全新开始[/green] : 共 {total_q} 个任务")

    if s.get("pending",0) == 0 and s.get("claimed",0) == 0:
        console.print(f"[green]Stage {stage} 队列无待处理任务，跳过。[/green]"); return False

    if args.dry_run: return False

    # 部署本阶段所需脚本（仅首次）
    if args.deploy and not deployed:
        console.print(f"\n[bold]部署脚本（Stage {stage}）...[/bold]")
        project_root = Path(__file__).resolve().parents[2]
        files_to_deploy = [
            "process_videos.py",
            "scene_split.py",
            "subtitle_remove.py",
            "shot_classify.py",
            "src/common/__init__.py",
            "src/common/task_queue.py",
            "src/workers/__init__.py",
            "src/workers/process_videos.py",
            "src/workers/scene_split.py",
            "src/workers/subtitle_remove.py",
            "src/workers/shot_classify.py",
        ]
        for sv in op_servers:
            remote_root = Path(sv.remote_script).parent
            for rel in files_to_deploy:
                local = str(project_root / rel)
                if not Path(local).exists(): continue
                dst = str(remote_root / rel)
                sv.upload(local, dst)
            log.info(f"[{sv.name}] 脚本部署 ✅")

    # 启动 worker
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    jobs = []

    # ── 构建各机器的启动参数 ──────────────────────────────────
    launch_targets = []   # (sv, cmd, log_path, log_stdout, pid_file)
    workers_n = args.workers or (2 if stage == "2" else 1 if stage in ("3","4") else 0)
    for sv in op_servers:
        log_path   = f"~/pipeline_{sv.name}{log_suffix}.log"
        log_stdout = log_path + ".stdout"
        pid_file   = pid_files[sv.name]

        worker_alive, worker_pid = is_worker_alive(sv, pid_file)
        if worker_alive:
            log.info(f"[{sv.name}] Stage {stage} Worker 已在运行 (PID={worker_pid})")
            jobs.append((sv, log_stdout, pid_file))
            threading.Thread(target=tail_log,
                             args=(sv, log_stdout, stop_ev), daemon=True).start()
            continue

        if not check_queue_access(sv, queue_dir, qname):
            console.print(
                f"\n  [bold red][{sv.name}] ⚠ 队列文件不可访问[/bold red]\n"
                f"  路径: {queue_dir}/{qname}.json\n"
                f"  请确认该机器已挂载共享存储，跳过此机器。")
            continue

        if stage == "1":
            cmd = build_cmd_stage1(sv, py_paths[sv.name], args.input_dir,
                                   output_dir, queue_dir, pid_file, workers_n, log_path)
        elif stage == "2":
            cmd = build_cmd_stage2(sv, py_paths[sv.name], output_dir,
                                   clips_dir, queue_dir, pid_file, workers_n, log_path,
                                   trim_shots=getattr(args,"trim_shots",10))
        elif stage == "4":
            cmd = build_cmd_stage4(sv, py_paths[sv.name], clips_dir,
                                   output_dir, queue_dir, pid_file, workers_n, log_path,
                                   face_conf=getattr(args,"face_conf",0.35))
        else:
            vsr = getattr(args,"vsr_dir","") or sv.vsr_dir or "~/video-subtitle-remover"
            cmd = build_cmd_stage3(sv, py_paths[sv.name], clips_dir,
                                   clean_dir, queue_dir, pid_file, workers_n, log_path,
                                   vsr_dir=vsr)

        console.print(f"\n[bold][{sv.name}][/bold] Stage {stage}:\n  [dim]{cmd}[/dim]")
        launch_targets.append((sv, cmd, log_path, log_stdout, pid_file))

    # ── 并行启动所有机器，再并行等待 PID ─────────────────────
    launched: dict[str, tuple] = {}   # name → (sv, log_stdout, pid_file)

    def _launch_one(sv, cmd, log_path, log_stdout, pid_file):
        if not sv.launch_bg(cmd, log_stdout):
            return
        log.info(f"[{sv.name}] 等待 PID 文件（最多 30s）...")
        if wait_for_pid(sv, pid_file, timeout=30):
            log.info(f"[{sv.name}] Worker 已确认启动 ✅")
            launched[sv.name] = (sv, log_stdout, pid_file)
        else:
            # 本地进程可能已快速完成（无 pending 任务）：检查进程是否已退出
            already_done = False
            if sv.is_local:
                log_p = os.path.expanduser(log_stdout)
                if os.path.exists(log_p):
                    content = Path(log_p).read_text(errors="replace")
                    if "完成" in content or "done=0" in content or "✅=0" in content:
                        already_done = True
            if already_done:
                log.info(f"[{sv.name}] Worker 已处理完毕（无待处理任务）⏭")
                launched[sv.name] = (sv, log_stdout, pid_file)
            else:
                console.print(
                    f"  [bold red][{sv.name}] ❌ 30s 内未见 PID 文件\n"
                    f"  检查日志: {log_stdout}[/bold red]")

    threads = [threading.Thread(target=_launch_one, args=t, daemon=True)
               for t in launch_targets]
    for t in threads: t.start()
    for t in threads: t.join()

    for sv_name, (sv, log_stdout, pid_file) in launched.items():
        jobs.append((sv, log_stdout, pid_file))
        threading.Thread(target=tail_log,
                         args=(sv, log_stdout, stop_ev), daemon=True).start()

    if not jobs:
        console.print(f"[red]Stage {stage}：没有服务器启动成功！[/red]"); return False

    # 等待完成
    sv_names = ",".join(s.name for s, *_ in jobs)
    console.print(
        f"\n[bold]Stage {stage} 等待完成...[/bold]  (每 30s 刷新进度)\n"
        f"  Ctrl+C → 主控退出（worker 继续）  |  停止 → --stop")
    wts = []
    for sv, _, pf in jobs:
        t = threading.Thread(
            target=lambda s=sv, p=pf: (wait_done(s, p), log.info(f"[{s.name}] ✅ 完成")),
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
            f"  [dim]{time.strftime('%H:%M:%S')}[/dim]  Stage {stage}  "
            f"进度 [bold]{pct2}[/bold]  "
            f"[green]done={done2}[/green]  "
            f"[yellow]pending={s2.get('pending',0)}[/yellow]  "
            f"[blue]claimed={s2.get('claimed',0)}[/blue]  "
            f"[red]error={s2.get('error',0)}[/red]")

    stop_ev.clear()  # 重置事件，为下一阶段复用
    return True


def main():
    parser = argparse.ArgumentParser(description="分布式调度器 v6.0")
    parser.add_argument("--input-dir");  parser.add_argument("--output-dir")
    parser.add_argument("--servers", default="configs/servers.yaml",
                        help="服务器配置文件（默认 configs/servers.yaml）")
    parser.add_argument("--deploy",      action="store_true", help="部署脚本到各机器")
    parser.add_argument("--git-pull",    action="store_true", help="部署前 git pull")
    parser.add_argument("--force",       action="store_true", help="重置队列（全部重跑）")
    parser.add_argument("--workers",     type=int, default=0)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--check",       action="store_true")
    parser.add_argument("--stop",        action="store_true", help="终止服务器（可配合 --target）")
    parser.add_argument("--status",      action="store_true", help="查看各服务器运行状态和队列进度")
    parser.add_argument("--target",      type=str, default="",
                        help="指定操作的机器名，逗号分隔。例: --target A6000 或 --target A6000,A8000")
    parser.add_argument("--stage",       type=str, default="1",
                        choices=["1","2","3","4","all"],
                        help="处理阶段: 1=转码, 2=镜头切分, 3=字幕去除(可选), 4=镜头分类, all=默认跑 1+2+4")
    parser.add_argument("--trim-shots",  type=int, default=10,
                        help="Stage 2：首尾各去掉的镜头数（默认 10）")
    parser.add_argument("--vsr-dir",     type=str, default="",
                        help="Stage 3：video-subtitle-remover 路径，覆盖 servers.yaml 里的 vsr_dir")
    parser.add_argument("--face-conf",   type=float, default=0.35,
                        help="Stage 4：YOLO 检测置信度阈值（默认 0.35）")
    args = parser.parse_args()

    servers = load_servers(args.servers)

    # ── 连通性 ──
    console.rule("[bold cyan]🎬  分布式调度器 v6.0[/bold cyan]")
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

    # ── --stop：尝试停止所有阶段的 worker ──
    if args.stop:
        all_pid_files = {}
        for s in op_servers:
            for sfx in ("", "_split", "_subtitle", "_classify"):
                key = f"{s.name}{sfx}"
                all_pid_files[key] = f"~/pipeline_{s.name}{sfx}.pid"
        # stop_all 需要 Server 列表和 pid_files 字典
        # 对每个后缀单独广播 kill
        for sfx, label in (("","Stage1"), ("_split","Stage2"), ("_subtitle","Stage3"), ("_classify","Stage4")):
            pf = {s.name: f"~/pipeline_{s.name}{sfx}.pid" for s in op_servers}
            # 检查是否有该阶段的 PID 文件存在再 kill（跳过不存在的）
            targets = []
            for sv in op_servers:
                r = sv.run(f"[ -f {pf[sv.name]} ] && echo yes || echo no", timeout=5)
                if "yes" in r.stdout: targets.append(sv)
            if targets:
                console.print(f"\n[bold]{label}[/bold]")
                stop_all(targets, pf)
        return

    if args.status:
        qd = str(Path(args.output_dir) / ".queue") if args.output_dir else ""
        status_all(alive, pid_files, qd); return   # status 始终显示所有机器
    if args.check:    return
    if not alive:     console.print("[red]没有可用服务器！[/red]"); sys.exit(1)

    # Stage 1 需要 input_dir；Stage 2/3 只需 output_dir
    if args.stage in ("1","all") and not args.input_dir:
        console.print("[red]Stage 1 需要 --input-dir[/red]"); sys.exit(1)
    if not args.output_dir:
        console.print("[red]需要 --output-dir[/red]"); sys.exit(1)

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

    # ── 服务器表 ──
    tbl = Table(title="参与服务器", box=None)
    tbl.add_column("服务器",style="cyan"); tbl.add_column("GPU",style="yellow")
    tbl.add_column("权重",style="dim");   tbl.add_column("VSR路径",style="dim")
    for sv in op_servers:
        tbl.add_row(sv.name, str(sv.gpus), str(sv.weight),
                    sv.vsr_dir or getattr(args,"vsr_dir","") or "-")
    console.print(tbl)

    sys.path.insert(0, str(Path(__file__).parent))
    queue_dir = str(Path(args.output_dir) / ".queue")

    # ── Ctrl+C：主控退出，worker 继续 ──
    stop_ev = threading.Event()
    sv_names = ",".join(s.name for s in op_servers)
    def _sigint(sig, frame):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        stop_ev.set()
        _restore_terminal()
        console.print(
            "\n[yellow]⚠  主控已退出，worker 继续运行。[/yellow]\n"
            f"  查看进度  →  --status --output-dir <路径>\n"
            f"  停止全部  →  --stop\n"
            f"  停止单台  →  --stop --target [bold]{sv_names.split(',')[0]}[/bold]\n"
            f"  继续监控  →  重新运行原命令（已运行的 worker 不会重复启动）")
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    # ── 按阶段依次执行 ──
    stages = ["1","2","4"] if args.stage == "all" else [args.stage]
    deployed = False
    any_started = False
    for stage in stages:
        stage_labels = {"1":"转码","2":"镜头切分","3":"字幕去除","4":"镜头分类"}
        console.rule(f"[bold cyan]Stage {stage}：{stage_labels[stage]}[/bold cyan]")
        ok = _run_stage(stage, op_servers, py_paths, args,
                        pid_files, queue_dir, stop_ev, deployed=deployed)
        if ok:
            any_started = True
        deployed = True  # 后续阶段无需重复部署

    if any_started:
        console.rule("[bold green]所有阶段完成 🎉[/bold green]")

if __name__ == "__main__":
    main()