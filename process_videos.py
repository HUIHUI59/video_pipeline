#!/usr/bin/env python3
"""
process_videos.py  v5.2
════════════════════════════════════════════════════════════════
修复：
  - os.setsid() 在 main() 最顶部调用（parse_args 之前），确保
    Python 进程本身成为进程组 leader（PGID == PID）
  - kill 方案：调度器用 kill -TERM/-KILL -- -$PGID 覆盖整个进程组
    （包括所有 ffmpeg 子进程），不依赖 session
  - ffmpeg 子进程不再用 start_new_session=True，统一在同一进程组里
    → Python 死 → ffmpeg 随之死（PGID kill 一次搞定）
  - 命名：原名前6字_uuid8.mp4
════════════════════════════════════════════════════════════════
"""

import os, sys, re, json, time, uuid, signal, socket, atexit
import logging, argparse, threading, subprocess
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeRemainingColumn, MofNCompleteColumn, TaskProgressColumn,
)
from rich.logging import RichHandler

# ══════════════════════════════════════════════════════════════
# ① 最顶部立即成为进程组 leader
#    必须在任何 subprocess/threading 之前调用
#    调度器 kill -- -$PGID 覆盖本进程 + 所有子进程
# ══════════════════════════════════════════════════════════════
try:
    os.setsid()
except OSError:
    pass   # 已经是进程组 leader 时忽略

# ══════════════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {
    ".mp4",".mkv",".avi",".mov",".wmv",".flv",
    ".m4v",".ts",".mts",".m2ts",".webm",".rmvb",
    ".rm",".mpeg",".mpg",".vob",".3gp",
}

TARGET_WIDTH        = 1920
TARGET_FPS          = 24
AUDIO_BITRATE       = "128k"
QP_NVENC            = 23
CRF_SW              = 23
PRESET_NVENC        = "hq"
PRESET_SW           = "fast"
BITRATE_TABLE       = {2160:8000, 1440:5000, 1080:3500, 720:2000, 0:1200}
BITRATE_MIN         = 800
BITRATE_MAX         = 8000
MAX_SIZE_MB         = 2000
DEFAULT_WORKERS_GPU = 3
DEFAULT_WORKERS_CPU = 2
STATE_FILE          = ".pipeline_state.json"
HEARTBEAT_INTERVAL  = 60   # 秒

# ══════════════════════════════════════════════════════════════
# 日志
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)
log     = logging.getLogger("pipeline")
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

# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class VideoMeta:
    src_path:str; width:int=0; height:int=0; fps:float=0.0
    duration:float=0.0; codec:str=""; size_mb:float=0.0; has_audio:bool=True

@dataclass
class JobResult:
    src_path:str; dst_path:str=""; success:bool=False
    skip:bool=False; error:str=""; duration_s:float=0.0

# ══════════════════════════════════════════════════════════════
# 命名：原名前6字_uuid8.mp4
# ══════════════════════════════════════════════════════════════

def canonical_name(src_path: str) -> str:
    stem  = Path(src_path).stem
    clean = re.sub(r"[^\w\u4e00-\u9fff]", "", stem, flags=re.UNICODE)
    prefix = clean[:6] if clean else "video"
    return f"{prefix}_{uuid.uuid4().hex[:8]}.mp4"

# ══════════════════════════════════════════════════════════════
# ffprobe
# ══════════════════════════════════════════════════════════════

def probe(path: str) -> Optional[VideoMeta]:
    try:
        out  = subprocess.check_output(
            ["ffprobe","-v","quiet","-print_format","json",
             "-show_streams","-show_format", path],
            stderr=subprocess.DEVNULL, timeout=30)
        info = json.loads(out)
    except Exception as e:
        log.warning(f"ffprobe 失败: {path} — {e}"); return None

    meta = VideoMeta(src_path=path)
    fmt  = info.get("format", {})
    meta.size_mb  = int(fmt.get("size", 0)) / 1e6
    meta.duration = float(fmt.get("duration", 0))
    for s in info.get("streams", []):
        ct = s.get("codec_type")
        if ct == "video" and meta.width == 0:
            meta.width  = s.get("width", 0)
            meta.height = s.get("height", 0)
            meta.codec  = s.get("codec_name", "")
            try:
                n, d = s.get("r_frame_rate","0/1").split("/")
                meta.fps = float(n)/float(d) if float(d) else 0.0
            except Exception: pass
        elif ct == "audio":
            meta.has_audio = True
    return meta

# ══════════════════════════════════════════════════════════════
# scale & 码率
# ══════════════════════════════════════════════════════════════

def build_vf(meta: VideoMeta) -> str:
    if meta.width > 0 and meta.height > 0:
        out_h = int(round(TARGET_WIDTH * meta.height / meta.width))
        out_h += out_h % 2   # 必须偶数
    else:
        out_h = 1080
    return (f"scale={TARGET_WIDTH}:{out_h}:flags=lanczos"
            f",fps={TARGET_FPS},format=yuv420p")

def calc_br(meta: VideoMeta) -> int:
    base = next((BITRATE_TABLE[h] for h in sorted(BITRATE_TABLE, reverse=True)
                 if meta.height >= h), BITRATE_TABLE[0])
    if meta.duration > 0:
        base = min(base, int(MAX_SIZE_MB * 8000 / meta.duration))
    return max(BITRATE_MIN, min(BITRATE_MAX, base))

# ══════════════════════════════════════════════════════════════
# ffmpeg 命令
# ══════════════════════════════════════════════════════════════

def build_cmd(src, dst, meta, gpu_id=0, nvenc=True) -> list[str]:
    vf = build_vf(meta); br = calc_br(meta)
    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error"]
    if nvenc:
        cmd += ["-hwaccel","cuda","-hwaccel_device",str(gpu_id),
                "-extra_hw_frames","4","-threads","0"]
    else:
        cmd += ["-threads","0"]
    cmd += ["-i", src]
    if nvenc:
        cmd += ["-vf",vf,"-c:v","h264_nvenc","-preset",PRESET_NVENC,
                "-qp",str(QP_NVENC),"-b:v",f"{br}k",
                "-maxrate",f"{int(br*1.5)}k","-bufsize",f"{br*2}k",
                "-profile:v","high","-level","4.1","-gpu",str(gpu_id)]
    else:
        cmd += ["-vf",vf,"-c:v","libx264","-preset",PRESET_SW,
                "-crf",str(CRF_SW),"-b:v",f"{br}k",
                "-maxrate",f"{int(br*1.5)}k","-bufsize",f"{br*2}k",
                "-profile:v","high","-level","4.1","-threads","0"]
    cmd += (["-c:a","aac","-b:a",AUDIO_BITRATE,"-ac","2"]
            if meta.has_audio else ["-an"])
    cmd += ["-movflags","+faststart", dst]
    return cmd

# ══════════════════════════════════════════════════════════════
# ② ffmpeg 进程管理
#    不用 start_new_session，让 ffmpeg 留在同一进程组
#    → kill -- -$PGID 可以一次覆盖 Python + 所有 ffmpeg
# ══════════════════════════════════════════════════════════════

_procs: list[subprocess.Popen] = []
_lock  = threading.Lock()

def _run(cmd: list[str], timeout=7200) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with _lock: _procs.append(proc)
    try:
        out, err = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(
            cmd, proc.returncode,
            out.decode(errors="replace"), err.decode(errors="replace"))
    except subprocess.TimeoutExpired:
        proc.kill(); raise
    finally:
        with _lock:
            if proc in _procs: _procs.remove(proc)

def _kill_all():
    """Python 收到 SIGTERM 时主动 kill 所有子 ffmpeg"""
    with _lock:
        for p in list(_procs):
            try: p.terminate()
            except Exception: pass
    time.sleep(1)
    with _lock:
        for p in list(_procs):
            try: p.kill()
            except Exception: pass
    _procs.clear()

# ══════════════════════════════════════════════════════════════
# 本地断点状态
# ══════════════════════════════════════════════════════════════

def load_state(root: Path) -> dict:
    sf = root / STATE_FILE
    try: return json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
    except Exception: return {}

def save_state(root: Path, state: dict):
    sf = root / STATE_FILE; tmp = sf.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(sf)

# ══════════════════════════════════════════════════════════════
# 单任务
# ══════════════════════════════════════════════════════════════

def process_one(src, dst, gpu_id, nvenc,
                state=None, state_lock=None, out_root=None,
                queue=None) -> JobResult:
    res = JobResult(src_path=src, dst_path=dst); t0 = time.time()

    # 本地断点检查
    if state is not None:
        done = state.get(src)
        if done and Path(done).exists() and Path(done).stat().st_size > 1024:
            res.skip = res.success = True; res.dst_path = done; return res

    meta = probe(src)
    if meta is None:
        res.error = "ffprobe 失败"
        if queue: queue.mark_failed(src, res.error)
        return res

    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    try:
        r = _run(build_cmd(src, dst, meta, gpu_id, nvenc))
        if r.returncode != 0 and nvenc:
            log.warning(f"[yellow]NVENC 失败，回退 CPU: {Path(src).name}[/yellow]")
            r = _run(build_cmd(src, dst, meta, gpu_id, False))
        if r.returncode != 0:
            res.error = r.stderr[-400:]
            if queue: queue.mark_failed(src, res.error)
            return res

        res.success = True
        if queue:
            queue.mark_done(src, dst)
        elif state is not None and state_lock and out_root:
            with state_lock: state[src] = dst; save_state(out_root, state)

    except subprocess.TimeoutExpired:
        res.error = "超时(>2h)"
        if queue: queue.mark_failed(src, res.error)
    except Exception as e:
        res.error = str(e)
        if queue: queue.mark_failed(src, res.error)

    res.duration_s = time.time() - t0
    return res

# ══════════════════════════════════════════════════════════════
# 队列模式主循环
# ══════════════════════════════════════════════════════════════

def run_queue(queue, out_root, gpu_ids, nvenc, workers, stop_ev):
    counts  = {"ok": 0, "skip": 0, "err": 0}   # 只记计数，不积累 result 对象
    current = {}   # future → src，用于心跳
    hb_lock = threading.Lock()

    def hb_loop():
        while not stop_ev.is_set():
            with hb_lock:
                for src in list(current.values()):
                    try: queue.heartbeat(src)
                    except Exception: pass
            time.sleep(HEARTBEAT_INTERVAL)

    threading.Thread(target=hb_loop, daemon=True).start()

    stats     = queue.stats()
    total_now = sum(stats.values())

    with Progress(SpinnerColumn(),
                  TextColumn("[progress.description]{task.description}"),
                  BarColumn(bar_width=28), TaskProgressColumn(),
                  TextColumn("{task.fields[extra]}"),
                  TimeRemainingColumn(),
                  console=console, refresh_per_second=4) as prog:
        pid = prog.add_task(f"队列[{queue.worker_id}]",
                            total=total_now, extra="")
        prog.update(pid, completed=stats.get("done", 0))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending_futs: dict[concurrent.futures.Future, str] = {}

            def submit_next():
                if stop_ev.is_set(): return False
                item = queue.claim_next()
                if item is None: return False
                src, _, idx = item
                dst    = str(out_root / canonical_name(src))
                gpu_id = gpu_ids[len(pending_futs) % len(gpu_ids)]
                fut    = pool.submit(process_one, src, dst, gpu_id, nvenc,
                                     queue=queue)
                pending_futs[fut] = src
                with hb_lock: current[fut] = src
                return True

            # 自适应初始认领量：任务充足时填满所有 worker 槽位，任务稀少时留给其他机器公平竞争
            # burst = min(workers, pending // workers)，确保 pending >= workers² 时才全速启动
            _pending = queue.pending_count()
            _burst   = max(1, min(workers, _pending // max(workers, 1)))
            for _ in range(_burst):
                if not submit_next(): break

            while not stop_ev.is_set():
                if pending_futs:
                    done_set, _ = concurrent.futures.wait(
                        pending_futs, timeout=5,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done_set:
                        src = pending_futs.pop(fut)
                        with hb_lock: current.pop(fut, None)
                        try: res = fut.result()
                        except Exception as e:
                            res = JobResult(src_path=src, error=str(e))
                            queue.mark_failed(src, str(e))
                        if res.skip: counts["skip"] += 1
                        elif res.success: counts["ok"] += 1
                        else: counts["err"] += 1
                        icon = "⏭" if res.skip else ("✅" if res.success else "❌")
                        log.info(f"{icon} {Path(res.src_path).name}"
                                 + (f" → {Path(res.dst_path).name}" if res.dst_path else "")
                                 + (f" [{res.duration_s:.0f}s]" if res.duration_s else "")
                                 + (f" ERR:{res.error[:60]}" if res.error else ""))
                        s2 = queue.stats()
                        prog.update(pid,
                                    completed=s2.get("done",0),
                                    extra=f"done={s2.get('done',0)} pending={s2.get('pending',0)}")
                        if not stop_ev.is_set(): submit_next()

                if not pending_futs:
                    if queue.is_all_done(): break
                    time.sleep(8)
                    submit_next()

    return counts

# ══════════════════════════════════════════════════════════════
# 扫描 & NVENC 检测
# ══════════════════════════════════════════════════════════════

def scan(root: str) -> list[str]:
    found = []
    for d,_,fns in os.walk(root):
        for fn in sorted(fns):
            if Path(fn).suffix.lower() in VIDEO_EXTENSIONS:
                found.append(os.path.join(d, fn))
    return found

def detect_nvenc(gpu_id=0) -> bool:
    try:
        r = subprocess.run(["ffmpeg","-hide_banner","-encoders"],
                           capture_output=True, text=True)
        if "h264_nvenc" not in r.stdout: return False
    except Exception: return False
    null = "NUL" if sys.platform=="win32" else "/dev/null"
    r = subprocess.run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-hwaccel","cuda","-hwaccel_device",str(gpu_id),
        "-extra_hw_frames","4",
        "-f","lavfi","-i","color=c=black:s=1920x1080:r=24",
        "-frames:v","10","-c:v","h264_nvenc",
        "-preset","hq","-qp","23","-gpu",str(gpu_id),
        "-f","null", null,
    ], capture_output=True, text=True, timeout=20)
    ok = r.returncode == 0
    log.info(f"NVENC {'✅' if ok else '❌'} (GPU {gpu_id})")
    return ok

def detect_gpu_count() -> int:
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                           capture_output=True, text=True)
        return len([l for l in r.stdout.strip().split("\n") if l.strip()])
    except Exception: return 0

# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="视频预处理 Pipeline v5.2")
    parser.add_argument("input_dir");  parser.add_argument("output_dir")
    parser.add_argument("--workers",   type=int, default=0)
    parser.add_argument("--gpu-ids",   type=str, default="auto")
    parser.add_argument("--no-nvenc",  action="store_true")
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--log-file",  type=str, default="pipeline.log")
    parser.add_argument("--file-list", type=str, default="")
    parser.add_argument("--pid-file",  type=str, default="")
    parser.add_argument("--queue-dir", type=str, default="")
    parser.add_argument("--worker-id", type=str, default="")
    args = parser.parse_args()

    # PID 文件（setsid 已在模块顶部完成）
    if args.pid_file:
        pp = Path(os.path.expanduser(args.pid_file))
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(str(os.getpid()))

    # 日志文件
    fh = logging.FileHandler(args.log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)

    # GPU
    gc      = detect_gpu_count()
    gpu_ids = (list(range(max(gc,1))) if args.gpu_ids=="auto"
               else [int(x) for x in args.gpu_ids.split(",")])
    nvenc   = (not args.no_nvenc) and detect_nvenc(gpu_id=gpu_ids[0])
    workers = args.workers or (DEFAULT_WORKERS_GPU if nvenc else DEFAULT_WORKERS_CPU)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 信号：SIGTERM/SIGINT → kill ffmpeg → 删 PID → 退出
    stop_ev = threading.Event()
    def _terminate(sig, frame):
        console.print(f"\n[yellow]⚠  信号{sig}，终止 ffmpeg 子进程...[/yellow]")
        stop_ev.set()
        _kill_all()
        if args.pid_file:
            try: Path(os.path.expanduser(args.pid_file)).unlink(missing_ok=True)
            except Exception: pass
        _restore_terminal()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _terminate)
    signal.signal(signal.SIGTERM, _terminate)

    # ══ 队列模式 ══════════════════════════════════════════════
    if args.queue_dir:
        sys.path.insert(0, str(Path(__file__).parent))
        from task_queue import TaskQueue
        wid   = args.worker_id or socket.gethostname()
        queue = TaskQueue(queue_dir=os.path.expanduser(args.queue_dir),
                          worker_id=wid)
        console.rule(f"[bold cyan]🎬  队列模式  [{wid}][/bold cyan]")
        mode = f"NVENC-{PRESET_NVENC}" if nvenc else f"libx264-{PRESET_SW}"
        console.print(f"  编码  : [yellow]{mode} × {workers}路[/yellow]")
        console.print(f"  队列  : [green]{args.queue_dir}[/green]")
        s = queue.stats()
        console.print(f"  状态  : pending={s.get('pending',0)}  "
                      f"claimed={s.get('claimed',0)}  done={s.get('done',0)}")
        if s.get("pending",0) + s.get("claimed",0) == 0:
            console.print("[green]队列无待处理任务，退出。[/green]"); return
        results = run_queue(queue, out_root, gpu_ids, nvenc, workers, stop_ev)

    # ══ 本地模式 ══════════════════════════════════════════════
    else:
        state = {} if args.force else load_state(out_root)
        state_lock = threading.Lock()
        if args.file_list:
            fl = Path(os.path.expanduser(args.file_list))
            videos = [l.strip() for l in fl.read_text(encoding="utf-8").splitlines()
                      if l.strip() and Path(l.strip()).exists()]
        else:
            videos = scan(args.input_dir)
        if not videos:
            console.print("[red]未找到视频文件！[/red]"); sys.exit(1)

        tasks   = [(src, str(out_root/canonical_name(src)),
                    gpu_ids[i%len(gpu_ids)], nvenc)
                   for i,src in enumerate(videos)]
        pending = sum(1 for src,*_ in tasks if src not in state)
        console.rule("[bold cyan]🎬  本地模式  v5.2[/bold cyan]")
        console.print(f"  总文件={len(videos)}  已完成={len(state)}  "
                      f"待处理=[bold]{pending}[/bold]")
        if args.dry_run:
            for i,(src,dst,g,_) in enumerate(tasks[:40],1):
                m = probe(src)
                console.print(f"  {i:3d}. GPU={g} {calc_br(m) if m else '?'}kbps "
                              f"{Path(src).name}")
            return
        results = []
        with Progress(SpinnerColumn(),TextColumn("[progress.description]{task.description}"),
                      BarColumn(bar_width=30),TaskProgressColumn(),
                      MofNCompleteColumn(),TimeRemainingColumn(),
                      console=console,refresh_per_second=4) as prog:
            pid2 = prog.add_task(f"转码×{workers}", total=len(tasks))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(process_one,src,dst,g,nv,
                                    state=state,state_lock=state_lock,
                                    out_root=out_root): src
                        for src,dst,g,nv in tasks}
                for fut in concurrent.futures.as_completed(futs):
                    if stop_ev.is_set():
                        for f in futs: f.cancel(); break
                    res = fut.result(); results.append(res)
                    icon = "⏭" if res.skip else ("✅" if res.success else "❌")
                    log.info(f"{icon} {Path(res.src_path).name}"
                             +(f" [{res.duration_s:.0f}s]" if res.duration_s else "")
                             +(f" ERR:{res.error[:60]}" if res.error else ""))
                    prog.advance(pid2)

    # 汇总
    ok = sum(1 for r in results if r.success and not r.skip)
    sk = sum(1 for r in results if r.skip)
    er = sum(1 for r in results if not r.success)
    console.rule("[bold]完成[/bold]")
    console.print(f"  ✅={ok}  ⏭={sk}  ❌={er}")
    if er:
        for r in results:
            if not r.success: console.print(f"  {Path(r.src_path).name}: {r.error}")

    # 清除 PID 文件
    if args.pid_file:
        try: Path(os.path.expanduser(args.pid_file)).unlink(missing_ok=True)
        except Exception: pass

if __name__ == "__main__":
    main()