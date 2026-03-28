#!/usr/bin/env python3
"""
scene_split.py  v1.0
════════════════════════════════════════════════════════════════
Stage 2：镜头检测 & 切分
  - 使用 PySceneDetect 按场景切分转码后的视频
  - 去掉首尾各 N 个镜头（默认 10）
  - 输出目录结构：clips_root/{video_stem}/shot_NNN.mp4
  - 支持分布式队列模式（--queue-dir）和本地模式

依赖：pip install scenedetect[opencv]
════════════════════════════════════════════════════════════════
"""

import os, sys, time, signal, socket, atexit
import logging, argparse, threading, subprocess
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeRemainingColumn, TaskProgressColumn,
)
from rich.logging import RichHandler

# ── 成为进程组 leader，确保 dispatcher kill -- -$PGID 能覆盖子进程 ──
try:
    os.setsid()
except OSError:
    pass

# ══════════════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".m4v", ".ts", ".mts", ".m2ts", ".webm", ".rmvb",
    ".rm", ".mpeg", ".mpg", ".vob", ".3gp",
}

DEFAULT_WORKERS    = 2
HEARTBEAT_INTERVAL = 60
DEFAULT_TRIM_SHOTS = 10

# ══════════════════════════════════════════════════════════════
# 日志
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)
log     = logging.getLogger("scene_split")
console = Console()

def _restore_terminal():
    try: console.show_cursor(True)
    except Exception: pass
    try:
        subprocess.run(["stty", "sane"], check=False, timeout=2,
                       stdin=subprocess.DEVNULL, capture_output=True)
    except Exception: pass

atexit.register(_restore_terminal)

# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class SplitResult:
    src_path:   str
    clips:      list = field(default_factory=list)
    skip:       bool = False
    success:    bool = False
    error:      str  = ""
    duration_s: float = 0.0

# ══════════════════════════════════════════════════════════════
# 单任务：镜头切分
# ══════════════════════════════════════════════════════════════

def split_one(src: str, clips_root: str,
              trim_shots: int = DEFAULT_TRIM_SHOTS,
              queue=None) -> SplitResult:
    res = SplitResult(src_path=src)
    t0  = time.time()

    try:
        from scenedetect import detect, ContentDetector, split_video_ffmpeg
    except ImportError:
        res.error = "scenedetect 未安装，请运行: pip install scenedetect[opencv]"
        if queue: queue.mark_failed(src, res.error)
        return res

    # 场景检测
    try:
        scenes = detect(src, ContentDetector())
    except Exception as e:
        res.error = f"场景检测失败: {e}"
        if queue: queue.mark_failed(src, res.error)
        return res

    if not scenes:
        res.error = "未检测到任何镜头"
        if queue: queue.mark_failed(src, res.error)
        return res

    # 镜头数足够则去掉首尾各 trim_shots，否则保留全部
    if len(scenes) > 2 * trim_shots:
        kept = scenes[trim_shots:-trim_shots]
    else:
        kept = scenes

    # 输出目录：clips_root/{video_stem}/
    stem    = Path(src).stem
    out_dir = Path(clips_root) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        split_video_ffmpeg(
            src, kept,
            output_dir=str(out_dir),
            output_file_template="shot_$SCENE_NUMBER.mp4",
        )
    except Exception as e:
        res.error = f"视频切分失败: {e}"
        if queue: queue.mark_failed(src, res.error)
        return res

    clips = sorted(out_dir.glob("shot_*.mp4"))
    res.clips      = [str(c) for c in clips]
    res.success    = True
    res.duration_s = time.time() - t0

    if queue:
        queue.mark_done(src, str(out_dir))
    return res

# ══════════════════════════════════════════════════════════════
# 扫描
# ══════════════════════════════════════════════════════════════

def scan_videos(root: str) -> list[str]:
    found = []
    for d, _, fns in os.walk(root):
        for fn in sorted(fns):
            if Path(fn).suffix.lower() in VIDEO_EXTENSIONS:
                found.append(os.path.join(d, fn))
    return found

# ══════════════════════════════════════════════════════════════
# 队列模式主循环
# ══════════════════════════════════════════════════════════════

def run_queue(queue, clips_root: str, workers: int,
              trim_shots: int, stop_ev):
    results = []
    current = {}
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
        pid = prog.add_task(f"镜头切分[{queue.worker_id}]",
                            total=total_now, extra="")
        prog.update(pid, completed=stats.get("done", 0))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending_futs: dict[concurrent.futures.Future, str] = {}

            def submit_next():
                if stop_ev.is_set(): return False
                item = queue.claim_next()
                if item is None: return False
                src, _, idx = item
                fut = pool.submit(split_one, src, clips_root, trim_shots, queue)
                pending_futs[fut] = src
                with hb_lock: current[fut] = src
                return True

            for _ in range(workers): submit_next()

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
                            res = SplitResult(src_path=src, error=str(e))
                            queue.mark_failed(src, str(e))
                        results.append(res)
                        icon = "⏭" if res.skip else ("✅" if res.success else "❌")
                        log.info(
                            f"{icon} {Path(res.src_path).name}"
                            + (f" → {len(res.clips)} 个镜头" if res.clips else "")
                            + (f" [{res.duration_s:.0f}s]" if res.duration_s else "")
                            + (f" ERR:{res.error[:60]}" if res.error else ""))
                        s2 = queue.stats()
                        prog.update(pid,
                                    completed=s2.get("done", 0),
                                    extra=f"done={s2.get('done',0)} "
                                          f"pending={s2.get('pending',0)}")
                        if not stop_ev.is_set(): submit_next()

                if not pending_futs:
                    if queue.is_all_done(): break
                    time.sleep(8)
                    submit_next()

    return results

# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="场景切分 Pipeline Stage 2 v1.0")
    parser.add_argument("input_dir",  help="转码后视频目录（Stage 1 输出）")
    parser.add_argument("output_dir", help="clips 输出根目录")
    parser.add_argument("--workers",    type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--trim-shots", type=int, default=DEFAULT_TRIM_SHOTS,
                        help=f"首尾各去掉的镜头数（默认 {DEFAULT_TRIM_SHOTS}）")
    parser.add_argument("--log-file",   type=str, default="scene_split.log")
    parser.add_argument("--pid-file",   type=str, default="")
    parser.add_argument("--queue-dir",  type=str, default="")
    parser.add_argument("--worker-id",  type=str, default="")
    args = parser.parse_args()

    # PID 文件（setsid 已在模块顶部完成）
    if args.pid_file:
        pp = Path(os.path.expanduser(args.pid_file))
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(str(os.getpid()))

    fh = logging.FileHandler(args.log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)

    stop_ev = threading.Event()

    def _terminate(sig, frame):
        console.print(f"\n[yellow]⚠  信号 {sig}，正在退出...[/yellow]")
        stop_ev.set()
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
                          worker_id=wid, queue_name="split_queue")
        console.rule(f"[bold cyan]✂  镜头切分队列模式  [{wid}][/bold cyan]")
        console.print(f"  输入 : {args.input_dir}")
        console.print(f"  输出 : {args.output_dir}")
        console.print(f"  截断 : 首尾各 {args.trim_shots} 个镜头")
        s = queue.stats()
        console.print(f"  状态 : pending={s.get('pending',0)}  "
                      f"claimed={s.get('claimed',0)}  done={s.get('done',0)}")
        if s.get("pending", 0) + s.get("claimed", 0) == 0:
            console.print("[green]队列无待处理任务，退出。[/green]")
        else:
            run_queue(queue, args.output_dir, args.workers,
                      args.trim_shots, stop_ev)

    # ══ 本地模式 ══════════════════════════════════════════════
    else:
        videos = scan_videos(args.input_dir)
        if not videos:
            console.print("[red]未找到视频文件！[/red]"); sys.exit(1)
        console.rule("[bold cyan]✂  镜头切分本地模式  v1.0[/bold cyan]")
        console.print(f"  共 {len(videos)} 个视频，首尾各去掉 {args.trim_shots} 个镜头")
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(split_one, src, args.output_dir,
                                args.trim_shots): src for src in videos}
            for fut in concurrent.futures.as_completed(futs):
                if stop_ev.is_set():
                    for f in futs: f.cancel(); break
                res = fut.result(); results.append(res)
                icon = "⏭" if res.skip else ("✅" if res.success else "❌")
                log.info(
                    f"{icon} {Path(res.src_path).name}"
                    + (f" → {len(res.clips)} 个镜头" if res.clips else "")
                    + (f" ERR:{res.error[:60]}" if res.error else ""))

        ok = sum(1 for r in results if r.success)
        er = sum(1 for r in results if not r.success and not r.skip)
        console.rule("[bold]切分完成[/bold]")
        console.print(f"  ✅={ok}  ❌={er}")
        if er:
            for r in results:
                if not r.success and not r.skip:
                    console.print(f"  {Path(r.src_path).name}: {r.error}")

    if args.pid_file:
        try: Path(os.path.expanduser(args.pid_file)).unlink(missing_ok=True)
        except Exception: pass


if __name__ == "__main__":
    main()
