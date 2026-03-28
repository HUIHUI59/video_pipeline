#!/usr/bin/env python3
"""
subtitle_remove.py  v1.0
════════════════════════════════════════════════════════════════
Stage 3：字幕去除
  - 对每个 clip 调用 video-subtitle-remover (VSR) 去除硬字幕
  - 保持 clips 的目录层级：clean_root/{movie_stem}/shot_NNN.mp4
  - 已存在且非空的输出文件自动跳过（断点续传）
  - 支持分布式队列模式（--queue-dir）和本地模式

VSR: https://github.com/YaoFANGUK/video-subtitle-remover
安装：git clone <VSR仓库> ~/video-subtitle-remover && pip install -r requirements.txt
════════════════════════════════════════════════════════════════
"""

import os, sys, time, signal, socket, atexit, shutil, tempfile
import logging, argparse, threading, subprocess
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeRemainingColumn, TaskProgressColumn,
)
from rich.logging import RichHandler

try:
    os.setsid()
except OSError:
    pass

# ══════════════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".m4v", ".ts", ".mts", ".m2ts", ".webm",
}

DEFAULT_WORKERS    = 1   # VSR 是 GPU 密集型，默认单进程
HEARTBEAT_INTERVAL = 60

# ══════════════════════════════════════════════════════════════
# 日志
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)
log     = logging.getLogger("subtitle_remove")
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
class SubResult:
    src_path:   str
    dst_path:   str   = ""
    skip:       bool  = False
    success:    bool  = False
    error:      str   = ""
    duration_s: float = 0.0

# ══════════════════════════════════════════════════════════════
# 单任务：字幕去除
# ══════════════════════════════════════════════════════════════

def _load_vsr(vsr_dir: str):
    """将 VSR 路径插入 sys.path 并返回 SubtitleRemover 类（仅导入一次）。"""
    vsr_path = str(Path(os.path.expanduser(vsr_dir)))
    if vsr_path not in sys.path:
        sys.path.insert(0, vsr_path)
    from backend.main import SubtitleRemover  # noqa: PLC0415
    return SubtitleRemover


def _patch_vsr_config():
    """
    修正 VSR 运行时配置。
    必须在 SubtitleRemover.__init__() 之后调用（init 会 importlib.reload config，
    会重置文件中的原始值），run() 之前调用（run 读取 config 属性时使用修正值）。

    关键修正：
    - STTN_SKIP_DETECTION=False：启用字幕区域检测，避免对整帧画面做 inpaint
      （默认 True + 未指定 sub_area → 对全屏做 inpaint → 误删人脸/眼睛）
    """
    try:
        import backend.config as vsr_cfg  # noqa: PLC0415
        vsr_cfg.STTN_SKIP_DETECTION = False
    except Exception as e:
        log.warning(f"VSR config patch 失败（影响检测精度）: {e}")


def remove_one(src: str, clean_root: str, vsr_dir: str,
               conda_env: str = "vsr",
               queue=None) -> SubResult:
    res = SubResult(src_path=src)
    t0  = time.time()

    # 保持两级目录结构：clean_root/{movie_stem}/shot_NNN.mp4
    movie_stem = Path(src).parent.name
    dst_dir    = Path(clean_root) / movie_stem
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst        = str(dst_dir / Path(src).name)
    res.dst_path = dst

    # 已存在且非空则跳过
    if Path(dst).exists() and Path(dst).stat().st_size > 1024:
        res.skip = res.success = True
        if queue: queue.mark_done(src, dst)
        return res

    # 验证 VSR 路径
    vsr_backend = Path(os.path.expanduser(vsr_dir)) / "backend" / "main.py"
    if not vsr_backend.exists():
        res.error = (f"VSR backend/main.py 不存在: {vsr_backend}\n"
                     f"请先 clone VSR 仓库到 {vsr_dir}")
        if queue: queue.mark_failed(src, res.error)
        return res

    # VSR 的 SubtitleRemover 总是将结果写到与输入相同的目录下，
    # 文件名为 {stem}_no_sub.mp4。使用临时目录中转，再移动到目标路径。
    tmp_dir = None
    try:
        SubtitleRemover = _load_vsr(vsr_dir)

        tmp_dir = tempfile.mkdtemp(prefix="vsr_")
        tmp_src = str(Path(tmp_dir) / Path(src).name)
        # 建立符号链接（避免大文件复制）
        os.symlink(os.path.abspath(src), tmp_src)

        vsr_output = str(Path(tmp_dir) / f"{Path(src).stem}_no_sub.mp4")

        remover = SubtitleRemover(tmp_src, gui_mode=False)
        _patch_vsr_config()   # 必须在 init 之后、run 之前
        remover.run()

        if not Path(vsr_output).exists():
            res.error = "VSR 未生成输出文件"
            if queue: queue.mark_failed(src, res.error)
            return res

        shutil.move(vsr_output, dst)

    except Exception as e:
        res.error = str(e)[:400]
        if queue: queue.mark_failed(src, res.error)
        return res
    finally:
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    res.success    = True
    res.duration_s = time.time() - t0
    if queue: queue.mark_done(src, dst)
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

def run_queue(queue, clean_root: str, vsr_dir: str, _conda_env: str,
              workers: int, stop_ev):
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
        pid = prog.add_task(f"字幕去除[{queue.worker_id}]",
                            total=total_now, extra="")
        prog.update(pid, completed=stats.get("done", 0))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending_futs: dict[concurrent.futures.Future, str] = {}

            def submit_next():
                if stop_ev.is_set(): return False
                item = queue.claim_next()
                if item is None: return False
                src, _, idx = item
                fut = pool.submit(remove_one, src, clean_root,
                                  vsr_dir, "vsr", queue)
                pending_futs[fut] = src
                with hb_lock: current[fut] = src
                return True

            submit_next()  # 只认领 1 个，后续完成一个再认领一个，保证多机公平竞争

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
                            res = SubResult(src_path=src, error=str(e))
                            queue.mark_failed(src, str(e))
                        results.append(res)
                        icon = "⏭" if res.skip else ("✅" if res.success else "❌")
                        log.info(
                            f"{icon} {Path(res.src_path).name}"
                            + (f" → {Path(res.dst_path).name}" if res.dst_path else "")
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
                    # 暂无任务但其他 worker 还在跑；等待可能的超时重分配
                    time.sleep(8)
                    submit_next()

    return results

# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="字幕去除 Pipeline Stage 3 v1.0")
    parser.add_argument("input_dir",  help="clips 目录（Stage 2 输出）")
    parser.add_argument("output_dir", help="clean 输出根目录")
    parser.add_argument("--vsr-dir",   required=True,
                        help="video-subtitle-remover 的 clone 路径，如 ~/video-subtitle-remover")
    parser.add_argument("--workers",   type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--log-file",  type=str, default="subtitle_remove.log")
    parser.add_argument("--pid-file",  type=str, default="")
    parser.add_argument("--queue-dir", type=str, default="")
    parser.add_argument("--worker-id", type=str, default="")
    args = parser.parse_args()

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
                          worker_id=wid, queue_name="subtitle_queue")
        console.rule(f"[bold cyan]🎬  字幕去除队列模式  [{wid}][/bold cyan]")
        console.print(f"  输入 : {args.input_dir}")
        console.print(f"  输出 : {args.output_dir}")
        console.print(f"  VSR  : {args.vsr_dir}")
        s = queue.stats()
        console.print(f"  状态 : pending={s.get('pending',0)}  "
                      f"claimed={s.get('claimed',0)}  done={s.get('done',0)}")
        if s.get("pending", 0) + s.get("claimed", 0) == 0:
            console.print("[green]队列无待处理任务，退出。[/green]")
        else:
            run_queue(queue, args.output_dir, args.vsr_dir,
                      "vsr", args.workers, stop_ev)

    # ══ 本地模式 ══════════════════════════════════════════════
    else:
        videos = scan_videos(args.input_dir)
        if not videos:
            console.print("[red]未找到视频文件！[/red]"); sys.exit(1)
        console.rule("[bold cyan]🎬  字幕去除本地模式  v1.0[/bold cyan]")
        console.print(f"  共 {len(videos)} 个 clip，VSR: {args.vsr_dir}")
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(remove_one, src, args.output_dir,
                                args.vsr_dir): src
                    for src in videos}
            for fut in concurrent.futures.as_completed(futs):
                if stop_ev.is_set():
                    for f in futs: f.cancel(); break
                res = fut.result(); results.append(res)
                icon = "⏭" if res.skip else ("✅" if res.success else "❌")
                log.info(
                    f"{icon} {Path(res.src_path).name}"
                    + (f" ERR:{res.error[:60]}" if res.error else ""))

        ok = sum(1 for r in results if r.success and not r.skip)
        sk = sum(1 for r in results if r.skip)
        er = sum(1 for r in results if not r.success and not r.skip)
        console.rule("[bold]字幕去除完成[/bold]")
        console.print(f"  ✅={ok}  ⏭={sk}  ❌={er}")
        if er:
            for r in results:
                if not r.success and not r.skip:
                    console.print(f"  {Path(r.src_path).name}: {r.error}")

    if args.pid_file:
        try: Path(os.path.expanduser(args.pid_file)).unlink(missing_ok=True)
        except Exception: pass


if __name__ == "__main__":
    main()
