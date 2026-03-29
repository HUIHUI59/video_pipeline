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

import os, sys, time, signal, socket, atexit, shutil, tempfile, textwrap
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

# PaddlePaddle 在同一 Python 进程内无法二次初始化 libpaddle.so：
# 第一个 SubtitleRemover 结束后，paddle 的 C++ 侧状态已不可恢复，
# 第二次 import paddle 会抛出
# "Can not import paddle core while this file exists: .../libpaddle.so"。
# 解决方案：每个 clip 单独起一个子进程，子进程退出后 paddle 状态随之消亡。
#
# 重要：不通过 sys.argv 传参，而是将值直接嵌入脚本字符串。
# VSR 的部分依赖（如 backend/tools/train_sttn.py）在模块作用域调用
# argparse.parse_args()，若 sys.argv 含额外参数会以 exit(2) 退出。
# 嵌入值 + sys.argv=['vsr_worker'] 可彻底规避此问题。
def _build_worker_script(vsr_path: str, src: str, dst: str) -> str:
    return textwrap.dedent(f"""\
        import sys, os, shutil, tempfile
        from pathlib import Path

        # 清空 sys.argv，防止 VSR 依赖中模块级 argparse.parse_args() 报错退出
        sys.argv = ['vsr_worker']
        sys.path.insert(0, {repr(vsr_path)})

        from backend.main import SubtitleRemover
        import backend.config as vsr_cfg

        src = {repr(src)}
        dst = {repr(dst)}

        tmp_dir = tempfile.mkdtemp(prefix="vsr_")
        try:
            tmp_src = str(Path(tmp_dir) / Path(src).name)
            os.symlink(os.path.abspath(src), tmp_src)
            vsr_output = str(Path(tmp_dir) / (Path(src).stem + "_no_sub.mp4"))

            remover = SubtitleRemover(tmp_src, gui_mode=False)
            # patch AFTER __init__（init 会 importlib.reload config 重置原始值）
            vsr_cfg.STTN_SKIP_DETECTION = False
            remover.run()

            if not Path(vsr_output).exists():
                print("VSR_ERROR: no output file", flush=True)
                sys.exit(1)
            shutil.move(vsr_output, dst)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    """)


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

    # 每个 clip 在独立子进程中运行，避免 paddle libpaddle.so 无法二次初始化
    vsr_path = str(Path(os.path.expanduser(vsr_dir)))
    worker_code = _build_worker_script(vsr_path, src, dst)

    # 确保子进程能找到 cuDNN 等 NVIDIA 库
    # 优先级：~/cudnn8/lib（cuDNN 8，供 paddle cu118）> site-packages nvidia/*/lib（cuDNN 9，供 torch）
    # RTX4090 使用 cu126，paddle 也需要 cuDNN 9，~/cudnn8/lib 不存在时自动跳过
    env = os.environ.copy()
    lib_dirs = []

    # 1) cuDNN 8 目录（A6000/A8000 cu118 专用，cu126 机器无此目录则跳过）
    cudnn8_dir = Path.home() / "cudnn8" / "lib"
    if cudnn8_dir.is_dir():
        lib_dirs.append(str(cudnn8_dir))

    # 2) pip 安装的 nvidia-* 包的 lib 目录（包含 cuDNN 9 / cuBLAS / cuSolver 等）
    try:
        import site as _site
        for sp in _site.getsitepackages():
            nvidia_base = Path(sp) / "nvidia"
            if nvidia_base.is_dir():
                for sub in nvidia_base.iterdir():
                    lib_dir = sub / "lib"
                    if lib_dir.is_dir():
                        lib_dirs.append(str(lib_dir))
    except Exception:
        pass

    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", worker_code],
            timeout=3600,
            env=env,
        )
        if proc.returncode != 0:
            res.error = f"VSR 子进程退出码 {proc.returncode}"
            if queue: queue.mark_failed(src, res.error)
            return res
    except subprocess.TimeoutExpired:
        res.error = "VSR 处理超时（>1h）"
        if queue: queue.mark_failed(src, res.error)
        return res
    except Exception as e:
        res.error = str(e)[:400]
        if queue: queue.mark_failed(src, res.error)
        return res

    if not Path(dst).exists() or Path(dst).stat().st_size < 1024:
        res.error = "VSR 子进程未生成有效输出文件"
        if queue: queue.mark_failed(src, res.error)
        return res

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
    """
    Stage 3 队列主循环。
    故意使用顺序（单任务）模式而非 ThreadPoolExecutor：
    PaddlePaddle 在同一进程中连续创建多个 SubtitleRemover 实例时，
    第二次 paddle 初始化会与第一次的 libpaddle.so 残留状态冲突并报错。
    顺序处理确保每个任务完全结束、paddle 资源释放后再处理下一个。
    """
    counts    = {"ok": 0, "skip": 0, "err": 0}
    cur_src   = [None]   # 供心跳线程读取当前任务路径

    def hb_loop():
        while not stop_ev.is_set():
            src = cur_src[0]
            if src:
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

        while not stop_ev.is_set():
            item = queue.claim_next()
            if item is None:
                if queue.is_all_done(): break
                time.sleep(8)
                continue

            src, _, _ = item
            cur_src[0] = src

            try:
                res = remove_one(src, clean_root, vsr_dir, "vsr", queue)
            except Exception as e:
                res = SubResult(src_path=src, error=str(e))
                queue.mark_failed(src, str(e))
            finally:
                cur_src[0] = None

            if res.skip: counts["skip"] += 1
            elif res.success: counts["ok"] += 1
            else: counts["err"] += 1
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

    return counts

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
