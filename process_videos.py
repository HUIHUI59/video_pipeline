#!/usr/bin/env python3
"""
Video Preprocessing Pipeline  v5.0
════════════════════════════════════════════════════════════════
修复历史：
  v5: ⑤ 命名改为"原名前6字_uuid8.mp4"（去掉序号）
      ⑥ kill 机制：用进程组 os.setsid() + kill(-pgid) 确保
         ffmpeg 子进程一定随 Python 进程一起死亡
      ⑦ 队列模式兼容本地模式，逻辑统一
════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import json
import time
import uuid
import signal
import socket
import logging
import argparse
import threading
import subprocess
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeRemainingColumn, MofNCompleteColumn,
    TaskProgressColumn,
)
from rich.logging import RichHandler

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".m4v", ".ts", ".mts", ".m2ts", ".webm", ".rmvb",
    ".rm", ".mpeg", ".mpg", ".vob", ".3gp",
}

TARGET_WIDTH  = 1920
TARGET_FPS    = 24
AUDIO_BITRATE = "128k"
QP_NVENC      = 23
CRF_SW        = 23
PRESET_NVENC  = "hq"
PRESET_SW     = "fast"

BITRATE_TABLE = {
    2160: 8000,
    1440: 5000,
    1080: 3500,
    720:  2000,
    0:    1200,
}
BITRATE_MIN_KBPS  = 800
BITRATE_MAX_KBPS  = 8000
MAX_SIZE_MB       = 2000

# NVENC 模式下默认 3 路并发（压榨 CPU 解码余量）
DEFAULT_WORKERS_NVENC = 3
DEFAULT_WORKERS_SW    = 2

STATE_FILENAME     = ".pipeline_state.json"
HEARTBEAT_INTERVAL = 60   # 秒


# ══════════════════════════════════════════════════════════════
# 日志
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)
log     = logging.getLogger("pipeline")
console = Console()


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class VideoMeta:
    src_path:  str
    width:     int   = 0
    height:    int   = 0
    fps:       float = 0.0
    duration:  float = 0.0
    codec:     str   = ""
    size_mb:   float = 0.0
    has_audio: bool  = True


@dataclass
class JobResult:
    src_path:   str
    dst_path:   str   = ""
    success:    bool  = False
    skip:       bool  = False
    error:      str   = ""
    duration_s: float = 0.0


# ══════════════════════════════════════════════════════════════
# ⑤ 命名：原名前6字_uuid8.mp4
# ══════════════════════════════════════════════════════════════

def canonical_name(src_path: str) -> str:
    """
    格式：{原始文件名前6个有效字符}_{uuid前8位}.mp4
    例：极品老妈_a3f2c1b4.mp4 / Broker_2022_9e8d7c6b.mp4
    - 只保留字母、数字、中文，去掉其余特殊符号
    - 不足6字符则用全部
    """
    stem = Path(src_path).stem
    # 保留字母、数字、中文，去掉点、括号、空格等
    clean = re.sub(r"[^\w\u4e00-\u9fff]", "", stem, flags=re.UNICODE)
    prefix = clean[:6] if clean else "video"
    uid    = uuid.uuid4().hex[:8]
    return f"{prefix}_{uid}.mp4"


# ══════════════════════════════════════════════════════════════
# ffprobe
# ══════════════════════════════════════════════════════════════

def probe(path: str) -> Optional[VideoMeta]:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path,
    ]
    try:
        out  = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30)
        info = json.loads(out)
    except Exception as e:
        log.warning(f"ffprobe 失败: {path} — {e}")
        return None

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
            fps_str = s.get("r_frame_rate", "0/1")
            try:
                n, d = fps_str.split("/")
                meta.fps = float(n) / float(d) if float(d) else 0.0
            except Exception:
                pass
        elif ct == "audio":
            meta.has_audio = True
    return meta


# ══════════════════════════════════════════════════════════════
# Scale & 码率
# ══════════════════════════════════════════════════════════════

def build_scale_filter(meta: VideoMeta) -> str:
    if meta.width > 0 and meta.height > 0:
        out_h = int(round(TARGET_WIDTH * meta.height / meta.width))
        out_h = out_h if out_h % 2 == 0 else out_h + 1
    else:
        out_h = 1080
    return (
        f"scale={TARGET_WIDTH}:{out_h}:flags=lanczos"
        f",fps={TARGET_FPS},format=yuv420p"
    )


def calc_bitrate(meta: VideoMeta) -> int:
    base = BITRATE_TABLE[0]
    for h in sorted(BITRATE_TABLE.keys(), reverse=True):
        if meta.height >= h:
            base = BITRATE_TABLE[h]
            break
    if meta.duration > 0:
        base = min(base, int(MAX_SIZE_MB * 8 * 1000 / meta.duration))
    return max(BITRATE_MIN_KBPS, min(BITRATE_MAX_KBPS, base))


# ══════════════════════════════════════════════════════════════
# ffmpeg 命令
# ══════════════════════════════════════════════════════════════

def build_ffmpeg_cmd(src, dst, meta, gpu_id=0, use_nvenc=True) -> list[str]:
    vf  = build_scale_filter(meta)
    br  = calc_bitrate(meta)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    if use_nvenc:
        cmd += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu_id),
                "-extra_hw_frames", "4", "-threads", "0"]
    else:
        cmd += ["-threads", "0"]

    cmd += ["-i", src]

    if use_nvenc:
        cmd += ["-vf", vf, "-c:v", "h264_nvenc",
                "-preset", PRESET_NVENC, "-qp", str(QP_NVENC),
                "-b:v", f"{br}k", "-maxrate", f"{int(br*1.5)}k",
                "-bufsize", f"{br*2}k",
                "-profile:v", "high", "-level", "4.1", "-gpu", str(gpu_id)]
    else:
        cmd += ["-vf", vf, "-c:v", "libx264",
                "-preset", PRESET_SW, "-crf", str(CRF_SW),
                "-b:v", f"{br}k", "-maxrate", f"{int(br*1.5)}k",
                "-bufsize", f"{br*2}k",
                "-profile:v", "high", "-level", "4.1", "-threads", "0"]

    cmd += (["-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "2"]
            if meta.has_audio else ["-an"])
    cmd += ["-movflags", "+faststart", dst]
    return cmd


# ══════════════════════════════════════════════════════════════
# ⑥ ffmpeg 进程管理：用进程组确保子进程随父进程一起死亡
# ══════════════════════════════════════════════════════════════

def run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess:
    """
    启动 ffmpeg，并将其放入独立进程组（os.setsid）。
    当 Python 进程收到 SIGTERM/SIGKILL 时，在 __del__ 或 signal handler
    里 kill 整个进程组，确保 ffmpeg 不会变成孤儿。
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # 让 ffmpeg 成为新进程组的组长，方便 killpg 一次性杀掉
        start_new_session=True,
    )
    _register_proc(proc)
    try:
        out, err = proc.communicate(timeout=7200)
        return subprocess.CompletedProcess(
            cmd, proc.returncode,
            out.decode(errors="replace"),
            err.decode(errors="replace"),
        )
    except subprocess.TimeoutExpired:
        _kill_proc(proc)
        raise
    finally:
        _unregister_proc(proc)


# 全局进程注册表（signal handler 用）
_active_procs: list[subprocess.Popen] = []
_procs_lock   = threading.Lock()


def _register_proc(proc: subprocess.Popen):
    with _procs_lock:
        _active_procs.append(proc)


def _unregister_proc(proc: subprocess.Popen):
    with _procs_lock:
        if proc in _active_procs:
            _active_procs.remove(proc)


def _kill_proc(proc: subprocess.Popen):
    """kill 进程组（包含 ffmpeg 所有子线程）"""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def kill_all_ffmpeg():
    """kill 所有已注册的 ffmpeg 进程组"""
    with _procs_lock:
        for proc in list(_active_procs):
            _kill_proc(proc)
    _active_procs.clear()


# ══════════════════════════════════════════════════════════════
# 状态（本地模式断点续传）
# ══════════════════════════════════════════════════════════════

def load_state(out_root: Path) -> dict:
    sf = out_root / STATE_FILENAME
    if sf.exists():
        try:
            return json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(out_root: Path, state: dict):
    sf  = out_root / STATE_FILENAME
    tmp = sf.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(sf)


# ══════════════════════════════════════════════════════════════
# 单任务处理
# ══════════════════════════════════════════════════════════════

def process_one(
    src:        str,
    dst:        str,
    gpu_id:     int,
    use_nvenc:  bool,
    state:      Optional[dict]           = None,
    state_lock: Optional[threading.Lock] = None,
    out_root:   Optional[Path]           = None,
    queue=None,
) -> JobResult:
    result = JobResult(src_path=src, dst_path=dst)
    t0     = time.time()

    # 本地模式：断点检查
    if state is not None and src in state:
        done_dst = state[src]
        if Path(done_dst).exists() and Path(done_dst).stat().st_size > 1024:
            result.skip = result.success = True
            result.dst_path = done_dst
            return result

    meta = probe(src)
    if meta is None:
        result.error = "ffprobe 失败"
        if queue:
            queue.mark_failed(src, result.error)
        return result

    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    def try_encode(nvenc: bool) -> subprocess.CompletedProcess:
        return run_ffmpeg(build_ffmpeg_cmd(src, dst, meta,
                                          gpu_id=gpu_id, use_nvenc=nvenc))

    try:
        proc = try_encode(use_nvenc)
        if proc.returncode != 0 and use_nvenc:
            log.warning(f"[yellow]NVENC 失败，回退 CPU: {Path(src).name}[/yellow]")
            proc = try_encode(False)
        if proc.returncode != 0:
            result.error = proc.stderr[-600:]
            if queue:
                queue.mark_failed(src, result.error)
            return result

        result.success  = True
        result.dst_path = dst

        if queue:
            queue.mark_done(src, dst)
        elif state is not None and state_lock and out_root:
            with state_lock:
                state[src] = dst
                save_state(out_root, state)

    except subprocess.TimeoutExpired:
        result.error = "超时 >2h"
        if queue:
            queue.mark_failed(src, result.error)
    except Exception as e:
        result.error = str(e)
        if queue:
            queue.mark_failed(src, result.error)

    result.duration_s = time.time() - t0
    return result


# ══════════════════════════════════════════════════════════════
# 队列模式主循环
# ══════════════════════════════════════════════════════════════

def run_queue_mode(queue, out_root, gpu_ids, nvenc_ok, workers, stop_event):
    results    = []
    hb_srcs: dict = {}   # {future: src}
    hb_lock   = threading.Lock()

    def heartbeat_loop():
        while not stop_event.is_set():
            with hb_lock:
                for src in list(hb_srcs.values()):
                    try:
                        queue.heartbeat(src)
                    except Exception:
                        pass
            time.sleep(HEARTBEAT_INTERVAL)

    threading.Thread(target=heartbeat_loop, daemon=True).start()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        TaskProgressColumn(),
        TextColumn("{task.fields[extra]}"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        stats      = queue.stats()
        total_todo = sum(stats.values())
        pid        = progress.add_task(
            f"队列模式 [{queue.worker_id}]",
            total=total_todo,
            extra="",
        )
        progress.update(pid, completed=stats.get("done", 0))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs: dict[concurrent.futures.Future, str] = {}

            def submit_next() -> bool:
                if stop_event.is_set():
                    return False
                item = queue.claim_next()
                if item is None:
                    return False
                src, _, _ = item
                # ⑤ 命名在认领时确定，写入队列的 dst 字段
                dst    = str(out_root / canonical_name(src))
                gpu_id = gpu_ids[len(futs) % len(gpu_ids)]
                fut    = pool.submit(process_one, src, dst, gpu_id, nvenc_ok,
                                     queue=queue)
                futs[fut] = src
                with hb_lock:
                    hb_srcs[fut] = src
                return True

            for _ in range(workers):
                submit_next()

            while futs and not stop_event.is_set():
                done_set, _ = concurrent.futures.wait(
                    futs.keys(), timeout=5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done_set:
                    src = futs.pop(fut)
                    with hb_lock:
                        hb_srcs.pop(fut, None)
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = JobResult(src_path=src, error=str(e))
                        queue.mark_failed(src, str(e))
                    results.append(res)
                    icon = "⏭" if res.skip else ("✅" if res.success else "❌")
                    log.info(
                        f"{icon} {Path(res.src_path).name}"
                        + (f" → {Path(res.dst_path).name}" if res.dst_path else "")
                        + (f" [{res.duration_s:.0f}s]" if res.duration_s else "")
                        + (f" ERR:{res.error[:80]}" if res.error else "")
                    )
                    cur = queue.stats()
                    progress.update(
                        pid,
                        completed=cur.get("done", 0),
                        extra=(f"done={cur.get('done',0)} "
                               f"pending={cur.get('pending',0)} "
                               f"claimed={cur.get('claimed',0)}"),
                    )
                    if not stop_event.is_set():
                        submit_next()

                if not futs:
                    if queue.is_all_done():
                        break
                    time.sleep(10)
                    submit_next()
    return results


# ══════════════════════════════════════════════════════════════
# 扫描
# ══════════════════════════════════════════════════════════════

def scan_videos(root: str) -> list[str]:
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() in VIDEO_EXTENSIONS:
                found.append(os.path.join(dirpath, fn))
    return found


# ══════════════════════════════════════════════════════════════
# NVENC 检测
# ══════════════════════════════════════════════════════════════

def detect_nvenc(gpu_id: int = 0) -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True)
        if "h264_nvenc" not in r.stdout:
            return False
    except Exception:
        return False

    null_out = "NUL" if sys.platform == "win32" else "/dev/null"
    res = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_device", str(gpu_id),
        "-extra_hw_frames", "4",
        "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=24",
        "-frames:v", "10", "-c:v", "h264_nvenc",
        "-preset", "hq", "-qp", "23",
        "-gpu", str(gpu_id), "-f", "null", null_out,
    ], capture_output=True, text=True, timeout=20)
    ok = res.returncode == 0
    log.info(f"NVENC {'✅' if ok else '❌'} (GPU {gpu_id})")
    return ok


def detect_gpu_count() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        return len([l for l in r.stdout.strip().split("\n") if l.strip()])
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="视频预处理 Pipeline v5.1")
    parser.add_argument("input_dir",   help="源视频根目录")
    parser.add_argument("output_dir",  help="输出目录")
    parser.add_argument("--workers",   type=int, default=0)
    parser.add_argument("--gpu-ids",   type=str, default="auto")
    parser.add_argument("--no-nvenc",  action="store_true")
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--log-file",  type=str, default="pipeline.log")
    parser.add_argument("--file-list", type=str, default="")
    parser.add_argument("--pid-file",  type=str, default="")
    parser.add_argument("--queue-dir", type=str, default="",
                        help="共享队列目录，启用队列模式（推荐多机使用）")
    parser.add_argument("--worker-id", type=str, default="",
                        help="队列模式下本机标识（默认 hostname）")
    args = parser.parse_args()

    # ── ② 成为进程组 leader（PGID == PID）────────────────────────
    # 这样调度器可以用 kill -- -PGID 一次性杀掉本进程 + 所有 ffmpeg 子进程
    # 必须在写 PID 文件之前调用
    try:
        os.setsid()
    except OSError:
        pass   # 已经是进程组 leader 时会抛 OSError，忽略即可

    # ── PID 文件 ─────────────────────────────────────────────────
    if args.pid_file:
        pid_path = Path(os.path.expanduser(args.pid_file))
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()))

    # ── 日志 ─────────────────────────────────────────────────────
    fh = logging.FileHandler(args.log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)

    # ── GPU / NVENC ──────────────────────────────────────────────
    gpu_count = detect_gpu_count()
    gpu_ids   = (list(range(max(gpu_count, 1))) if args.gpu_ids == "auto"
                 else [int(x) for x in args.gpu_ids.split(",")])
    nvenc_ok  = (not args.no_nvenc) and detect_nvenc(gpu_id=gpu_ids[0])
    workers   = args.workers or (DEFAULT_WORKERS_NVENC if nvenc_ok else DEFAULT_WORKERS_SW)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── ⑥ 信号处理：收到任何终止信号，先 kill 所有 ffmpeg ────────
    stop_event = threading.Event()

    def _terminate(sig, frame):
        console.print(f"\n[yellow]⚠  收到信号 {sig}，终止所有 ffmpeg 子进程...[/yellow]")
        stop_event.set()
        kill_all_ffmpeg()
        # 删除 PID 文件（表明已退出）
        if args.pid_file:
            try:
                Path(os.path.expanduser(args.pid_file)).unlink(missing_ok=True)
            except Exception:
                pass

    signal.signal(signal.SIGINT,  _terminate)
    signal.signal(signal.SIGTERM, _terminate)

    # ══════════════════════════════════════════════════════════
    # 队列模式
    # ══════════════════════════════════════════════════════════
    if args.queue_dir:
        from task_queue import TaskQueue
        worker_id = args.worker_id or socket.gethostname()
        queue     = TaskQueue(
            queue_dir = os.path.expanduser(args.queue_dir),
            worker_id = worker_id,
        )
        console.rule(f"[bold cyan]🎬  队列模式  [{worker_id}][/bold cyan]")
        mode = f"NVENC-{PRESET_NVENC}" if nvenc_ok else f"libx264-{PRESET_SW}"
        console.print(f"  编码  : [yellow]{mode} × {workers}路[/yellow]")
        console.print(f"  队列  : [green]{args.queue_dir}[/green]")
        stats = queue.stats()
        console.print(
            f"  状态  : pending={stats.get('pending',0)}  "
            f"claimed={stats.get('claimed',0)}  done={stats.get('done',0)}"
        )
        if stats.get("pending", 0) + stats.get("claimed", 0) == 0:
            console.print("[green]队列中无待处理任务，退出。[/green]")
            return

        results = run_queue_mode(queue, out_root, gpu_ids, nvenc_ok, workers, stop_event)

    # ══════════════════════════════════════════════════════════
    # 本地模式
    # ══════════════════════════════════════════════════════════
    else:
        state = {} if args.force else load_state(out_root)
        state_lock = threading.Lock()

        if args.file_list:
            fl = Path(os.path.expanduser(args.file_list))
            videos = [ln.strip() for ln in fl.read_text(encoding="utf-8").splitlines()
                      if ln.strip() and Path(ln.strip()).exists()]
        else:
            videos = scan_videos(args.input_dir)

        if not videos:
            console.print("[red]未找到视频文件！[/red]")
            sys.exit(1)

        # 本地模式下 dst 在提交时由 canonical_name 生成（每次唯一）
        tasks = [(src, str(out_root / canonical_name(src)),
                  gpu_ids[i % len(gpu_ids)], nvenc_ok)
                 for i, src in enumerate(videos)]

        pending = sum(1 for src, *_ in tasks if src not in state)
        console.rule("[bold cyan]🎬  视频预处理 Pipeline  v5.0[/bold cyan]")
        console.print(f"  源目录  : [green]{args.input_dir}[/green]")
        console.print(f"  输出    : [green]{args.output_dir}[/green]")
        mode = f"NVENC-{PRESET_NVENC}" if nvenc_ok else f"libx264-{PRESET_SW}"
        console.print(f"  编码    : [yellow]{mode} × {workers}路[/yellow]")
        console.print(f"  命名    : 原名前6字_uuid8.mp4")
        console.print(f"  文件    : {len(videos)} 总  {len(state)} 已完成  [bold]{pending}[/bold] 待处理")

        if args.dry_run:
            tbl = Table(box=None)
            tbl.add_column("状态", style="green", width=8)
            tbl.add_column("输出名", style="cyan")
            tbl.add_column("源文件", style="white")
            for src, dst, *_ in tasks[:40]:
                done = "✓ skip" if src in state else "→"
                tbl.add_row(done, Path(dst).name, Path(src).name)
            console.print(tbl)
            return

        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=34),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console, refresh_per_second=4,
        ) as prog:
            pid = prog.add_task(f"转码 ({workers}路)...", total=len(tasks))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {
                    pool.submit(process_one, src, dst, gid, nvenc,
                                state=state, state_lock=state_lock,
                                out_root=out_root): src
                    for src, dst, gid, nvenc in tasks
                }
                for fut in concurrent.futures.as_completed(futs):
                    if stop_event.is_set():
                        for f in futs:
                            f.cancel()
                        break
                    res = fut.result()
                    results.append(res)
                    icon = "⏭" if res.skip else ("✅" if res.success else "❌")
                    log.info(
                        f"{icon} {Path(res.src_path).name}"
                        + (f" → {Path(res.dst_path).name}" if not res.skip else "")
                        + (f" [{res.duration_s:.0f}s]" if res.duration_s else "")
                        + (f" ERR:{res.error[:80]}" if res.error else "")
                    )
                    prog.advance(pid)

    # ── 汇总 ─────────────────────────────────────────────────────
    ok_n   = sum(1 for r in results if r.success and not r.skip)
    skip_n = sum(1 for r in results if r.skip)
    fail_n = sum(1 for r in results if not r.success)
    console.rule("[bold]完成[/bold]")
    console.print(f"  ✅ {ok_n}  ⏭ {skip_n}  ❌ {fail_n}")

    if args.pid_file:
        try:
            Path(os.path.expanduser(args.pid_file)).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()