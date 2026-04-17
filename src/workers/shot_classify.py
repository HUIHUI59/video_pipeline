#!/usr/bin/env python3
"""
shot_classify.py  v1.0
════════════════════════════════════════════════════════════════
Stage 4：镜头预分类
  - 从 Stage 2 输出的 clips (shot_NNN.mp4) 读入
  - 用 YOLOv8 (ultralytics) 检测每个 clip 的"人"数量和最大人体框占比
  - 按规则分类为 single / dominant / multi / wide / landscape
  - 每个 clip 追加一行 JSONL 到 output_dir/manifest/{movie_stem}.jsonl
  - 字段对齐 docs/labelingStandards/json_schema_integrated.md 的 manifest 约定
  - 支持分布式队列模式 (--queue-dir) 和本地模式

分类规则：
  num_people == 0                              → landscape
  num_people == 1, subject_ratio >= single_th  → single
  num_people == 1, subject_ratio <= wide_th    → wide
  num_people == 1, 介于之间                     → single (保守归类)
  num_people in [2,3], 最大框 > 2.5*均值        → dominant
  num_people in [2,3], 否则                    → multi
  num_people >= 4                              → multi (crowd 留给 Stage 5 细分)

依赖：pip install ultralytics opencv-python-headless
════════════════════════════════════════════════════════════════
"""

import os, sys, time, signal, socket, atexit, json, statistics
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

DEFAULT_WORKERS        = 1                 # YOLO 串行在 GPU，单线程即可
HEARTBEAT_INTERVAL     = 60
DEFAULT_MODEL          = "yolov8n.pt"       # COCO，class 0=person；首次自动下载
DEFAULT_FACE_CONF      = 0.35
DEFAULT_SINGLE_RATIO   = 0.40               # 人体框占帧面积 ≥40% 视为 close-up
DEFAULT_WIDE_RATIO     = 0.10               # 人体框占帧面积 ≤10% 视为 wide
DEFAULT_SAMPLE_FRACS   = (0.25, 0.50, 0.75) # 每个 clip 采 3 帧

# ══════════════════════════════════════════════════════════════
# 日志
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)
log     = logging.getLogger("shot_classify")
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
class ClassifyResult:
    src_path:   str
    entry:      dict = field(default_factory=dict)
    category:   str  = ""
    num_people: int  = 0
    success:    bool = False
    error:      str  = ""
    duration_s: float = 0.0

# ══════════════════════════════════════════════════════════════
# YOLO 模型（进程内单例，多线程共享）
# ══════════════════════════════════════════════════════════════

_yolo_model  = None
_yolo_lock   = threading.Lock()
_manifest_locks: dict[str, threading.Lock] = {}
_manifest_lock_guard = threading.Lock()

def get_yolo(model_path: str):
    global _yolo_model
    if _yolo_model is None:
        with _yolo_lock:
            if _yolo_model is None:
                from ultralytics import YOLO
                _yolo_model = YOLO(model_path)
    return _yolo_model

def _manifest_lock(path: str) -> threading.Lock:
    with _manifest_lock_guard:
        lk = _manifest_locks.get(path)
        if lk is None:
            lk = threading.Lock()
            _manifest_locks[path] = lk
        return lk

# ══════════════════════════════════════════════════════════════
# 单任务：分类一个 clip
# ══════════════════════════════════════════════════════════════

def classify_one(src: str, manifest_dir: str,
                 model_path: str      = DEFAULT_MODEL,
                 face_conf: float     = DEFAULT_FACE_CONF,
                 single_ratio: float  = DEFAULT_SINGLE_RATIO,
                 wide_ratio: float    = DEFAULT_WIDE_RATIO,
                 sample_fracs: tuple  = DEFAULT_SAMPLE_FRACS,
                 queue=None) -> ClassifyResult:
    res = ClassifyResult(src_path=src)
    t0  = time.time()

    try:
        import cv2
    except ImportError:
        res.error = "opencv-python 未安装，请运行: pip install opencv-python-headless"
        if queue: queue.mark_failed(src, res.error)
        return res

    try:
        yolo = get_yolo(model_path)
    except ImportError:
        res.error = "ultralytics 未安装，请运行: pip install ultralytics"
        if queue: queue.mark_failed(src, res.error)
        return res
    except Exception as e:
        res.error = f"YOLO 加载失败: {e}"
        if queue: queue.mark_failed(src, res.error)
        return res

    # 打开视频
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        res.error = "无法打开视频"
        if queue: queue.mark_failed(src, res.error)
        return res

    fps         = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration    = frame_count / fps if fps > 0 else 0.0
    frame_area  = max(width * height, 1)

    if frame_count < 2 or width <= 0 or height <= 0:
        cap.release()
        res.error = f"视频元数据异常 frames={frame_count} wh={width}x{height}"
        if queue: queue.mark_failed(src, res.error)
        return res

    # 采样 3 帧，对每帧跑检测
    frame_results: list[tuple[int, list[float]]] = []
    for frac in sample_fracs:
        idx = max(0, min(frame_count - 1, int(frame_count * frac)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        try:
            det = yolo(frame, conf=face_conf, classes=[0], verbose=False)
        except Exception as e:
            cap.release()
            res.error = f"YOLO 推理失败: {e}"
            if queue: queue.mark_failed(src, res.error)
            return res
        if not det:
            frame_results.append((0, []))
            continue
        boxes = det[0].boxes
        if boxes is None or len(boxes) == 0:
            frame_results.append((0, []))
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        areas  = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        ratios = (areas / frame_area).tolist()
        frame_results.append((len(boxes), ratios))
    cap.release()

    if not frame_results:
        res.error = "未采样到有效帧"
        if queue: queue.mark_failed(src, res.error)
        return res

    # 聚合：中位数去抖
    counts = [fc for fc, _ in frame_results]
    num_subjects = int(statistics.median(counts))
    all_ratios: list[float] = []
    max_ratios: list[float] = []
    for _fc, rs in frame_results:
        if rs:
            max_ratios.append(max(rs))
            all_ratios.extend(rs)
    largest_ratio = max(max_ratios) if max_ratios else 0.0
    avg_ratio     = (sum(all_ratios) / len(all_ratios)) if all_ratios else 0.0

    # 分类规则
    if num_subjects == 0:
        category   = "landscape"
        num_people = 0
    elif num_subjects == 1:
        num_people = 1
        if largest_ratio >= single_ratio:
            category = "single"
        elif largest_ratio <= wide_ratio:
            category = "wide"
        else:
            category = "single"
    elif 2 <= num_subjects <= 3:
        num_people = num_subjects
        category   = "dominant" if (avg_ratio > 0 and largest_ratio > 2.5 * avg_ratio) else "multi"
    else:
        num_people = num_subjects
        category   = "multi"

    # 置信度（启发式：多帧一致性越高越置信）
    if len(counts) >= 2:
        same_count = sum(1 for c in counts if c == num_subjects)
        confidence = round(same_count / len(counts), 3)
    else:
        confidence = 0.5

    # movie_stem = 父目录名；shot_id = "movie/shot_stem"
    p          = Path(src)
    movie_stem = p.parent.name or "unknown"
    shot_id    = f"{movie_stem}/{p.stem}"

    # 相对路径：保留到 clips/ 开始
    try:
        parts = p.parts
        if "clips" in parts:
            path_str = "/".join(parts[parts.index("clips"):])
        else:
            path_str = f"{movie_stem}/{p.name}"
    except Exception:
        path_str = str(src)

    entry = {
        "shot_id":              shot_id,
        "source_movie":         movie_stem,
        "path":                 path_str,
        "num_people":           num_people,
        "shot_category":        category,
        "duration_sec":         round(duration, 3),
        "width":                width,
        "height":               height,
        "fps":                  round(fps, 3),
        "largest_subject_ratio": round(largest_ratio, 4),
        "classifier_confidence": confidence,
        "classified_at":        time.time(),
    }

    # 写 manifest：每部电影一个 JSONL
    manifest_path = str(Path(manifest_dir) / f"{movie_stem}.jsonl")
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with _manifest_lock(manifest_path):
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    res.entry      = entry
    res.category   = category
    res.num_people = num_people
    res.success    = True
    res.duration_s = time.time() - t0

    if queue:
        queue.mark_done(src, manifest_path)
    return res

# ══════════════════════════════════════════════════════════════
# 扫描
# ══════════════════════════════════════════════════════════════

def scan_clips(root: str) -> list[str]:
    """递归扫描 clips 目录下所有 shot_*.mp4。"""
    root_path = Path(root).resolve()
    found = []
    for d, _dirs, fns in os.walk(root_path):
        for fn in sorted(fns):
            if fn.lower().endswith(".mp4") and fn.startswith("shot_"):
                found.append(os.path.join(d, fn))
    return found

# ══════════════════════════════════════════════════════════════
# 队列模式主循环
# ══════════════════════════════════════════════════════════════

def run_queue(queue, manifest_dir: str, workers: int,
              model_path: str, face_conf: float,
              single_ratio: float, wide_ratio: float, stop_ev):
    counts  = {"ok": 0, "err": 0, "landscape": 0, "single": 0,
               "dominant": 0, "multi": 0, "wide": 0}
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
        pid = prog.add_task(f"镜头分类[{queue.worker_id}]",
                            total=total_now, extra="")
        prog.update(pid, completed=stats.get("done", 0))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending_futs: dict[concurrent.futures.Future, str] = {}

            def submit_next():
                if stop_ev.is_set(): return False
                item = queue.claim_next()
                if item is None: return False
                src, _, _idx = item
                fut = pool.submit(classify_one, src, manifest_dir,
                                  model_path, face_conf,
                                  single_ratio, wide_ratio,
                                  DEFAULT_SAMPLE_FRACS, queue)
                pending_futs[fut] = src
                with hb_lock: current[fut] = src
                return True

            for _ in range(max(1, workers)):
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
                            res = ClassifyResult(src_path=src, error=str(e))
                            queue.mark_failed(src, str(e))
                        if res.success:
                            counts["ok"] += 1
                            counts[res.category] = counts.get(res.category, 0) + 1
                            icon = "✅"
                            detail = f"{res.category} n={res.num_people}"
                        else:
                            counts["err"] += 1
                            icon = "❌"
                            detail = f"ERR:{res.error[:60]}"
                        log.info(f"{icon} {Path(res.src_path).name}  {detail}"
                                 + (f" [{res.duration_s:.1f}s]" if res.duration_s else ""))
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

    return counts

# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="镜头分类 Pipeline Stage 4 v1.0")
    parser.add_argument("input_dir",  help="clips 根目录（Stage 2 输出）")
    parser.add_argument("output_dir", help="manifest 输出根目录")
    parser.add_argument("--workers",     type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--model",       type=str,   default=DEFAULT_MODEL,
                        help=f"YOLO 模型路径（默认 {DEFAULT_MODEL}，首次自动下载）")
    parser.add_argument("--face-conf",   type=float, default=DEFAULT_FACE_CONF,
                        help=f"YOLO 置信度阈值（默认 {DEFAULT_FACE_CONF}）")
    parser.add_argument("--single-face-ratio", type=float, default=DEFAULT_SINGLE_RATIO,
                        help=f"单人特写阈值：最大人体框占帧面积 ≥ 该值判为 single（默认 {DEFAULT_SINGLE_RATIO}）")
    parser.add_argument("--wide-face-ratio",   type=float, default=DEFAULT_WIDE_RATIO,
                        help=f"远景阈值：最大人体框 ≤ 该值判为 wide（默认 {DEFAULT_WIDE_RATIO}）")
    parser.add_argument("--log-file",    type=str, default="shot_classify.log")
    parser.add_argument("--pid-file",    type=str, default="")
    parser.add_argument("--queue-dir",   type=str, default="")
    parser.add_argument("--worker-id",   type=str, default="")
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

    manifest_dir = str(Path(args.output_dir) / "manifest")
    Path(manifest_dir).mkdir(parents=True, exist_ok=True)

    # ══ 队列模式 ══════════════════════════════════════════════
    if args.queue_dir:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from common.task_queue import TaskQueue
        wid   = args.worker_id or socket.gethostname()
        queue = TaskQueue(queue_dir=os.path.expanduser(args.queue_dir),
                          worker_id=wid, queue_name="classify_queue")
        console.rule(f"[bold cyan]🎯  镜头分类队列模式  [{wid}][/bold cyan]")
        console.print(f"  输入     : {args.input_dir}")
        console.print(f"  Manifest : {manifest_dir}")
        console.print(f"  模型     : {args.model}  conf={args.face_conf}")
        console.print(f"  阈值     : single≥{args.single_face_ratio}  wide≤{args.wide_face_ratio}")
        s = queue.stats()
        console.print(f"  状态     : pending={s.get('pending',0)}  "
                      f"claimed={s.get('claimed',0)}  done={s.get('done',0)}")
        if s.get("pending", 0) + s.get("claimed", 0) == 0:
            console.print("[green]队列无待处理任务，退出。[/green]")
        else:
            c = run_queue(queue, manifest_dir, args.workers,
                          args.model, args.face_conf,
                          args.single_face_ratio, args.wide_face_ratio, stop_ev)
            console.rule("[bold]分类完成[/bold]")
            console.print(
                f"  ✅ ok={c['ok']}  ❌ err={c['err']}  |  "
                f"landscape={c.get('landscape',0)}  single={c.get('single',0)}  "
                f"dominant={c.get('dominant',0)}  multi={c.get('multi',0)}  "
                f"wide={c.get('wide',0)}")

    # ══ 本地模式 ══════════════════════════════════════════════
    else:
        clips = scan_clips(args.input_dir)
        if not clips:
            console.print("[red]未找到 shot_*.mp4 文件！[/red]"); sys.exit(1)
        console.rule("[bold cyan]🎯  镜头分类本地模式  v1.0[/bold cyan]")
        console.print(f"  共 {len(clips)} 个 clips → {manifest_dir}")
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(classify_one, src, manifest_dir,
                                args.model, args.face_conf,
                                args.single_face_ratio, args.wide_face_ratio): src
                    for src in clips}
            for fut in concurrent.futures.as_completed(futs):
                if stop_ev.is_set():
                    for f in futs: f.cancel()
                    break
                res = fut.result(); results.append(res)
                icon = "✅" if res.success else "❌"
                if res.success:
                    log.info(f"{icon} {Path(res.src_path).name}  "
                             f"{res.category} n={res.num_people} [{res.duration_s:.1f}s]")
                else:
                    log.info(f"{icon} {Path(res.src_path).name}  ERR:{res.error[:60]}")

        ok = sum(1 for r in results if r.success)
        er = len(results) - ok
        console.rule("[bold]分类完成[/bold]")
        console.print(f"  ✅={ok}  ❌={er}")
        if er:
            for r in results:
                if not r.success:
                    console.print(f"  {Path(r.src_path).name}: {r.error}")

    if args.pid_file:
        try: Path(os.path.expanduser(args.pid_file)).unlink(missing_ok=True)
        except Exception: pass


if __name__ == "__main__":
    main()
