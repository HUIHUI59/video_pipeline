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
DEFAULT_MODEL          = "yolov8l.pt"       # Large：COCO person AP ~52，首次自动下载 ~90MB
DEFAULT_FACE_MODEL     = "yolov8n-face.pt"  # 社区脸检测模型，首次自动下载 ~6MB
DEFAULT_PERSON_CONF    = 0.35
DEFAULT_FACE_CONF      = 0.30
DEFAULT_SINGLE_RATIO   = 0.15               # 脸框占帧面积 ≥15% 视为 close-up
DEFAULT_WIDE_RATIO     = 0.03               # 脸框占帧面积 ≤3% 视为 wide
DEFAULT_SAMPLE_FRACS   = (0.15, 0.30, 0.50, 0.70, 0.85)   # 5 帧，避开首尾过渡

# 画质阈值（灰度 0-255）
QUALITY_MIN_BRIGHTNESS  = 25.0   # 均值 < 25 → too_dark
QUALITY_MAX_BRIGHTNESS  = 230.0  # 均值 > 230 → too_bright
QUALITY_MIN_CONTRAST    = 15.0   # 标准差 < 15 → low_contrast
QUALITY_MIN_SHARPNESS   = 50.0   # Laplacian 方差 < 50 → blurry

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
# YOLO 模型（按 path 缓存，进程内单例，多线程共享）
# ══════════════════════════════════════════════════════════════

_yolo_cache: dict[str, object] = {}
_yolo_lock   = threading.Lock()
_manifest_locks: dict[str, threading.Lock] = {}
_manifest_lock_guard = threading.Lock()

def get_yolo(model_path: str):
    """按 path 缓存 YOLO 模型；加载失败返回 None（调用方做降级）。"""
    if model_path in _yolo_cache:
        return _yolo_cache[model_path]
    with _yolo_lock:
        if model_path in _yolo_cache:
            return _yolo_cache[model_path]
        try:
            from ultralytics import YOLO
            _yolo_cache[model_path] = YOLO(model_path)
        except Exception as e:
            log.warning(f"加载 YOLO 模型 {model_path} 失败: {e}")
            _yolo_cache[model_path] = None
        return _yolo_cache[model_path]

def _manifest_lock(path: str) -> threading.Lock:
    with _manifest_lock_guard:
        lk = _manifest_locks.get(path)
        if lk is None:
            lk = threading.Lock()
            _manifest_locks[path] = lk
        return lk


def _compute_quality(frame) -> dict:
    """
    计算一帧的画质指标：
      - mean_brightness：亮度均值（0-255）
      - brightness_std：亮度标准差（对比度代理）
      - sharpness：Laplacian 方差（清晰度代理，越大越清晰）
    """
    import cv2
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_b = float(gray.mean())
        std_b  = float(gray.std())
        lap    = cv2.Laplacian(gray, cv2.CV_64F)
        sharp  = float(lap.var())
        return {"mean_brightness": mean_b,
                "brightness_std": std_b,
                "sharpness": sharp}
    except Exception:
        return {"mean_brightness": 0.0,
                "brightness_std": 0.0,
                "sharpness": 0.0}


def _detect_boxes(yolo_model, frame, conf: float, class_filter=None):
    """
    返回 (count, ratios) — 每帧检测框个数 + 每个框面积占比列表。
    模型为 None 时返回 (0, [])。
    """
    if yolo_model is None:
        return 0, []
    try:
        if class_filter is not None:
            det = yolo_model(frame, conf=conf, classes=class_filter, verbose=False)
        else:
            det = yolo_model(frame, conf=conf, verbose=False)
    except Exception:
        return 0, []
    if not det:
        return 0, []
    boxes = det[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0, []
    import numpy as np
    xyxy = boxes.xyxy.cpu().numpy()
    h, w = frame.shape[:2]
    area = max(w * h, 1)
    ratios = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) / area).tolist()
    return len(boxes), ratios

# ══════════════════════════════════════════════════════════════
# 单任务：分类一个 clip
# ══════════════════════════════════════════════════════════════

def classify_one(src: str, manifest_dir: str,
                 model_path: str        = DEFAULT_MODEL,
                 face_model_path: str   = DEFAULT_FACE_MODEL,
                 person_conf: float     = DEFAULT_PERSON_CONF,
                 face_conf: float       = DEFAULT_FACE_CONF,
                 single_ratio: float    = DEFAULT_SINGLE_RATIO,
                 wide_ratio: float      = DEFAULT_WIDE_RATIO,
                 sample_fracs: tuple    = DEFAULT_SAMPLE_FRACS,
                 queue=None) -> ClassifyResult:
    """
    对一个 clip 做：
      1. 采多帧，每帧跑人体检测 (yolov8l) + 脸检测 (yolov8n-face) + 画质计算
      2. 取跨帧 max 作为人/脸数量（避免漏人）
      3. 用脸框而不是人体框做 shot_type 分类（贴合 labelingStandards）
      4. 画质不合格的 clip 标 quality_ok=False，上传时可过滤
    """
    res = ClassifyResult(src_path=src)
    t0  = time.time()

    try:
        import cv2
    except ImportError:
        res.error = "opencv-python 未安装，请运行: pip install opencv-python-headless"
        if queue: queue.mark_failed(src, res.error)
        return res

    try:
        yolo_person = get_yolo(model_path)
    except ImportError:
        res.error = "ultralytics 未安装，请运行: pip install ultralytics"
        if queue: queue.mark_failed(src, res.error)
        return res
    if yolo_person is None:
        res.error = f"无法加载 person 模型 {model_path}"
        if queue: queue.mark_failed(src, res.error)
        return res
    # 脸模型允许加载失败（降级：仅用人体框）
    yolo_face = get_yolo(face_model_path)

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

    if frame_count < 2 or width <= 0 or height <= 0:
        cap.release()
        res.error = f"视频元数据异常 frames={frame_count} wh={width}x{height}"
        if queue: queue.mark_failed(src, res.error)
        return res

    # 采样多帧；每帧：person + face + quality
    per_frame: list[dict] = []  # {"persons":N, "p_ratios":[...],
                                 #  "faces":N, "f_ratios":[...],
                                 #  "quality":{mean_brightness,...}}
    for frac in sample_fracs:
        idx = max(0, min(frame_count - 1, int(frame_count * frac)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        p_n, p_r = _detect_boxes(yolo_person, frame, person_conf, class_filter=[0])
        f_n, f_r = _detect_boxes(yolo_face,   frame, face_conf,   class_filter=None)
        per_frame.append({
            "persons":  p_n, "p_ratios": p_r,
            "faces":    f_n, "f_ratios": f_r,
            "quality":  _compute_quality(frame),
        })
    cap.release()

    if not per_frame:
        res.error = "未采样到有效帧"
        if queue: queue.mark_failed(src, res.error)
        return res

    # 聚合计数：取跨帧 max（避免漏人）
    num_persons = max((fr["persons"] for fr in per_frame), default=0)
    num_faces   = max((fr["faces"]   for fr in per_frame), default=0)

    # 脸框比例聚合
    all_f_ratios: list[float] = []
    max_f_ratios: list[float] = []
    for fr in per_frame:
        if fr["f_ratios"]:
            max_f_ratios.append(max(fr["f_ratios"]))
            all_f_ratios.extend(fr["f_ratios"])
    largest_face_ratio = max(max_f_ratios) if max_f_ratios else 0.0
    avg_face_ratio     = (sum(all_f_ratios) / len(all_f_ratios)) if all_f_ratios else 0.0

    # 人体框比例聚合（作为辅助）
    all_p_ratios: list[float] = []
    max_p_ratios: list[float] = []
    for fr in per_frame:
        if fr["p_ratios"]:
            max_p_ratios.append(max(fr["p_ratios"]))
            all_p_ratios.extend(fr["p_ratios"])
    largest_subject_ratio = max(max_p_ratios) if max_p_ratios else 0.0

    # 分类（脸优先；无脸模型时降级用人体框）
    if num_persons == 0:
        category = "landscape"
        num_people = 0
    elif num_faces == 0:
        # 有人但看不到脸 → 背对镜头或远景
        category = "wide"
        num_people = num_persons
    elif num_faces == 1:
        num_people = max(num_persons, 1)
        if largest_face_ratio >= single_ratio:
            category = "single"
        elif largest_face_ratio <= wide_ratio:
            category = "wide"
        else:
            category = "single"
    elif num_faces in (2, 3):
        num_people = max(num_persons, num_faces)
        category = "dominant" if (avg_face_ratio > 0 and largest_face_ratio > 2.5 * avg_face_ratio) else "multi"
    else:  # 4+ faces
        num_people = max(num_persons, num_faces)
        category = "multi"

    # 画质聚合（跨帧均值）
    q_means = [fr["quality"]["mean_brightness"] for fr in per_frame]
    q_stds  = [fr["quality"]["brightness_std"]  for fr in per_frame]
    q_shrp  = [fr["quality"]["sharpness"]       for fr in per_frame]
    mean_brightness = sum(q_means) / len(q_means) if q_means else 0.0
    brightness_std  = sum(q_stds)  / len(q_stds)  if q_stds  else 0.0
    sharpness       = sum(q_shrp)  / len(q_shrp)  if q_shrp  else 0.0

    issues: list[str] = []
    if mean_brightness < QUALITY_MIN_BRIGHTNESS:  issues.append("too_dark")
    if mean_brightness > QUALITY_MAX_BRIGHTNESS:  issues.append("too_bright")
    if brightness_std  < QUALITY_MIN_CONTRAST:    issues.append("low_contrast")
    if sharpness       < QUALITY_MIN_SHARPNESS:   issues.append("blurry")
    quality_ok = len(issues) == 0

    # 置信度：跨帧人/脸计数一致性
    same_p = sum(1 for fr in per_frame if fr["persons"] == num_persons)
    same_f = sum(1 for fr in per_frame if fr["faces"]   == num_faces)
    confidence = round((same_p + same_f) / (2 * len(per_frame)), 3)

    p          = Path(src)
    movie_stem = p.parent.name or "unknown"
    shot_id    = f"{movie_stem}/{p.stem}"
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
        "num_faces":            num_faces,
        "shot_category":        category,
        "duration_sec":         round(duration, 3),
        "width":                width,
        "height":               height,
        "fps":                  round(fps, 3),
        "largest_subject_ratio": round(largest_subject_ratio, 4),
        "largest_face_ratio":   round(largest_face_ratio, 4),
        "classifier_confidence": confidence,
        "classified_at":        time.time(),
        "quality_ok":           quality_ok,
        "quality_metrics": {
            "mean_brightness": round(mean_brightness, 2),
            "brightness_std":  round(brightness_std, 2),
            "sharpness":       round(sharpness, 2),
            "issues":          issues,
        },
    }

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
              model_path: str, face_model_path: str,
              person_conf: float, face_conf: float,
              single_ratio: float, wide_ratio: float,
              sample_fracs: tuple, stop_ev):
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
                try:
                    item = queue.claim_next()
                except Exception as e:
                    log.warning(f"claim_next 异常 ({e})，稍后重试")
                    return False
                if item is None: return False
                src, _, _idx = item
                try:
                    fut = pool.submit(classify_one, src, manifest_dir,
                                      model_path, face_model_path,
                                      person_conf, face_conf,
                                      single_ratio, wide_ratio,
                                      sample_fracs, queue)
                except Exception as e:
                    log.error(f"submit 异常 ({e})，释放 claim 后重试")
                    try: queue.mark_failed(src, f"submit error: {e}")
                    except Exception: pass
                    return False
                pending_futs[fut] = src
                with hb_lock: current[fut] = src
                return True

            for _ in range(max(1, workers)):
                if not submit_next(): break

            while not stop_ev.is_set():
                try:
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
                                try: queue.mark_failed(src, str(e))
                                except Exception as ee:
                                    log.warning(f"mark_failed 异常 ({ee})")
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
                            try:
                                s2 = queue.stats()
                                prog.update(pid,
                                            completed=s2.get("done", 0),
                                            extra=f"done={s2.get('done',0)} "
                                                  f"pending={s2.get('pending',0)}")
                            except Exception as e:
                                log.warning(f"progress update 异常 ({e})")
                            if not stop_ev.is_set(): submit_next()

                    if not pending_futs:
                        done = False
                        try:
                            done = queue.is_all_done()
                        except Exception as e:
                            log.warning(f"is_all_done 异常 ({e})，保守继续等待")
                        if done: break
                        time.sleep(8)
                        submit_next()
                except Exception as e:
                    # 任何未预料的异常：记日志，sleep 一下，继续主循环（永不崩）
                    log.error(f"run_queue 主循环异常 ({e})，5s 后继续")
                    time.sleep(5)

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
                        help=f"人体检测模型（默认 {DEFAULT_MODEL}，首次自动下载）")
    parser.add_argument("--face-model",  type=str,   default=DEFAULT_FACE_MODEL,
                        help=f"人脸检测模型（默认 {DEFAULT_FACE_MODEL}；加载失败会降级为仅人体检测）")
    parser.add_argument("--person-conf", type=float, default=DEFAULT_PERSON_CONF,
                        help=f"人体检测置信度（默认 {DEFAULT_PERSON_CONF}）")
    parser.add_argument("--face-conf",   type=float, default=DEFAULT_FACE_CONF,
                        help=f"人脸检测置信度（默认 {DEFAULT_FACE_CONF}）")
    parser.add_argument("--single-face-ratio", type=float, default=DEFAULT_SINGLE_RATIO,
                        help=f"single 阈值：最大脸框占帧面积 ≥ 该值判 close-up（默认 {DEFAULT_SINGLE_RATIO}）")
    parser.add_argument("--wide-face-ratio",   type=float, default=DEFAULT_WIDE_RATIO,
                        help=f"wide 阈值：最大脸框 ≤ 该值判 wide（默认 {DEFAULT_WIDE_RATIO}）")
    parser.add_argument("--sample-frames",     type=int,   default=len(DEFAULT_SAMPLE_FRACS),
                        help=f"每个 clip 采样帧数（默认 {len(DEFAULT_SAMPLE_FRACS)}）")
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

    # 按 --sample-frames 计算均匀分布的 fracs，避开首尾 15% 避免过渡帧
    n = max(1, int(args.sample_frames))
    if n == 1:
        sample_fracs = (0.5,)
    else:
        lo, hi = 0.15, 0.85
        sample_fracs = tuple(lo + (hi - lo) * i / (n - 1) for i in range(n))

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
        console.print(f"  人体模型 : {args.model}  conf={args.person_conf}")
        console.print(f"  人脸模型 : {args.face_model}  conf={args.face_conf}")
        console.print(f"  阈值     : single≥{args.single_face_ratio}  wide≤{args.wide_face_ratio}  frames={n}")
        s = queue.stats()
        console.print(f"  状态     : pending={s.get('pending',0)}  "
                      f"claimed={s.get('claimed',0)}  done={s.get('done',0)}")
        if s.get("pending", 0) + s.get("claimed", 0) == 0:
            console.print("[green]队列无待处理任务，退出。[/green]")
        else:
            c = run_queue(queue, manifest_dir, args.workers,
                          args.model, args.face_model,
                          args.person_conf, args.face_conf,
                          args.single_face_ratio, args.wide_face_ratio,
                          sample_fracs, stop_ev)
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
                                args.model, args.face_model,
                                args.person_conf, args.face_conf,
                                args.single_face_ratio, args.wide_face_ratio,
                                sample_fracs): src
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
