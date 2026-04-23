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

import argparse
import atexit
import concurrent.futures
import contextlib
import fcntl
import json
import logging
import os
import signal
import socket
import statistics  # noqa: F401 — kept for potential external use in run_queue extension
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from common.task_queue import TaskQueue

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

DEFAULT_WORKERS        = 2                 # 2 worker 让 GPU 推理与下一帧的 IO/解码 overlap；
                                            # face detector 现已 thread-local，可安全并发
HEARTBEAT_INTERVAL     = 60
DEFAULT_MODEL          = "yolov8l.pt"       # Large：COCO person AP ~52，首次自动下载 ~90MB
DEFAULT_FACE_MODEL     = "yolov8n-face.pt"  # 社区脸检测模型，首次自动下载 ~6MB
DEFAULT_PERSON_CONF    = 0.35
DEFAULT_FACE_CONF      = 0.30
DEFAULT_SINGLE_RATIO   = 0.15               # 脸框占帧面积 ≥15% 视为 close-up
DEFAULT_WIDE_RATIO     = 0.03               # 脸框占帧面积 ≤3% 视为 wide
DEFAULT_SAMPLE_FRACS   = (0.15, 0.30, 0.50, 0.70, 0.85)   # 5 帧，避开首尾过渡

# 画质阈值（灰度 0-255）—— 默认"spec 模式"（delivery_v1 规范值，适合电影软光影）
# 历史行为（更严）通过 --quality-mode strict 恢复。
# 通过 --quality-config YAML 或 --brightness-min / --brightness-max /
# --contrast-min / --sharpness-min / --camera-motion-max CLI flag 覆盖。
QUALITY_MIN_BRIGHTNESS  = 12.0   # 均值 < 该值 → too_dark
QUALITY_MAX_BRIGHTNESS  = 242.0  # 均值 > 该值 → too_bright
QUALITY_MIN_CONTRAST    = 5.0    # 标准差 < 该值 → low_contrast
QUALITY_MIN_SHARPNESS   = 15.0   # Laplacian 方差 < 该值 → blurry
QUALITY_MAX_CAMERA_MOTION = 6.0  # Farneback 光流平均位移（480-宽灰图）> 该值 → camera_shake

# 规范模式快捷值（--quality-mode spec 会应用这五个；跟模块默认一致）
QUALITY_SPEC_MODE = {
    "min_brightness":    QUALITY_MIN_BRIGHTNESS,
    "max_brightness":    QUALITY_MAX_BRIGHTNESS,
    "min_contrast":      QUALITY_MIN_CONTRAST,
    "min_sharpness":     QUALITY_MIN_SHARPNESS,
    "max_camera_motion": QUALITY_MAX_CAMERA_MOTION,
}

# 严格模式（--quality-mode strict 会应用这五个；沿用历史默认，保留给 COCO-grade 清晰度场景）
QUALITY_STRICT_MODE = {
    "min_brightness":    25.0,
    "max_brightness":    230.0,
    "min_contrast":      15.0,
    "min_sharpness":     50.0,
    "max_camera_motion": 6.0,
}

# 光流抖动检测参数
CAMERA_MOTION_N_SAMPLE_FRAMES = 5   # 中间位置连续采几帧做光流
CAMERA_MOTION_RESIZE_WIDTH    = 480 # Farneback 前下采样宽度（原生 1920 上太慢）

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


# ── 脸检测 tier fallback ─────────────────────────────────────────────
# 优先级：
#   0. MediaPipe Tasks FaceDetector（blaze_face_full_range.tflite, conf>=0.5）
#   1. 本地已下载的 YOLO face 模型（yolo_face_path，若存在）
#   2. OpenCV Haar Cascade（opencv-python 自带）—— 最后兜底
#   3. None —— 警告后降级为仅人体检测
#
# Thread-local backend instance: mediapipe FaceDetector is NOT thread-safe.
# Each worker thread builds its own detector via get_face_detector(); the
# backend kind ("mediapipe" / "yolo_face" / "haar" / "none") is selected
# once globally to keep priority logic consistent across workers.
_face_tls = threading.local()           # per-thread (detector, backend)
_face_backend_global: str | None = None # cached "kind" picked once
_face_yolo_path: str | None = None      # remembered yolo_face_path arg
_face_lock = threading.Lock()
MEDIAPIPE_MIN_CONF = 0.5         # 教授标准
# 官方 Google Cloud 下载链接；换成镜像就改这一行
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_full_range/float16/latest/"
    "blaze_face_full_range.tflite"
)
MEDIAPIPE_MODEL_DEFAULT_PATH = str(
    Path.home() / ".cache" / "mediapipe-models" / "blaze_face_full_range.tflite"
)


def _mediapipe_model_path() -> str:
    """Return cached tflite path; download on first use."""
    p = Path(os.environ.get("MEDIAPIPE_FACE_MODEL_PATH",
                            MEDIAPIPE_MODEL_DEFAULT_PATH)).expanduser()
    if p.exists():
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    log.info(f"首次使用 MediaPipe：下载模型 {MEDIAPIPE_MODEL_URL} → {p}")
    urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, str(p))
    return str(p)


def _build_face_detector(backend: str, yolo_face_path: str | None):
    """Construct a fresh detector instance for the given backend kind.
    Called once per thread by get_face_detector — mediapipe FaceDetector
    is NOT thread-safe so each worker thread holds its own copy."""
    if backend == "mediapipe":
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks import python as mp_py
        from mediapipe.tasks.python import vision as mp_vision
        base = mp_py.BaseOptions(model_asset_path=_mediapipe_model_path())
        opts = mp_vision.FaceDetectorOptions(
            base_options=base,
            min_detection_confidence=MEDIAPIPE_MIN_CONF,
        )
        return mp_vision.FaceDetector.create_from_options(opts)
    if backend == "yolo_face" and yolo_face_path:
        from ultralytics import YOLO
        return YOLO(str(Path(yolo_face_path).expanduser()))
    if backend == "haar":
        import cv2
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        return cv2.CascadeClassifier(xml)
    return None


def get_face_detector(yolo_face_path: str | None = None):
    """Return (detector, backend) for the CURRENT thread.

    Backend kind is selected once globally (priority: mediapipe → yolo →
    haar → none) the first time any thread calls in. Each thread then
    instantiates its own detector instance lazily so mediapipe's lack of
    thread-safety doesn't cause races across workers.

    Detection results are bit-identical to the previous single-instance
    version: same model path, same conf threshold, same backend.
    """
    global _face_backend_global, _face_yolo_path
    # Fast path: this thread already has an instance.
    cached = getattr(_face_tls, "value", None)
    if cached is not None:
        return cached
    # First-call path: pick the backend kind once, globally.
    with _face_lock:
        if _face_backend_global is None:
            _face_yolo_path = yolo_face_path
            picked = None
            for kind in ("mediapipe", "yolo_face", "haar"):
                try:
                    inst = _build_face_detector(kind, yolo_face_path)
                    if inst is None:
                        continue
                    if kind == "haar" and getattr(inst, "empty", lambda: False)():
                        raise RuntimeError("Haar cascade xml empty")
                    picked = kind
                    label = {
                        "mediapipe": f"MediaPipe Tasks FaceDetector "
                                     f"(full-range, conf>={MEDIAPIPE_MIN_CONF})",
                        "yolo_face": f"YOLO {yolo_face_path}",
                        "haar": "OpenCV Haar Cascade (fallback)",
                    }[kind]
                    log.info(f"脸检测：使用 {label}")
                    # We just built one — keep it for THIS thread.
                    _face_tls.value = (inst, kind)
                    break
                except Exception as e:
                    log.info(f"{kind} 不可用 ({e})")
            if picked is None:
                _face_backend_global = "none"
                log.warning("所有脸检测后端均不可用。分类器只用人体检测。")
            else:
                _face_backend_global = picked
            if cached is None and getattr(_face_tls, "value", None) is not None:
                return _face_tls.value
    # Backend kind already known — this thread just needs its own instance.
    backend = _face_backend_global
    if backend == "none":
        _face_tls.value = (None, "none")
        return _face_tls.value
    try:
        inst = _build_face_detector(backend, _face_yolo_path)
    except Exception as e:
        log.warning(f"thread-local face detector init failed ({e}); "
                    f"falling back to no face detection in this thread")
        inst = None
        backend = "none"
    _face_tls.value = (inst, backend)
    return _face_tls.value


def detect_faces(frame) -> tuple[int, list[float]]:
    """返回 (脸数, 脸框面积占帧比例列表)。"""
    det, backend = get_face_detector()
    if det is None:
        return 0, []

    if backend == "mediapipe":
        try:
            import cv2
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = det.detect(mp_img)
        except Exception:
            return 0, []
        detections = getattr(result, "detections", None) or []
        if not detections:
            return 0, []
        h, w = frame.shape[:2]
        frame_area = max(1, w * h)
        ratios = []
        for d in detections:
            bb = d.bounding_box  # pixel coords: origin_x/y + width/height
            area = max(0, int(bb.width)) * max(0, int(bb.height))
            r = max(0.0, min(1.0, area / frame_area))
            ratios.append(r)
        return len(ratios), ratios

    if backend == "yolo_face":
        n, r, _bb = _detect_boxes(det, frame, DEFAULT_FACE_CONF, class_filter=None)
        return n, r

    if backend == "haar":
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Haar 对正脸敏感，侧脸容易漏；阈值从默认放宽提高 recall
            faces = det.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4,
                minSize=(24, 24))
        except Exception:
            return 0, []
        if len(faces) == 0:
            return 0, []
        h, w = frame.shape[:2]
        area = max(w * h, 1)
        ratios = [float((fw * fh) / area) for (_fx, _fy, fw, fh) in faces]
        return len(faces), ratios

    return 0, []


@contextlib.contextmanager
def _manifest_lock(path: str):
    """Cross-process advisory lock around manifest append.

    Old code used threading.Lock — only protected within ONE process. With
    distributed_dispatch spawning shot_classify on multiple machines all
    writing the same SMB-shared <movie>.jsonl, that produced byte-level
    interleaving (truncated rows / "Extra data" / empty lines that
    pod_control then warned about).

    fcntl.flock(LOCK_EX) is honored cross-process by both local fs and
    SMB/NFS mounts (with `lock` mount option, which is the default).
    Each manifest path gets a sidecar `.lock` file (zero-byte) used only
    as the lock target — the manifest itself is opened separately as
    append-only, never truncated, so already-classified rows are never
    touched.

    Also keeps the in-process threading.Lock layer to serialize within
    one machine's worker pool BEFORE racing for the cross-machine flock.
    """
    # In-process serialization first (cheap, prevents N threads piling on
    # the kernel lock).
    with _manifest_lock_guard:
        thr_lk = _manifest_locks.get(path)
        if thr_lk is None:
            thr_lk = threading.Lock()
            _manifest_locks[path] = thr_lk
    lock_path = path + ".lock"
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with thr_lk:
        # Cross-process advisory lock via sidecar file.
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception:
                pass
            os.close(fd)


def _compute_camera_motion(
    cap,
    mid_idx: int,
    n_sample: int = CAMERA_MOTION_N_SAMPLE_FRAMES,
    resize_width: int = CAMERA_MOTION_RESIZE_WIDTH,
) -> Optional[float]:
    """粗略估计整 clip 的镜头抖动程度。

    实现：从镜头中央位置连续采 ``n_sample`` 帧，相邻帧对计算
    Farneback 密集光流，返回每对 L2 位移的均值中的最大值（px/frame）。
    下采样到 ``resize_width`` 宽以保速。失败返回 None，不阻塞主流程。
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    try:
        frames = []
        for off in range(n_sample):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, mid_idx + off))
            ok, f = cap.read()
            if not ok or f is None:
                break
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            if w > resize_width:
                new_h = int(h * resize_width / w)
                gray = cv2.resize(gray, (resize_width, new_h))
            frames.append(gray)
        if len(frames) < 2:
            return None
        mags: list[float] = []
        for a, b in zip(frames[:-1], frames[1:]):
            flow = cv2.calcOpticalFlowFarneback(
                a, b, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.linalg.norm(flow, axis=2)
            mags.append(float(mag.mean()))
        return max(mags) if mags else None
    except Exception as e:
        log.debug(f"_compute_camera_motion 失败: {e}")
        return None


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
    """Single-frame YOLO. Kept as backward-compat shim; classify_one
    uses _detect_boxes_batch for the actual hot path."""
    if yolo_model is None:
        return 0, [], []
    try:
        if class_filter is not None:
            det = yolo_model(frame, conf=conf, classes=class_filter, verbose=False)
        else:
            det = yolo_model(frame, conf=conf, verbose=False)
    except Exception:
        return 0, [], []
    if not det:
        return 0, [], []
    boxes = det[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0, [], []
    xyxy = boxes.xyxy.cpu().numpy()
    h, w = frame.shape[:2]
    area = max(w * h, 1)
    ratios = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) / area).tolist()
    hh = float(max(h, 1)); ww = float(max(w, 1))
    bboxes_norm = [
        [float(row[0] / ww), float(row[1] / hh),
         float(row[2] / ww), float(row[3] / hh)]
        for row in xyxy
    ]
    return len(boxes), ratios, bboxes_norm


def _detect_boxes_batch(yolo_model, frames: list, conf: float,
                        class_filter=None):
    """Batched YOLO inference. ultralytics natively accepts a list of
    np.ndarrays and runs ONE GPU forward — but per-image NMS / decoding
    is still independent, so the result for each frame is bit-equivalent
    to calling the model on that frame alone. Same conf threshold, same
    class filter, no letterbox surprises (frames here all share the same
    video resolution).

    Returns list of (count, ratios, bboxes_norm) in input order, matching
    the per-frame shape of _detect_boxes.
    """
    if yolo_model is None or not frames:
        return [(0, [], []) for _ in frames]
    try:
        if class_filter is not None:
            results = yolo_model(frames, conf=conf, classes=class_filter,
                                 verbose=False)
        else:
            results = yolo_model(frames, conf=conf, verbose=False)
    except Exception:
        return [(0, [], []) for _ in frames]
    out = []
    for frame, det in zip(frames, results):
        h, w = frame.shape[:2]
        area = max(w * h, 1)
        boxes = getattr(det, "boxes", None)
        if boxes is None or len(boxes) == 0:
            out.append((0, [], []))
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        ratios = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) / area).tolist()
        hh = float(max(h, 1)); ww = float(max(w, 1))
        bboxes_norm = [
            [float(row[0] / ww), float(row[1] / hh),
             float(row[2] / ww), float(row[3] / hh)]
            for row in xyxy
        ]
        out.append((len(boxes), ratios, bboxes_norm))
    return out


def _detect_faces_batch(frames: list) -> list[tuple[int, list[float]]]:
    """Per-frame face detection. mediapipe doesn't truly batch; this
    centralizes the call site and uses thread-local backends so multi-
    worker classify is safe (mediapipe FaceDetector is NOT thread-safe).
    Result per frame is bit-equivalent to detect_faces(frame)."""
    return [detect_faces(f) for f in frames]

# ══════════════════════════════════════════════════════════════
# 单任务：分类一个 clip
# ══════════════════════════════════════════════════════════════

def classify_one(src: str, manifest_dir: str,
                 model_path: str               = DEFAULT_MODEL,
                 face_model_path: str          = DEFAULT_FACE_MODEL,
                 person_conf: float            = DEFAULT_PERSON_CONF,
                 face_conf: float              = DEFAULT_FACE_CONF,
                 single_ratio: float           = DEFAULT_SINGLE_RATIO,
                 wide_ratio: float             = DEFAULT_WIDE_RATIO,
                 sample_fracs: tuple[float, ...] = DEFAULT_SAMPLE_FRACS,
                 queue: Optional["TaskQueue"]  = None,
                 quality_thresholds: Optional[dict[str, float]] = None,
                 skip_motion_detect: bool = False) -> ClassifyResult:
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
    # 脸检测器初始化：YOLO face（若本地 .pt 存在）→ OpenCV Haar → none
    # （MediaPipe 已从项目里移除，保留的是 get_face_detector 里 YOLO > Haar 的顺序）
    get_face_detector(face_model_path)

    cap = cv2.VideoCapture(src)
    # 用 try/finally 保证任何异常路径都会 release cap，避免 FD 泄漏
    try:
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
            res.error = f"视频元数据异常 frames={frame_count} wh={width}x{height}"
            if queue: queue.mark_failed(src, res.error)
            return res

        # ── Sequential frame collection (precision: better than seek) ──
        # cv2 cap.set(POS_FRAMES) on H.264 is NOT frame-accurate (it
        # snaps to the nearest keyframe). Sequential read is exact, AND
        # avoids slow random-seek over SMB shares. We compute the target
        # indices first, then read sequentially keeping only the targets.
        target_indices = sorted({
            max(0, min(frame_count - 1, int(frame_count * frac)))
            for frac in sample_fracs
        })
        sampled_frames: list = []
        cur_idx = 0
        next_target_pos = 0
        while next_target_pos < len(target_indices):
            target = target_indices[next_target_pos]
            # Skip-read until we reach `target`. cv2.read() advances
            # one frame and decodes; passing grab() instead avoids
            # decode for the throwaway frames in between.
            while cur_idx < target:
                if not cap.grab():
                    break
                cur_idx += 1
            if cur_idx != target:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                next_target_pos += 1
                cur_idx += 1
                continue
            sampled_frames.append(frame)
            cur_idx += 1
            next_target_pos += 1

        # ── Batched detection (precision: bit-equivalent to per-frame) ──
        # ultralytics accepts list[np.ndarray] → 1 GPU forward for N
        # frames. NMS / decoding still per-image, so per-frame results
        # match what we would have gotten calling the model on each
        # frame individually.
        person_results = _detect_boxes_batch(
            yolo_person, sampled_frames, person_conf, class_filter=[0]
        )
        face_results = _detect_faces_batch(sampled_frames)

        per_frame: list[dict] = []
        for frame, (p_n, p_r, p_bb), (f_n, f_r) in zip(
            sampled_frames, person_results, face_results
        ):
            per_frame.append({
                "persons":  p_n, "p_ratios": p_r, "p_bboxes": p_bb,
                "faces":    f_n, "f_ratios": f_r,
                "quality":  _compute_quality(frame),
            })

        # 光流抖动检测（必须在 cap 释放前；本身只读不写，安全）
        camera_motion: Optional[float] = None
        if not skip_motion_detect:
            camera_motion = _compute_camera_motion(cap, mid_idx=frame_count // 2)
    finally:
        try:
            cap.release()
        except Exception:
            pass

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

    # 人体框比例聚合（作为辅助），同时抓跨帧面积最大的 bbox 给 Stage 5
    all_p_ratios: list[float] = []
    max_p_ratios: list[float] = []
    largest_subject_bbox: Optional[list[float]] = None
    largest_subject_area_norm: float = 0.0
    for fr in per_frame:
        if fr["p_ratios"]:
            max_p_ratios.append(max(fr["p_ratios"]))
            all_p_ratios.extend(fr["p_ratios"])
            # 在这一帧里找最大的 person bbox
            p_ratios_f = fr["p_ratios"]
            p_bboxes_f = fr.get("p_bboxes") or []
            if p_bboxes_f and len(p_bboxes_f) == len(p_ratios_f):
                k = max(range(len(p_ratios_f)), key=lambda i: p_ratios_f[i])
                if p_ratios_f[k] > largest_subject_area_norm:
                    largest_subject_area_norm = p_ratios_f[k]
                    largest_subject_bbox = list(p_bboxes_f[k])
    largest_subject_ratio = max(max_p_ratios) if max_p_ratios else 0.0
    largest_subject_vertical_center: Optional[float] = (
        (largest_subject_bbox[1] + largest_subject_bbox[3]) / 2.0
        if largest_subject_bbox else None
    )

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

    # 阈值：优先用传入的 dict，缺失项退回模块常量
    th = quality_thresholds or {}
    th_min_brightness   = float(th.get("min_brightness", QUALITY_MIN_BRIGHTNESS))
    th_max_brightness   = float(th.get("max_brightness", QUALITY_MAX_BRIGHTNESS))
    th_min_contrast     = float(th.get("min_contrast",   QUALITY_MIN_CONTRAST))
    th_min_sharpness    = float(th.get("min_sharpness",  QUALITY_MIN_SHARPNESS))
    th_max_camera_motion = float(th.get("max_camera_motion", QUALITY_MAX_CAMERA_MOTION))

    issues: list[str] = []
    if mean_brightness < th_min_brightness:  issues.append("too_dark")
    if mean_brightness > th_max_brightness:  issues.append("too_bright")
    if brightness_std  < th_min_contrast:    issues.append("low_contrast")
    if sharpness       < th_min_sharpness:   issues.append("blurry")
    if (camera_motion is not None
            and camera_motion > th_max_camera_motion):
        issues.append("camera_shake")
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
        "largest_subject_bbox": (
            [round(v, 4) for v in largest_subject_bbox]
            if largest_subject_bbox else None
        ),
        "largest_subject_vertical_center": (
            round(largest_subject_vertical_center, 4)
            if largest_subject_vertical_center is not None else None
        ),
        "classifier_confidence": confidence,
        "classified_at":        time.time(),
        "quality_ok":           quality_ok,
        "quality_metrics": {
            "mean_brightness": round(mean_brightness, 2),
            "brightness_std":  round(brightness_std, 2),
            "sharpness":       round(sharpness, 2),
            "camera_motion":   (round(camera_motion, 3)
                                if camera_motion is not None else None),
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
              sample_fracs: tuple, stop_ev,
              quality_thresholds: Optional[dict[str, float]] = None,
              skip_motion_detect: bool = False):
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
                                      sample_fracs, queue,
                                      quality_thresholds,
                                      skip_motion_detect)
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
    # ── 画质阈值：YAML 或 CLI 覆盖（不给则用严格模式常量）
    parser.add_argument("--quality-config", type=str, default="",
                        help="画质阈值 YAML，键：min_brightness/max_brightness/"
                             "min_contrast/min_sharpness。优先级低于下面 4 个 CLI flag。")
    parser.add_argument("--quality-mode", type=str, default="",
                        choices=["", "strict", "spec"],
                        help="'spec'/空（默认）= delivery_v1 规范值 12/242/5/15（电影素材推荐）；"
                             "'strict' = 严格模式（25/230/15/50，历史默认，适合 COCO 级清晰度要求）。")
    parser.add_argument("--brightness-min", type=float, default=None)
    parser.add_argument("--brightness-max", type=float, default=None)
    parser.add_argument("--contrast-min",   type=float, default=None)
    parser.add_argument("--sharpness-min",  type=float, default=None)
    parser.add_argument("--camera-motion-max", type=float, default=None,
                        help=f"Farneback 光流平均位移（480-宽灰图）> 该值判 camera_shake；"
                             f"默认 {QUALITY_MAX_CAMERA_MOTION}")
    parser.add_argument("--skip-motion-detect", action="store_true",
                        help="跳过光流抖动检测（省时间，但 manifest 没 camera_motion）")
    args = parser.parse_args()

    # ── 解析画质阈值：优先级 CLI flag > YAML > --quality-mode > 模块默认（= spec）
    quality_thresholds: dict[str, float] = {
        "min_brightness":    QUALITY_MIN_BRIGHTNESS,
        "max_brightness":    QUALITY_MAX_BRIGHTNESS,
        "min_contrast":      QUALITY_MIN_CONTRAST,
        "min_sharpness":     QUALITY_MIN_SHARPNESS,
        "max_camera_motion": QUALITY_MAX_CAMERA_MOTION,
    }
    if args.quality_mode == "strict":
        quality_thresholds.update(QUALITY_STRICT_MODE)
    # "spec" 或 "" 都保持模块默认（本来就是 spec 值）
    if args.quality_config:
        try:
            import yaml as _yaml
            cfg_path = Path(os.path.expanduser(args.quality_config))
            if cfg_path.exists():
                loaded = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                for k in ("min_brightness", "max_brightness",
                          "min_contrast", "min_sharpness",
                          "max_camera_motion"):
                    if k in loaded:
                        quality_thresholds[k] = float(loaded[k])
            else:
                log.warning(f"--quality-config {cfg_path} 不存在，忽略")
        except Exception as e:
            log.warning(f"--quality-config 解析失败 ({e})，使用默认")
    for cli_key, th_key in (("brightness_min",    "min_brightness"),
                            ("brightness_max",    "max_brightness"),
                            ("contrast_min",      "min_contrast"),
                            ("sharpness_min",     "min_sharpness"),
                            ("camera_motion_max", "max_camera_motion")):
        v = getattr(args, cli_key)
        if v is not None:
            quality_thresholds[th_key] = float(v)
    log.info(f"画质阈值：{quality_thresholds}")

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
                          sample_fracs, stop_ev,
                          quality_thresholds=quality_thresholds,
                          skip_motion_detect=args.skip_motion_detect)
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
                                sample_fracs, None,
                                quality_thresholds,
                                args.skip_motion_detect): src
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
