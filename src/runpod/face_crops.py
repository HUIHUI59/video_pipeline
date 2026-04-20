"""src/runpod/face_crops.py

Pod 端人脸检测 + 裁剪工具。用于 Stage 5 Round 2（face_analysis）给 VLM 送高
分辨率脸部 crop，增强细节识别（眉、眼、嘴的表情、朝向），同时降低单 shot 总
token 占用（比起发整帧，224px crop 只用 ~200 token vs 整帧 ~500 token，且 VLM
看到的面部区域放大 6-10 倍）。

检测后端优先级（自动降级）：
  1. yolo_face  — 需要本地 yolov8n-face.pt（Stage 4 已用的模型）
  2. haar       — OpenCV 内置 Haar Cascade（零下载，始终可用）
  3. 都失败 → 返回空列表，pod_runner 会退回到"只发 video"的 Round 2 模式

从 Stage 4 shot_classify.py 借的思路，但这里是 Pod 侧独立脚本，不依赖
Stage 4 的全局状态/线程锁。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger("face_crops")

# 进程内单例缓存（Pod 上只跑一次；避免每 shot 都重新加载 Haar XML）
_DETECTOR_LOCK = threading.Lock()
_DETECTOR_CACHE: dict = {}


def _load_detector(backend: str, yolo_weights: Optional[str] = None):
    """返回 (detector_obj, actual_backend_name)；失败返回 (None, "none")。"""
    key = f"{backend}::{yolo_weights or ''}"
    if key in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[key]

    with _DETECTOR_LOCK:
        if key in _DETECTOR_CACHE:
            return _DETECTOR_CACHE[key]

        chosen = (None, "none")

        # 1) YOLO face（精度最高）
        if backend in ("yolo_face", "auto") and yolo_weights:
            p = Path(yolo_weights).expanduser()
            if p.is_file():
                try:
                    from ultralytics import YOLO
                    chosen = (YOLO(str(p)), "yolo_face")
                    log.info(f"face_crops: 使用 YOLO face 模型 {p}")
                except Exception as ex:
                    log.warning(f"face_crops: YOLO face 加载失败 ({ex})，降级 Haar")

        # 2) OpenCV Haar Cascade（cv2 内置，永远可用）
        if chosen[0] is None and backend in ("haar", "auto", "yolo_face"):
            try:
                import cv2
                xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                cas = cv2.CascadeClassifier(xml_path)
                if cas.empty():
                    raise RuntimeError(f"Haar xml 为空: {xml_path}")
                chosen = (cas, "haar")
                log.info("face_crops: 使用 OpenCV Haar Cascade")
            except Exception as ex:
                log.warning(f"face_crops: Haar 也不可用 ({ex})")

        _DETECTOR_CACHE[key] = chosen
        return chosen


def _detect_faces_in_frame(detector, backend: str, frame_bgr,
                           min_conf: float = 0.30) -> list[tuple[int, int, int, int]]:
    """给一张 BGR 帧返回脸框 [(x, y, w, h), ...]，按面积从大到小。"""
    if detector is None:
        return []
    try:
        import cv2
        if backend == "yolo_face":
            det = detector(frame_bgr, conf=min_conf, verbose=False)
            if not det:
                return []
            boxes = det[0].boxes
            if boxes is None or len(boxes) == 0:
                return []
            xyxy = boxes.xyxy.cpu().numpy()
            rects = [
                (int(r[0]), int(r[1]),
                 int(r[2] - r[0]), int(r[3] - r[1]))
                for r in xyxy
            ]
        else:  # haar
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
            rects = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    except Exception as ex:
        log.debug(f"face detect 异常: {ex}")
        return []

    # 面积降序
    rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    return rects


def _crop_square(frame_bgr, x: int, y: int, w: int, h: int,
                 out_size: int, pad_frac: float = 0.20):
    """
    从 BGR 帧里裁一个正方形的 crop 并 resize 到 out_size。
    用边缘 padding（黑色）处理脸在帧边的情况。pad_frac=20% 给下巴/额头点余量。
    返回 PIL.Image (RGB)。
    """
    import cv2
    from PIL import Image

    h_frame, w_frame = frame_bgr.shape[:2]
    # 扩 padding
    pad = int(round(max(w, h) * pad_frac))
    cx = x + w // 2
    cy = y + h // 2
    side = max(w, h) + 2 * pad
    half = side // 2

    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    # 越界用 copyMakeBorder 补黑边
    top    = max(0, -y1)
    left   = max(0, -x1)
    bottom = max(0, y2 - h_frame)
    right  = max(0, x2 - w_frame)

    x1_c = max(0, x1); y1_c = max(0, y1)
    x2_c = min(w_frame, x2); y2_c = min(h_frame, y2)
    crop = frame_bgr[y1_c:y2_c, x1_c:x2_c]
    if crop.size == 0:
        return None
    if top or bottom or left or right:
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))

    crop_resized = cv2.resize(crop, (out_size, out_size),
                              interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def detect_and_crop_faces(
    video_path: "str | Path",
    *,
    fps: int = 2,
    crop_size: int = 224,
    max_per_frame: int = 4,
    backend: str = "haar",
    yolo_weights: Optional[str] = None,
    max_total_crops: int = 32,
) -> list[tuple[float, int, "object"]]:
    """检测视频里的人脸并裁剪。

    Args:
        video_path:     视频文件路径
        fps:            帧采样率（单位 fps，2 对表情动态已足够）
        crop_size:      输出 crop 边长（正方形）
        max_per_frame:  每帧最多保留多少张脸（按面积降序）
        backend:        "haar" | "yolo_face" | "auto"（先试 yolo 再 haar）
        yolo_weights:   YOLO face 模型文件路径（仅 backend 含 yolo 时）
        max_total_crops: 整条视频最多返回多少个 crop（控制 token 预算）

    Returns:
        [(timestamp_sec, person_slot, PIL.Image), ...]
        - timestamp_sec: 该帧对应视频时间戳（秒）
        - person_slot:   该帧内按面积降序编号（0 = 最大脸）
        - PIL.Image:     RGB，crop_size × crop_size
    """
    try:
        import cv2
    except ImportError:
        log.warning("face_crops: cv2 不可用，返回空 crop 列表")
        return []

    path = Path(video_path)
    if not path.is_file():
        log.warning(f"face_crops: 视频文件不存在 {path}")
        return []

    detector, actual_backend = _load_detector(backend, yolo_weights)
    if detector is None:
        return []

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            log.warning(f"face_crops: 无法打开视频 {path}")
            return []
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames < 1:
            return []
        duration = total_frames / video_fps
        # 生成采样时间戳（秒）
        n_samples = max(1, int(round(duration * fps)))
        timestamps = [duration * i / max(n_samples - 1, 1)
                      if n_samples > 1 else duration / 2
                      for i in range(n_samples)]

        out: list[tuple[float, int, object]] = []
        for ts in timestamps:
            if len(out) >= max_total_crops:
                break
            frame_idx = max(0, min(total_frames - 1,
                                   int(ts * video_fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            rects = _detect_faces_in_frame(detector, actual_backend, frame_bgr)
            for slot, (x, y, w, h) in enumerate(rects[:max_per_frame]):
                if len(out) >= max_total_crops:
                    break
                crop = _crop_square(frame_bgr, x, y, w, h, crop_size)
                if crop is not None:
                    out.append((ts, slot, crop))
        return out
    finally:
        try:
            cap.release()
        except Exception:
            pass
