"""Stage 4 画质阈值可配置 + manifest 新字段（bbox / vertical_center）的单元测试。

只测纯 Python 的阈值逻辑和 Pydantic 校验，不跑 YOLO/opencv 全链路。

运行：
  pytest tests/test_stage4_quality.py
  或  python tests/test_stage4_quality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT / "src"), str(_ROOT / "src" / "workers"),
          str(_ROOT / "src" / "common"), str(_ROOT / "src" / "runpod")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _threshold_decide(mean_b, std_b, sharp, thresholds, camera_motion=None):
    """复刻 shot_classify.classify_one 里的 issue 分类逻辑（含 camera_shake）。
    把被测逻辑抽出来单测，避免跑整条 OpenCV/YOLO 链。"""
    th_min_b = thresholds["min_brightness"]
    th_max_b = thresholds["max_brightness"]
    th_min_c = thresholds["min_contrast"]
    th_min_s = thresholds["min_sharpness"]
    th_max_m = thresholds.get("max_camera_motion")
    issues = []
    if mean_b < th_min_b: issues.append("too_dark")
    if mean_b > th_max_b: issues.append("too_bright")
    if std_b  < th_min_c: issues.append("low_contrast")
    if sharp  < th_min_s: issues.append("blurry")
    if (camera_motion is not None and th_max_m is not None
            and camera_motion > th_max_m):
        issues.append("camera_shake")
    return issues, len(issues) == 0


def _defaults():
    """Return (defaults=spec values as current module constants, strict=QUALITY_STRICT_MODE)."""
    from shot_classify import (
        QUALITY_MIN_BRIGHTNESS, QUALITY_MAX_BRIGHTNESS,
        QUALITY_MIN_CONTRAST, QUALITY_MIN_SHARPNESS,
        QUALITY_MAX_CAMERA_MOTION, QUALITY_STRICT_MODE,
    )
    return {
        "min_brightness":    QUALITY_MIN_BRIGHTNESS,
        "max_brightness":    QUALITY_MAX_BRIGHTNESS,
        "min_contrast":      QUALITY_MIN_CONTRAST,
        "min_sharpness":     QUALITY_MIN_SHARPNESS,
        "max_camera_motion": QUALITY_MAX_CAMERA_MOTION,
    }, QUALITY_STRICT_MODE


def test_defaults_are_spec():
    """Module defaults should be spec values (looser); strict mode tightens them."""
    defaults, strict = _defaults()
    # strict 更严：下界更大（越高越拒）、上界更小（越低越拒）
    assert defaults["min_brightness"] <= strict["min_brightness"]
    assert defaults["max_brightness"] >= strict["max_brightness"]
    assert defaults["min_contrast"]   <= strict["min_contrast"]
    assert defaults["min_sharpness"]  <= strict["min_sharpness"]
    # camera_motion 两种模式一致（教授标准 6.0）
    assert defaults["max_camera_motion"] == strict["max_camera_motion"]


def test_borderline_shot_passed_by_default_blocked_by_strict():
    """亮度/对比度/清晰度都处于 spec 松、strict 严之间的边界值。
    默认（spec）quality_ok=True；strict 模式 quality_ok=False。"""
    defaults, strict = _defaults()
    # 挑每个指标在"介于 spec 和 strict 之间"的值
    mean_b = 20.0   # spec 12..242 allows；strict 25..230 blocks (too_dark)
    std_b  = 10.0   # spec 5 allows；strict 15 blocks (low_contrast)
    sharp  = 30.0   # spec 15 allows；strict 50 blocks (blurry)

    issues_default, ok_default = _threshold_decide(mean_b, std_b, sharp, defaults)
    issues_strict,  ok_strict  = _threshold_decide(mean_b, std_b, sharp, strict)
    assert ok_default, f"spec should accept: {issues_default}"
    assert not ok_strict, f"strict should reject: {issues_strict}"


def test_spec_mode_values_match_delivery_v1():
    """Regression lock：默认 mode 是 12/242/5/15（04_shot_classify.md 规范默认值）。"""
    defaults, _ = _defaults()
    assert defaults["min_brightness"] == 12.0
    assert defaults["max_brightness"] == 242.0
    assert defaults["min_contrast"]   == 5.0
    assert defaults["min_sharpness"]  == 15.0


def test_strict_mode_values():
    """Regression lock：strict 模式保留老阈值 25/230/15/50。"""
    _, strict = _defaults()
    assert strict["min_brightness"] == 25.0
    assert strict["max_brightness"] == 230.0
    assert strict["min_contrast"]   == 15.0
    assert strict["min_sharpness"]  == 50.0


def test_manifestentry_accepts_new_bbox_fields():
    """ManifestEntry 同时接受带/不带 bbox 的 manifest 行（向后兼容）。"""
    from schemas import ManifestEntry
    old_row = {
        "shot_id": "m/shot_001", "source_movie": "m", "path": "clips/m/shot_001.mp4",
        "num_people": 1, "shot_category": "single",
        "duration_sec": 1.0, "width": 1920, "height": 804, "fps": 24.0,
        "largest_subject_ratio": 0.5, "classifier_confidence": 0.9,
        "classified_at": 1_700_000_000.0,
    }
    ManifestEntry.model_validate(old_row)

    new_row = {**old_row,
               "num_faces": 1, "largest_face_ratio": 0.18,
               "quality_ok": True,
               "quality_metrics": {"mean_brightness": 120.0,
                                   "brightness_std": 40.0,
                                   "sharpness": 80.0, "issues": []},
               "largest_subject_bbox": [0.1, 0.2, 0.6, 0.9],
               "largest_subject_vertical_center": 0.55}
    me = ManifestEntry.model_validate(new_row)
    assert me.largest_subject_bbox == [0.1, 0.2, 0.6, 0.9]
    assert me.largest_subject_vertical_center == 0.55


def test_camera_shake_threshold_logic():
    """camera_motion > max_camera_motion → issues 含 camera_shake + quality_ok=False。"""
    strict, _ = _defaults()
    # 其它指标都合格
    good_mean, good_std, good_sharp = 120.0, 40.0, 100.0
    # 未提供 camera_motion：通过
    issues, ok = _threshold_decide(good_mean, good_std, good_sharp, strict)
    assert ok and "camera_shake" not in issues
    # camera_motion 低于阈值：通过
    issues, ok = _threshold_decide(good_mean, good_std, good_sharp, strict,
                                   camera_motion=2.0)
    assert ok and "camera_shake" not in issues
    # camera_motion 高于阈值：不通过
    issues, ok = _threshold_decide(good_mean, good_std, good_sharp, strict,
                                   camera_motion=strict["max_camera_motion"] + 0.5)
    assert not ok and "camera_shake" in issues


def test_spec_mode_includes_max_camera_motion():
    """Regression：spec mode 字典包含 max_camera_motion 键，数值与 strict 默认对齐。"""
    strict, spec = _defaults()
    assert "max_camera_motion" in spec
    assert spec["max_camera_motion"] == strict["max_camera_motion"]


def test_manifestentry_accepts_camera_motion_field():
    """quality_metrics.camera_motion 是 Optional；不带 / 带都通过 Pydantic 校验。"""
    from schemas import ManifestEntry
    row_without = {
        "shot_id": "m/shot_001", "source_movie": "m", "path": "clips/m/shot_001.mp4",
        "num_people": 1, "shot_category": "single",
        "duration_sec": 1.0, "width": 1920, "height": 804, "fps": 24.0,
        "largest_subject_ratio": 0.5, "classifier_confidence": 0.9,
        "classified_at": 1_700_000_000.0,
        "quality_ok": True,
        "quality_metrics": {"mean_brightness": 120.0, "brightness_std": 40.0,
                            "sharpness": 80.0, "issues": []},
    }
    me = ManifestEntry.model_validate(row_without)
    assert me.quality_metrics is not None
    assert me.quality_metrics.camera_motion is None

    row_with = {**row_without,
                "quality_metrics": {"mean_brightness": 120.0, "brightness_std": 40.0,
                                    "sharpness": 80.0, "camera_motion": 7.35,
                                    "issues": ["camera_shake"]},
                "quality_ok": False}
    me2 = ManifestEntry.model_validate(row_with)
    assert me2.quality_metrics.camera_motion == 7.35
    assert "camera_shake" in me2.quality_metrics.issues


def test_face_detector_falls_back_when_mediapipe_missing(monkeypatch):
    """没有 mediapipe 包时 get_face_detector 不崩，回落到 Haar 兜底。"""
    import shot_classify
    import threading as _threading
    # Reset thread-local + global backend so this test forces re-detection
    # (face detector is now thread-local; mediapipe FaceDetector is not
    # thread-safe, each worker thread builds its own instance).
    monkeypatch.setattr(shot_classify, "_face_tls", _threading.local())
    monkeypatch.setattr(shot_classify, "_face_backend_global", None)

    # 让 `import mediapipe` 抛 ImportError
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "mediapipe" or name.startswith("mediapipe."):
            raise ImportError("mediapipe not installed (test)")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    det, backend = shot_classify.get_face_detector()
    # 回落到 Haar（opencv-python 自带），不允许炸也不允许变 mediapipe
    assert backend in ("haar", "yolo_face", "none")
    assert backend != "mediapipe"


if __name__ == "__main__":
    tests = [test_defaults_are_spec,
             test_borderline_shot_passed_by_default_blocked_by_strict,
             test_spec_mode_values_match_delivery_v1,
             test_strict_mode_values,
             test_manifestentry_accepts_new_bbox_fields,
             test_camera_shake_threshold_logic,
             test_spec_mode_includes_max_camera_motion,
             test_manifestentry_accepts_camera_motion_field]
    ok = 0
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
            ok += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n  {ok} pass / {len(tests) - ok} fail")
    sys.exit(0 if ok == len(tests) else 1)
