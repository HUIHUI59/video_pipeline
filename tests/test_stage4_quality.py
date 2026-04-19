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


def _threshold_decide(mean_b, std_b, sharp, thresholds):
    """复刻 shot_classify.classify_one 里的 issue 分类逻辑（Phase 1.1 新版）。
    把被测逻辑抽出来单测，避免跑整条 OpenCV/YOLO 链。"""
    th_min_b = thresholds["min_brightness"]
    th_max_b = thresholds["max_brightness"]
    th_min_c = thresholds["min_contrast"]
    th_min_s = thresholds["min_sharpness"]
    issues = []
    if mean_b < th_min_b: issues.append("too_dark")
    if mean_b > th_max_b: issues.append("too_bright")
    if std_b  < th_min_c: issues.append("low_contrast")
    if sharp  < th_min_s: issues.append("blurry")
    return issues, len(issues) == 0


def _defaults():
    from shot_classify import (
        QUALITY_MIN_BRIGHTNESS, QUALITY_MAX_BRIGHTNESS,
        QUALITY_MIN_CONTRAST, QUALITY_MIN_SHARPNESS,
        QUALITY_SPEC_MODE,
    )
    return {
        "min_brightness": QUALITY_MIN_BRIGHTNESS,
        "max_brightness": QUALITY_MAX_BRIGHTNESS,
        "min_contrast":   QUALITY_MIN_CONTRAST,
        "min_sharpness":  QUALITY_MIN_SHARPNESS,
    }, QUALITY_SPEC_MODE


def test_defaults_are_strict():
    strict, spec = _defaults()
    # 规范默认更宽松：下界更小，上界更大
    assert strict["min_brightness"] >= spec["min_brightness"]
    assert strict["max_brightness"] <= spec["max_brightness"]
    assert strict["min_contrast"]   >= spec["min_contrast"]
    assert strict["min_sharpness"]  >= spec["min_sharpness"]


def test_borderline_shot_blocked_in_strict_passed_in_spec():
    """亮度/对比度/清晰度都处于 strict 严、spec 松之间的边界值。
    严格模式 quality_ok=False；规范模式 quality_ok=True。"""
    strict, spec = _defaults()
    # 挑每个指标在"介于 spec 和 strict 之间"的值
    mean_b = 20.0   # spec 12..242 allows；strict 25..230 blocks (too_dark)
    std_b  = 10.0   # spec 5 allows；strict 15 blocks (low_contrast)
    sharp  = 30.0   # spec 15 allows；strict 50 blocks (blurry)

    issues_strict, ok_strict = _threshold_decide(mean_b, std_b, sharp, strict)
    issues_spec,   ok_spec   = _threshold_decide(mean_b, std_b, sharp, spec)
    assert not ok_strict, f"strict should reject: {issues_strict}"
    assert ok_spec, f"spec should accept: {issues_spec}"


def test_spec_mode_values_match_delivery_v1():
    """Regression lock：spec mode 是 12/242/5/15（04_shot_classify.md 规范默认值）。"""
    _, spec = _defaults()
    assert spec["min_brightness"] == 12.0
    assert spec["max_brightness"] == 242.0
    assert spec["min_contrast"]   == 5.0
    assert spec["min_sharpness"]  == 15.0


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


if __name__ == "__main__":
    tests = [test_defaults_are_strict,
             test_borderline_shot_blocked_in_strict_passed_in_spec,
             test_spec_mode_values_match_delivery_v1,
             test_manifestentry_accepts_new_bbox_fields]
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
