"""
src/runpod/schemas.py
════════════════════════════════════════════════════════════════
Pydantic v2 模型，严格对齐 docs/labelingStandards/json_schema_integrated.md

用途：
  1. pod_runner.py 里把 ShotLabel.model_json_schema() 传给 vLLM 的
     GuidedDecodingParams(json=...) 做 schema-constrained decoding
  2. download.py 拉回来的每份 JSON 先 ShotLabel.model_validate(...)
     校验，不合规的报到日志
  3. upload.py 用 ManifestEntry 校验 Stage 4 manifest 每一行

自检：
  python -m src.runpod.schemas
  预期：遍历 docs/labelingStandards/examples/*.json 全部通过
════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


# ══════════════════════════════════════════════════════════════
# 枚举（来自 standards 文档）
# ══════════════════════════════════════════════════════════════

ShotCategory = Literal["single", "dominant", "multi", "wide", "landscape"]

ShotType = Literal[
    "close-up", "medium close-up", "medium",
    "medium long", "wide", "extreme wide",
]

Emotion = Literal[
    "anger", "sadness", "joy", "fear", "surprise",
    "disgust", "contempt", "neutral", "complex",
]

SpatialPosition = Literal["center", "left", "right", "background"]

BodyShotFrame = Literal[
    "close_face", "bust", "half_body",
    "three_quarter", "full_body", "wide",
]

InteractionCount   = Literal["solo", "dyadic", "triadic", "crowd"]
InteractionContact = Literal["none", "incidental", "sustained"]
InteractionRelation = Literal[
    "parallel", "coordinated", "opposing", "hierarchical",
]

TemporalChange = Literal[
    "static", "building", "peak_then_release", "transition", "rapid_micro",
]

ActionIntensity = Literal["low", "mid", "high"]
# tone/tempo are free-form strings — the spec lists preferred values (relaxed/
# tense/controlled/contemplative for tone; sustained/punctuated/accelerating/
# decelerating for tempo) but worked examples also use values like "slow",
# "gentle", etc. VLM output reflects that.
ActionTone      = str
ActionTempo     = str

BodyFocus = Literal[
    "upper_body", "hands", "torso", "posture",
    "gesture", "full_body", "lower_body", "face_and_gaze",
]

# trajectory / duration_class are free-form strings (same reasoning as
# tone/tempo — worked examples drift: "stationary", "curved", "intermittent",
# "transitional" all appear). The spec lists preferred values but the schema
# tolerates any string so VLM output isn't rejected for vocabulary drift.
KinTrajectory   = str
KinPeriodicity  = Literal["periodic", "non_periodic"]
KinSymmetry     = Literal["bilateral_symmetric", "bilateral_asymmetric", "axial"]
KinDuration     = str

BlendshapeScale = Literal["none", "slight", "medium", "strong", "unknown"]
BlinkState      = Literal[
    "open", "half", "closed", "rapid_blink", "unknown",
    "slow_blink",  # seen in examples
]

Occlusion = Literal["none", "partial", "heavy"]
# Lighting is a free-form string — examples also use "acceptable".
Lighting  = str


# ══════════════════════════════════════════════════════════════
# 松散 base（允许 extra 字段，便于 VLM 输出演进）
# ══════════════════════════════════════════════════════════════

class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ══════════════════════════════════════════════════════════════
# shot_context
# ══════════════════════════════════════════════════════════════

class SceneContext(_Base):
    visible_setting:       str
    narrative_situation:   Optional[str]  = None
    narrative_confidence:  float = Field(0.0, ge=0.0, le=1.0)


class ShotContext(_Base):
    shot_type:             ShotType
    shot_emotion_summary:  str
    shot_motion_summary:   str
    scene_context:         SceneContext


# ══════════════════════════════════════════════════════════════
# face_analysis
# ══════════════════════════════════════════════════════════════

class AlternativeCaptions(_Base):
    # 任何单项都可能 null（当 face/body 的 clearly_visible=False 时，整个
    # alt_captions 对象也可能出现 dict-with-null-subfields 形式）
    direct:       Optional[str] = None
    literary:     Optional[str] = None
    direction:    Optional[str] = None
    situational:  Optional[str] = None


class FacialComponents(_Base):
    eyes:           Optional[str] = None
    eyebrows:       Optional[str] = None
    mouth:          Optional[str] = None
    jaw:            Optional[str] = None
    gaze_direction: Optional[str] = None
    head_pose:      Optional[str] = None


class FacialAttributes(_Base):
    apparent_gender:    Optional[str]  = None
    apparent_age_range: Optional[str]  = None
    glasses:            Optional[bool] = None
    facial_hair:        Optional[str]  = None
    head_covering:      Optional[str]  = None
    mask:               Optional[bool] = None
    makeup_visible:     Optional[bool] = None
    distinctive_notes:  Optional[str]  = ""


class BlendshapeHints(_Base):
    # 面部不可见时，整个 blendshape_hints 对象的子字段会是 null
    brow_raise_inner:   Optional[BlendshapeScale] = None
    brow_raise_outer:   Optional[BlendshapeScale] = None
    brow_furrow:        Optional[BlendshapeScale] = None
    eye_widen:          Optional[BlendshapeScale] = None
    eye_squint:         Optional[BlendshapeScale] = None
    eye_blink_state:    Optional[BlinkState]      = None
    cheek_raise:        Optional[BlendshapeScale] = None
    nose_wrinkle:       Optional[BlendshapeScale] = None
    upper_lip_raise:    Optional[BlendshapeScale] = None
    lip_corner_pull:    Optional[BlendshapeScale] = None
    lip_corner_depress: Optional[BlendshapeScale] = None
    lip_tighten:        Optional[BlendshapeScale] = None
    lip_part:           Optional[BlendshapeScale] = None
    jaw_clench:         Optional[BlendshapeScale] = None
    jaw_drop:           Optional[BlendshapeScale] = None


class FaceAnalysis(_Base):
    # face_clearly_visible 是 gate；gate=False 时下面字段都可为 null。
    # face_size_ratio 按规范即使不可见也要写，但 delivery_v1 的部分 example
    # 里也为 null，保持 Optional 最稳。
    face_clearly_visible:         bool
    face_size_ratio:              Optional[float] = Field(None, ge=0.0, le=1.0)
    primary_emotion:              Optional[Emotion] = None
    secondary_emotion:            Optional[Emotion] = None
    valence:                      Optional[float] = Field(None, ge=-1.0, le=1.0)
    arousal:                      Optional[float] = Field(None, ge=0.0, le=1.0)
    intensity:                    Optional[float] = Field(None, ge=0.0, le=1.0)
    expression_caption:           Optional[str] = None
    alternative_captions:         Optional[AlternativeCaptions] = None
    facial_components:            Optional[FacialComponents]    = None
    facial_attributes:            Optional[FacialAttributes]    = None
    temporal_change:              Optional[TemporalChange] = None
    micro_expression:             Optional[bool] = None
    observable_blendshape_hints:  Optional[BlendshapeHints] = None
    expression_confidence:        Optional[float] = Field(None, ge=0.0, le=1.0)


# ══════════════════════════════════════════════════════════════
# body_analysis
# ══════════════════════════════════════════════════════════════

class ActionQuality(_Base):
    intensity: Optional[ActionIntensity] = None
    tone:      Optional[ActionTone]      = None
    tempo:     Optional[ActionTempo]     = None


class KinematicsHint(_Base):
    trajectory:     Optional[KinTrajectory]  = None
    periodicity:    Optional[KinPeriodicity] = None
    symmetry:       Optional[KinSymmetry]    = None
    duration_class: Optional[KinDuration]    = None


class UpperBodyDetail(_Base):
    head:      Optional[str] = None
    neck:      Optional[str] = None
    shoulders: Optional[str] = None
    arms:      Optional[str] = None
    hands:     Optional[str] = None
    torso:     Optional[str] = None
    posture:   Optional[str] = None


class BodyInteraction(_Base):
    count:                        Optional[InteractionCount]    = None
    contact:                      Optional[InteractionContact]  = None
    relation:                     Optional[InteractionRelation] = None
    interacts_with_person_index:  list[int] = Field(default_factory=list)


class BodyAnalysis(_Base):
    # body_clearly_visible 是 gate；gate=False 时下面字段可为 null。
    # shot_frame_of_body 规范文本要求非空，但官方 example 里遇到不可见时也出过
    # null，放宽 Optional 最稳。
    body_clearly_visible:    bool
    shot_frame_of_body:      Optional[BodyShotFrame] = None
    visible_body_parts:      Optional[list[str]]     = None
    motion_caption:          Optional[str] = None
    alternative_captions:    Optional[AlternativeCaptions] = None
    action_primary:          Optional[str] = None       # taxonomy leaf 或 "other/xxx"
    action_quality:          Optional[ActionQuality] = None
    body_focus:              Optional[BodyFocus] = None
    kinematics_hint:         Optional[KinematicsHint] = None
    upper_body_detail:       Optional[UpperBodyDetail] = None
    gesture_detail:          Optional[str] = None
    hands_visible:           Optional[bool] = None
    interaction:             Optional[BodyInteraction] = None
    motion_confidence:       Optional[float] = Field(None, ge=0.0, le=1.0)


# ══════════════════════════════════════════════════════════════
# person / interaction / quality / usability / meta
# ══════════════════════════════════════════════════════════════

class Person(_Base):
    person_index:     int = Field(ge=0)
    spatial_position: SpatialPosition
    face_analysis:    Optional[FaceAnalysis] = None
    body_analysis:    Optional[BodyAnalysis] = None


class ShotInteraction(_Base):
    count:    InteractionCount
    contact:  InteractionContact
    relation: InteractionRelation


class QualityFlags(_Base):
    face_clearly_visible: bool
    body_clearly_visible: bool
    motion_blur:          bool
    occlusion:            Occlusion
    lighting:             Lighting
    camera_stable:        bool
    frame_sampling_ok:    bool
    vlm_confidence:       float = Field(ge=0.0, le=1.0)


class UsabilityScore(_Base):
    face:   float = Field(ge=0.0, le=1.0)
    motion: float = Field(ge=0.0, le=1.0)


class Meta(_Base):
    vlm_model:     str
    vlm_version:   str
    frames_used:   int = Field(ge=1)
    infer_time_ms: int = Field(ge=0)


# ══════════════════════════════════════════════════════════════
# 顶层：ShotLabel
# ══════════════════════════════════════════════════════════════

class ShotLabel(_Base):
    shot_id:          str
    source_movie:     str
    shot_context:     ShotContext
    persons:          list[Person]
    interaction:      ShotInteraction
    quality_flags:    QualityFlags
    usability_score:  UsabilityScore
    exclusion_reason: Optional[str] = None
    meta:             Meta


# ══════════════════════════════════════════════════════════════
# 三轮推理的子 schema（2026-04-20 架构演进）
#   Round 1 BODY  ：只出 persons[].body_analysis，每人独立 dict
#   Round 2 FACE  ：只出 persons[].face_analysis，附 face crop 高清细节
#   Round 3 SCENE ：只出 shot_context + interaction + quality_flags +
#                   usability_score（整体场景元数据）
# 合并由 pod_runner.main 负责；合并后跑 ShotLabel.model_validate + 16
# 条 ShotValidator 业务规则。
# ══════════════════════════════════════════════════════════════

class Round1BodyPerson(_Base):
    """Round 1 每人输出：身体分析 + 位置信息（不含脸）。"""
    person_index:     int = Field(ge=0)
    spatial_position: SpatialPosition
    body_analysis:    Optional[BodyAnalysis] = None


class Round1BodyLabel(_Base):
    """Round 1 整体输出：只有 persons 列表。"""
    persons: list[Round1BodyPerson]


class Round2Person(_Base):
    person_index:  int = Field(ge=0)
    face_analysis: Optional[FaceAnalysis] = None


class Round2Label(_Base):
    persons: list[Round2Person]


class Round3ShotLabel(_Base):
    """Round 3：shot 级元数据。VLM 只看视频 + R1 给出的 person 概览。"""
    shot_context:     ShotContext
    interaction:      ShotInteraction
    quality_flags:    QualityFlags
    usability_score:  UsabilityScore
    exclusion_reason: Optional[str] = None


# ══════════════════════════════════════════════════════════════
# Stage 4 manifest entry（upload.py 用它校验 manifest 每一行）
# ══════════════════════════════════════════════════════════════

class ManifestQuality(_Base):
    mean_brightness: float
    brightness_std:  float
    sharpness:       float
    issues:          list[str] = Field(default_factory=list)


class ManifestEntry(_Base):
    shot_id:               str
    source_movie:          str
    path:                  str
    num_people:            int = Field(ge=0)
    shot_category:         ShotCategory
    duration_sec:          float = Field(ge=0.0)
    width:                 int = Field(ge=0)
    height:                int = Field(ge=0)
    fps:                   float = Field(ge=0.0)
    largest_subject_ratio: float = Field(ge=0.0, le=1.0)
    classifier_confidence: float = Field(ge=0.0, le=1.0)
    classified_at:         float
    # ── v2 新增字段（旧 manifest 兼容：Optional, 默认 None）─────────
    num_faces:             Optional[int]              = None
    largest_face_ratio:    Optional[float]            = None
    quality_ok:            Optional[bool]             = None
    quality_metrics:       Optional[ManifestQuality]  = None
    # ── v3 新增（Stage 5 推断 shot_frame_of_body 的辅助信号）───────
    largest_subject_bbox:            Optional[list[float]] = None   # [x1,y1,x2,y2] 归一化 [0,1]
    largest_subject_vertical_center: Optional[float]       = None   # 纵向中心归一化 [0,1]


# ══════════════════════════════════════════════════════════════
# 自检入口
# ══════════════════════════════════════════════════════════════

def _self_test() -> int:
    """遍历 docs/labelingStandards/examples/*.json，全部过校验则返回 0。"""
    import json, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "docs" / "labelingStandards" / "examples"
    if not root.exists():
        print(f"[skip] {root} 不存在", file=sys.stderr)
        return 0
    ok = fail = 0
    for f in sorted(root.glob("*.json")):
        try:
            ShotLabel.model_validate(json.loads(f.read_text(encoding="utf-8")))
            print(f"  ✅  {f.name}")
            ok += 1
        except Exception as e:
            print(f"  ❌  {f.name}: {e}")
            fail += 1
    print(f"\n  {ok} pass / {fail} fail")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
