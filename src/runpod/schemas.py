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
    direct:       str
    literary:     str
    direction:    str
    situational:  str


class FacialComponents(_Base):
    eyes:           str
    eyebrows:       str
    mouth:          str
    jaw:            str
    gaze_direction: str
    head_pose:      str


class FacialAttributes(_Base):
    apparent_gender:    str
    apparent_age_range: str
    glasses:            bool
    facial_hair:        str
    head_covering:      str
    mask:               bool
    makeup_visible:     bool
    distinctive_notes:  str = ""


class BlendshapeHints(_Base):
    brow_raise_inner:   BlendshapeScale
    brow_raise_outer:   BlendshapeScale
    brow_furrow:        BlendshapeScale
    eye_widen:          BlendshapeScale
    eye_squint:         BlendshapeScale
    eye_blink_state:    BlinkState
    cheek_raise:        BlendshapeScale
    nose_wrinkle:       BlendshapeScale
    upper_lip_raise:    BlendshapeScale
    lip_corner_pull:    BlendshapeScale
    lip_corner_depress: BlendshapeScale
    lip_tighten:        BlendshapeScale
    lip_part:           BlendshapeScale
    jaw_clench:         BlendshapeScale
    jaw_drop:           BlendshapeScale


class FaceAnalysis(_Base):
    face_clearly_visible:         bool
    face_size_ratio:              float = Field(ge=0.0, le=1.0)
    primary_emotion:              Emotion
    secondary_emotion:            Optional[Emotion] = None
    valence:                      float = Field(ge=-1.0, le=1.0)
    arousal:                      float = Field(ge=0.0, le=1.0)
    intensity:                    float = Field(ge=0.0, le=1.0)
    expression_caption:           str
    alternative_captions:         AlternativeCaptions
    facial_components:            FacialComponents
    facial_attributes:            FacialAttributes
    temporal_change:              TemporalChange
    micro_expression:             bool
    observable_blendshape_hints:  BlendshapeHints
    expression_confidence:        float = Field(ge=0.0, le=1.0)


# ══════════════════════════════════════════════════════════════
# body_analysis
# ══════════════════════════════════════════════════════════════

class ActionQuality(_Base):
    intensity: ActionIntensity
    tone:      ActionTone
    tempo:     ActionTempo


class KinematicsHint(_Base):
    trajectory:     KinTrajectory
    periodicity:    KinPeriodicity
    symmetry:       KinSymmetry
    duration_class: KinDuration


class UpperBodyDetail(_Base):
    head:      str
    neck:      str
    shoulders: str
    arms:      str
    hands:     str
    torso:     str
    posture:   str


class BodyInteraction(_Base):
    count:                        InteractionCount
    contact:                      InteractionContact
    relation:                     InteractionRelation
    interacts_with_person_index:  list[int] = Field(default_factory=list)


class BodyAnalysis(_Base):
    body_clearly_visible:    bool
    shot_frame_of_body:      BodyShotFrame
    visible_body_parts:      list[str]
    motion_caption:          str
    alternative_captions:    AlternativeCaptions
    action_primary:          str                    # taxonomy leaf 或 "other/xxx"
    action_quality:          ActionQuality
    body_focus:              BodyFocus
    kinematics_hint:         KinematicsHint
    upper_body_detail:       UpperBodyDetail
    gesture_detail:          Optional[str] = None
    hands_visible:           bool
    interaction:             BodyInteraction
    motion_confidence:       float = Field(ge=0.0, le=1.0)


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
# 两轮推理的子 schema（docs/problem/01_stage5_output_truncation.md 方案 E）
# Round 1：body + scene + meta；Round 2：face only，按 person_index 合并。
# 每轮 output 预算 ≤ 6000 token，避免 max_tokens 截断。
# ══════════════════════════════════════════════════════════════

class Round1Person(_Base):
    person_index:     int = Field(ge=0)
    spatial_position: SpatialPosition
    body_analysis:    Optional[BodyAnalysis] = None


class Round1Label(_Base):
    shot_id:          str
    source_movie:     str
    shot_context:     ShotContext
    persons:          list[Round1Person]
    interaction:      ShotInteraction
    quality_flags:    QualityFlags
    usability_score:  UsabilityScore
    exclusion_reason: Optional[str] = None


class Round2Person(_Base):
    person_index:  int = Field(ge=0)
    face_analysis: Optional[FaceAnalysis] = None


class Round2Label(_Base):
    persons: list[Round2Person]


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
