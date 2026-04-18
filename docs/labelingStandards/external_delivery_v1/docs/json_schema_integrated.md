# Integrated JSON Output Schema -- Official Specification

**Version**: 1.0  
**Date**: 2026-04-17  
**Project**: Video_DB_Face  
**Status**: AUTHORITATIVE -- all validators and VLM prompts derive from this document.

**References**:
- `external_spec_face_EN.md` -- face_analysis field details and prompt template
- `external_spec_body_EN.md` -- body_analysis field details and prompt template
- `motion_taxonomy.yaml` -- canonical action vocabulary
- `motion_synonyms.yaml` -- normalization and forbidden-term list
- Few-shot examples: `docs/vlm_prompts/examples/`

---

## 1. Overview

This document defines the **complete, unified JSON output schema** for per-shot VLM inference in the Video_DB_Face pipeline. Each shot produces exactly one JSON object containing:

- **Shot-level context** (shot_context, interaction, quality_flags, meta)
- **Per-person analysis** (persons[], each with face_analysis + body_analysis)
- **Quality gating fields** (usability_score, exclusion_reason)

The schema is consumed by:
1. **vLLM guided_json** -- constrains VLM output at decode time
2. **Pydantic v2 validators** -- post-inference schema + cross-field validation
3. **Layer-3 curation scripts** -- filtering, clustering, tier assignment
4. **Layer-4 training pipelines** -- LoRA text conditioning, ControlNet signal extraction

---

## 2. Full JSON Tree

```
{
  "shot_id":        string,
  "source_movie":   string,

  "shot_context": {
    "shot_type":             enum,
    "shot_emotion_summary":  string,
    "shot_motion_summary":   string,
    "scene_context": {
      "visible_setting":       string,
      "narrative_situation":   string | null,
      "narrative_confidence":  float [0,1]
    }
  },

  "persons": [                          // array, 1-3 entries
    {
      "person_index":     int,
      "spatial_position": enum,

      "face_analysis": {                // null if face not visible
        "face_clearly_visible":  bool,
        "face_size_ratio":       float [0,1],

        "primary_emotion":       enum (9-class),
        "secondary_emotion":     enum (9-class) | null,
        "valence":               float [-1,+1],
        "arousal":               float [0,1],
        "intensity":             float [0,1],

        "expression_caption":    string (50-120 words),
        "alternative_captions": {
          "direct":      string (20-40 words),
          "literary":    string (30-60 words),
          "direction":   string (30-50 words),
          "situational": string (30-60 words)
        },

        "facial_components": {
          "eyes":           string,
          "eyebrows":       string,
          "mouth":          string,
          "jaw":            string,
          "gaze_direction": enum,
          "head_pose":      enum
        },

        "facial_attributes": {
          "apparent_gender":    enum,
          "apparent_age_range": enum,
          "glasses":            bool,
          "facial_hair":        enum,
          "head_covering":      enum,
          "mask":               bool,
          "makeup_visible":     bool,
          "distinctive_notes":  string
        },

        "temporal_change":  enum,
        "micro_expression": bool,

        "observable_blendshape_hints": {
          "brow_raise_inner":   enum (5-scale),
          "brow_raise_outer":   enum (5-scale),
          "brow_furrow":        enum (5-scale),
          "eye_widen":          enum (5-scale),
          "eye_squint":         enum (5-scale),
          "eye_blink_state":    enum (special),
          "cheek_raise":        enum (5-scale),
          "nose_wrinkle":       enum (5-scale),
          "upper_lip_raise":    enum (5-scale),
          "lip_corner_pull":    enum (5-scale),
          "lip_corner_depress": enum (5-scale),
          "lip_tighten":        enum (5-scale),
          "lip_part":           enum (5-scale),
          "jaw_clench":         enum (5-scale),
          "jaw_drop":           enum (5-scale)
        },

        "expression_confidence": float [0,1]
      },

      "body_analysis": {                // null if body not visible
        "body_clearly_visible":  bool,

        "shot_frame_of_body":    enum,
        "visible_body_parts":    string[],

        "motion_caption":        string (50-180 words),
        "alternative_captions": {
          "direct":      string (20-40 words),
          "literary":    string (30-60 words),
          "direction":   string (30-50 words),
          "situational": string (30-60 words)
        },

        "action_primary":  string (taxonomy leaf or other/<word>),
        "action_quality": {
          "intensity": enum,
          "tone":      enum,
          "tempo":     enum
        },
        "body_focus": enum,

        "kinematics_hint": {
          "trajectory":     enum,
          "periodicity":    enum,
          "symmetry":       enum,
          "duration_class": enum
        },

        "upper_body_detail": {
          "head":      string,
          "neck":      string,
          "shoulders": string,
          "arms":      string,
          "hands":     string,
          "torso":     string,
          "posture":   enum
        },

        "gesture_detail":  string,
        "hands_visible":   bool,

        "interaction": {
          "count":    enum,
          "contact":  enum,
          "relation": enum,
          "interacts_with_person_index": int[]
        },

        "motion_confidence": float [0,1]
      }
    }
  ],

  "interaction": {
    "count":    enum,
    "contact":  enum,
    "relation": enum
  },

  "quality_flags": {
    "face_clearly_visible":  bool,
    "body_clearly_visible":  bool,
    "motion_blur":           bool,
    "occlusion":             enum,
    "lighting":              enum,
    "camera_stable":         bool,
    "frame_sampling_ok":     bool,
    "vlm_confidence":        float [0,1]
  },

  "usability_score": {
    "face":   float [0,1],
    "motion": float [0,1]
  },

  "exclusion_reason": string | null,

  "meta": {
    "vlm_model":     string,
    "vlm_version":   string,
    "frames_used":   int,
    "infer_time_ms": int
  }
}
```

---

## 3. Field-by-Field Reference Table

### 3.1 Top-Level Fields

| Field Path | Type | Required | Range/Enum | NULL Condition | Dependencies | Notes |
|---|---|---|---|---|---|---|
| `shot_id` | string | yes | -- | never | -- | Unique identifier from manifest |
| `source_movie` | string | yes | -- | never | -- | Movie title from manifest |

### 3.2 shot_context

| Field Path | Type | Required | Range/Enum | NULL Condition | Dependencies | Notes |
|---|---|---|---|---|---|---|
| `shot_context.shot_type` | enum | yes | See Sec 4 | never | -- | VLM-verified final value |
| `shot_context.shot_emotion_summary` | string | yes | 1-2 sentences | never | -- | Scene-level mood |
| `shot_context.shot_motion_summary` | string | yes | 1-2 sentences | never | -- | Scene-level body motion |
| `shot_context.scene_context.visible_setting` | string | yes | -- | never | -- | Physical background description |
| `shot_context.scene_context.narrative_situation` | string | no | -- | `narrative_confidence < 0.3` | `narrative_confidence` | Story context; forced null if low confidence |
| `shot_context.scene_context.narrative_confidence` | float | yes | [0, 1] | never | -- | Confidence in narrative interpretation |

### 3.3 persons[] Envelope

| Field Path | Type | Required | Range/Enum | NULL Condition | Dependencies | Notes |
|---|---|---|---|---|---|---|
| `persons[].person_index` | int | yes | 0-based | never | -- | Order: largest face first |
| `persons[].spatial_position` | enum | yes | center, left, right, background | never | -- | Position in frame |

### 3.4 face_analysis (per person)

| Field Path | Type | Required | Range/Enum | NULL Condition | Dependencies | Notes |
|---|---|---|---|---|---|---|
| `face_analysis` | object | yes | -- | `face_clearly_visible=false` -> entire block null | `face_clearly_visible` | Null the entire block, not individual fields |
| `.face_clearly_visible` | bool | yes | -- | never | -- | Gate for remaining fields |
| `.face_size_ratio` | float | yes | [0, 1] | never | -- | Face bbox area / frame area |
| `.primary_emotion` | enum | yes | 9-class | never | -- | Always present when face visible |
| `.secondary_emotion` | enum | no | 9-class | When single clear emotion | -- | null if primary alone suffices |
| `.valence` | float | yes | [-1, +1] | `expression_confidence < 0.3` | `expression_confidence` | Negative=unpleasant, positive=pleasant |
| `.arousal` | float | yes | [0, 1] | `expression_confidence < 0.3` | `expression_confidence` | 0=calm, 1=excited |
| `.intensity` | float | yes | [0, 1] | `expression_confidence < 0.3` | `expression_confidence` | 0=subtle, 1=extreme |
| `.expression_caption` | string | yes | 50-120 words | `expression_confidence < 0.3` | `expression_confidence` | Primary training text signal |
| `.alternative_captions.direct` | string | yes | 20-40 words | `expression_confidence < 0.3` | -- | Plain descriptive |
| `.alternative_captions.literary` | string | yes | 30-60 words | `expression_confidence < 0.3` | -- | Cinematic/evocative |
| `.alternative_captions.direction` | string | yes | 30-50 words | `expression_confidence < 0.3` | -- | Actor/director notes |
| `.alternative_captions.situational` | string | yes | 30-60 words | `expression_confidence < 0.3` | -- | Internal state |
| `.facial_components.eyes` | string | yes | -- | `expression_confidence < 0.3` | -- | Shape, openness, tension |
| `.facial_components.eyebrows` | string | yes | -- | `expression_confidence < 0.3` | -- | Shape, height, asymmetry |
| `.facial_components.mouth` | string | yes | -- | `expression_confidence < 0.3` | -- | Shape, openness |
| `.facial_components.jaw` | string | yes | -- | `expression_confidence < 0.3` | -- | Tension, drop |
| `.facial_components.gaze_direction` | enum | yes | See Sec 4 | `expression_confidence < 0.3` | -- | camera/left/right/up/down/averted/closed |
| `.facial_components.head_pose` | enum | yes | See Sec 4 | `expression_confidence < 0.3` | -- | frontal/3q_left/3q_right/profile_left/profile_right/tilted_up/tilted_down |
| `.facial_attributes.apparent_gender` | enum | yes | male, female, ambiguous | never (when face visible) | -- | Visual observation only |
| `.facial_attributes.apparent_age_range` | enum | yes | See Sec 4 | never (when face visible) | -- | Visual estimate |
| `.facial_attributes.glasses` | bool | yes | -- | never (when face visible) | -- | -- |
| `.facial_attributes.facial_hair` | enum | yes | none, stubble, beard, mustache | never (when face visible) | -- | -- |
| `.facial_attributes.head_covering` | enum | yes | none, hat, hood, scarf | never (when face visible) | -- | -- |
| `.facial_attributes.mask` | bool | yes | -- | never (when face visible) | -- | -- |
| `.facial_attributes.makeup_visible` | bool | yes | -- | never (when face visible) | -- | -- |
| `.facial_attributes.distinctive_notes` | string | yes | -- | never (when face visible) | -- | Scars, birthmarks, lighting notes |
| `.temporal_change` | enum | yes | See Sec 4 | `expression_confidence < 0.3` | -- | Expression dynamics |
| `.micro_expression` | bool | yes | -- | `expression_confidence < 0.3` | -- | true -> auto Tier 3 candidate |
| `.observable_blendshape_hints.*` (14 fields) | enum | yes | See Sec 4 | `expression_confidence < 0.3` | -- | All 14 must be filled |
| `.observable_blendshape_hints.eye_blink_state` | enum | yes | open, half, closed, rapid_blink, unknown | `expression_confidence < 0.3` | -- | Special scale (not 5-scale) |
| `.expression_confidence` | float | yes | [0, 1] | never (when face visible) | -- | VLM self-assessed confidence |

### 3.5 body_analysis (per person)

| Field Path | Type | Required | Range/Enum | NULL Condition | Dependencies | Notes |
|---|---|---|---|---|---|---|
| `body_analysis` | object | yes | -- | `body_clearly_visible=false` -> entire block null | `body_clearly_visible` | Null the entire block |
| `.body_clearly_visible` | bool | yes | -- | never | -- | Gate for remaining fields |
| `.shot_frame_of_body` | enum | yes | See Sec 4 | never (when body visible) | -- | How much body is in frame |
| `.visible_body_parts` | string[] | yes | See Sec 4 | never (when body visible) | `shot_frame_of_body` | Only parts actually visible |
| `.motion_caption` | string | yes | 50-180 words | `motion_confidence < 0.3` | `motion_confidence` | Primary training text. Body ONLY. |
| `.alternative_captions.direct` | string | yes | 20-40 words | `motion_confidence < 0.3` | -- | Plain descriptive |
| `.alternative_captions.literary` | string | yes | 30-60 words | `motion_confidence < 0.3` | -- | Cinematic/evocative |
| `.alternative_captions.direction` | string | yes | 30-50 words | `motion_confidence < 0.3` | -- | Actor/choreography notes |
| `.alternative_captions.situational` | string | yes | 30-60 words | `motion_confidence < 0.3` | -- | Internal state |
| `.action_primary` | string | yes | taxonomy leaf or `other/<word>` | `motion_confidence < 0.3` | -- | See motion_taxonomy.yaml |
| `.action_quality.intensity` | enum | yes | low, mid, high | `motion_confidence < 0.3` | -- | Energetic magnitude |
| `.action_quality.tone` | enum | yes | relaxed, tense, controlled, contemplative | `motion_confidence < 0.3` | -- | Internal tension register |
| `.action_quality.tempo` | enum | yes | sustained, punctuated, accelerating, decelerating | `motion_confidence < 0.3` | -- | Temporal rhythm |
| `.body_focus` | enum | yes | See Sec 4 | `motion_confidence < 0.3` | -- | Region driving the action |
| `.kinematics_hint.trajectory` | enum | yes | linear, arc, circular, erratic, static | `motion_confidence < 0.3` | -- | Spatial path |
| `.kinematics_hint.periodicity` | enum | yes | periodic, non_periodic | `motion_confidence < 0.3` | -- | Cyclic repetition |
| `.kinematics_hint.symmetry` | enum | yes | bilateral_symmetric, bilateral_asymmetric, axial | `motion_confidence < 0.3` | -- | L-R symmetry |
| `.kinematics_hint.duration_class` | enum | yes | onset_only, ongoing, peak_then_release | `motion_confidence < 0.3` | -- | Temporal profile |
| `.upper_body_detail.head` | string | yes | -- | `motion_confidence < 0.3` | -- | Tilt, turn direction. NO facial expression. |
| `.upper_body_detail.neck` | string | yes | -- | `motion_confidence < 0.3` | -- | Extension, tension |
| `.upper_body_detail.shoulders` | string | yes | -- | `motion_confidence < 0.3` | -- | Level, rise/drop |
| `.upper_body_detail.arms` | string | yes | -- | `motion_confidence < 0.3` | -- | "not visible" if below frame |
| `.upper_body_detail.hands` | string | yes | -- | `motion_confidence < 0.3` | `hands_visible` | "not visible" if `hands_visible=false` |
| `.upper_body_detail.torso` | string | yes | -- | `motion_confidence < 0.3` | -- | Twist, lean, posture notes |
| `.upper_body_detail.posture` | enum | yes | See Sec 4 | `motion_confidence < 0.3` | -- | Canonical posture label |
| `.gesture_detail` | string | yes | -- | `motion_confidence < 0.3` | -- | Specific gestures with L/R sides |
| `.hands_visible` | bool | yes | -- | never (when body visible) | -- | Gate for hand detail |
| `.interaction.count` | enum | yes | solo, dyadic, triadic, crowd | never (when body visible) | -- | People count for this person's context |
| `.interaction.contact` | enum | yes | none, incidental, sustained | never (when body visible) | -- | Physical contact type |
| `.interaction.relation` | enum | yes | parallel, coordinated, opposing, hierarchical | never (when body visible) | -- | Motion relationship |
| `.interaction.interacts_with_person_index` | int[] | yes | -- | never (when body visible) | -- | Empty array if solo |
| `.motion_confidence` | float | yes | [0, 1] | never (when body visible) | -- | VLM self-assessed confidence |

### 3.6 Top-Level Interaction (Shot Property)

| Field Path | Type | Required | Range/Enum | NULL Condition | Dependencies | Notes |
|---|---|---|---|---|---|---|
| `interaction.count` | enum | yes | solo, dyadic, triadic, crowd | never | -- | Shot-level people count |
| `interaction.contact` | enum | yes | none, incidental, sustained | never | -- | Shot-level contact |
| `interaction.relation` | enum | yes | parallel, coordinated, opposing, hierarchical | never | -- | Shot-level relation |

### 3.7 Quality Flags, Usability, Exclusion

| Field Path | Type | Required | Range/Enum | NULL Condition | Dependencies | Notes |
|---|---|---|---|---|---|---|
| `quality_flags.face_clearly_visible` | bool | yes | -- | never | -- | Shot-level face gate |
| `quality_flags.body_clearly_visible` | bool | yes | -- | never | -- | Shot-level body gate |
| `quality_flags.motion_blur` | bool | yes | -- | never | -- | true = blur present |
| `quality_flags.occlusion` | enum | yes | none, partial, heavy | never | -- | Occlusion severity |
| `quality_flags.lighting` | enum | yes | good, low, mixed, backlit | never | -- | Lighting condition |
| `quality_flags.camera_stable` | bool | yes | -- | never | -- | false for intentional handheld too |
| `quality_flags.frame_sampling_ok` | bool | yes | -- | never | -- | false if decode errors |
| `quality_flags.vlm_confidence` | float | yes | [0, 1] | never | -- | Overall VLM confidence |
| `usability_score.face` | float | yes | [0, 1] | never | -- | Face training usability |
| `usability_score.motion` | float | yes | [0, 1] | never | -- | Body/motion training usability |
| `exclusion_reason` | string | no | -- | When shot is usable | -- | Reason for exclusion, or null |
| `meta.vlm_model` | string | yes | -- | never | -- | Model identifier |
| `meta.vlm_version` | string | yes | -- | never | -- | Model version tag |
| `meta.frames_used` | int | yes | 4 or 8 | never | -- | Frames sampled for this shot |
| `meta.infer_time_ms` | int | yes | -- | never | -- | Wall-clock inference time |

---

## 4. Enum and Allowed Values

### 4.1 primary_emotion / secondary_emotion (9-class)
```
anger | sadness | joy | fear | surprise | disgust | contempt | neutral | complex
```
`secondary_emotion` may also be `null`.

### 4.2 shot_type
```
close-up | medium close-up | medium | medium long | wide | extreme wide
```

### 4.3 spatial_position
```
center | left | right | background
```

### 4.4 gaze_direction
```
camera | left | right | up | down | averted | closed
```

### 4.5 head_pose
```
frontal | 3q_left | 3q_right | profile_left | profile_right | tilted_up | tilted_down
```

### 4.6 temporal_change
```
static | building | peak_then_release | transition | rapid_micro
```

### 4.7 Blendshape 5-scale (14 fields except eye_blink_state)
```
none | slight | medium | strong | unknown
```
Use `unknown` only when the area is occluded (hair, shadow, mask).

### 4.8 eye_blink_state (special scale)
```
open | half | closed | rapid_blink | unknown
```

### 4.9 apparent_gender
```
male | female | ambiguous
```

### 4.10 apparent_age_range
```
child | teen | young_adult | adult | middle_aged | elderly
```

### 4.11 facial_hair
```
none | stubble | beard | mustache
```

### 4.12 head_covering
```
none | hat | hood | scarf
```

### 4.13 shot_frame_of_body
```
close_face | bust | half_body | three_quarter | full_body | wide
```

### 4.14 visible_body_parts (allowed values in array)
```
head | neck | shoulders | upper_arms | forearms | hands | torso | hips | thighs | shins | feet
```

### 4.15 action_primary (taxonomy leaves -- authoritative list)

**locomotion**: `walking`, `running`, `moving`, `positioning`, `leaning`, `turning`, `stepping`, `stumbling`

**manipulation**: `reaching`, `carrying`, `holding`, `grasping`, `clasping`, `releasing`, `lifting`, `placing`, `pushing`, `pulling`, `writing`, `pouring`, `opening`, `closing`

**posture**: `standing`, `sitting`, `kneeling`, `lying`, `crouching`, `reclining`, `bending`, `straightening`

**communication**: `talking`, `gesturing`, `looking`, `focusing`, `interacting`, `engaging`, `petting`, `nodding`, `shaking_head`, `shrugging`, `pointing`, `waving`, `beckoning`, `hugging`, `clapping`

**impact**: `kicking`, `spiking`, `striking`, `throwing`, `catching`, `punching`, `slapping`, `blocking`, `jumping`, `landing`, `falling`, `diving`

**self_directed**: `adjusting`, `scratching`, `rubbing`, `wiping`, `stretching`, `yawning`, `breathing_heavy`, `fidgeting`, `smoking`, `drinking`, `eating`

**other**: Format `other/<single_lowercase_word>` (e.g., `other/rappelling`).

Category names (`locomotion`, `manipulation`, etc.) are NEVER valid output.

### 4.16 action_quality.intensity
```
low | mid | high
```

### 4.17 action_quality.tone
```
relaxed | tense | controlled | contemplative
```

### 4.18 action_quality.tempo
```
sustained | punctuated | accelerating | decelerating
```

### 4.19 body_focus
```
upper_body | hands | torso | posture | gesture | full_body | lower_body | face_and_gaze
```

### 4.20 kinematics_hint.trajectory
```
linear | arc | circular | erratic | static
```

### 4.21 kinematics_hint.periodicity
```
periodic | non_periodic
```

### 4.22 kinematics_hint.symmetry
```
bilateral_symmetric | bilateral_asymmetric | axial
```

### 4.23 kinematics_hint.duration_class
```
onset_only | ongoing | peak_then_release
```

### 4.24 interaction.count
```
solo | dyadic | triadic | crowd
```

### 4.25 interaction.contact
```
none | incidental | sustained
```

### 4.26 interaction.relation
```
parallel | coordinated | opposing | hierarchical
```

### 4.27 upper_body_detail.posture
```
upright | leaning_forward | leaning_back | slumped | turning | crouching | reclining | walking | running
```

### 4.28 quality_flags.occlusion
```
none | partial | heavy
```

### 4.29 quality_flags.lighting
```
good | low | mixed | backlit
```

---

## 5. NULL Rules and Dependencies

### 5.1 Block-Level NULL Cascades

| Condition | Effect |
|---|---|
| `face_clearly_visible = false` | Entire `face_analysis` block = `null` |
| `body_clearly_visible = false` | Entire `body_analysis` block = `null` |
| `face_size_ratio < 0.03` | Forces `face_clearly_visible = false` (cascade above) |
| Subject occupies < 10% frame height | Forces `body_clearly_visible = false` (cascade above) |

### 5.2 Confidence-Gated Partial NULLs

| Condition | Effect |
|---|---|
| `expression_confidence < 0.3` | Keep `primary_emotion` only; NULL all other face_analysis fields |
| `motion_confidence < 0.3` | Keep `shot_frame_of_body` + `visible_body_parts` only; NULL rest of body_analysis |
| `narrative_confidence < 0.3` | `narrative_situation = null` |
| `vlm_confidence < 0.3` (shot-level) | `exclusion_reason = "low_vlm_confidence"`, skip shot |

### 5.3 Cross-Field Constraints (MUST enforce)

| Rule | Enforcement |
|---|---|
| `micro_expression = true` | `temporal_change` MUST be `"rapid_micro"` |
| `interaction.count = "solo"` (top-level) | `interaction.contact` MUST be `"none"` |
| `hands_visible = false` | `upper_body_detail.hands` MUST be `"not visible"` |
| `eye_blink_state = "closed"` | `eye_widen` and `eye_squint` SHOULD be `"unknown"` (occluded) |
| `shot_frame_of_body = "close_face"` | `body_clearly_visible` SHOULD be `false` (body absent) |
| `body_focus = "full_body"` | `shot_frame_of_body` MUST NOT be `close_face` or `bust` |
| All 4 `alternative_captions` keys | MUST be non-null strings (20+ words each) when parent block is non-null |
| `expression_caption` | MUST NOT contain body/posture descriptions |
| `motion_caption` | MUST NOT contain facial expression descriptions |
| `observable_blendshape_hints` | All 14 fields + `eye_blink_state` MUST be filled (use `unknown` if occluded) |

### 5.4 NULL Conventions

- NULL an entire block: `"face_analysis": null`
- NULL a single field: use JSON `null` literal
- NEVER use empty string `""`, empty array `[]`, or the string `"null"` as NULL substitutes

---

## 6. Forbidden Values

The following camera-related terms are **FORBIDDEN** in `action_primary`, `action_quality.*`, `body_focus`, and `kinematics_hint.*` fields:

```
close-up, close up, closeup, medium shot, medium-shot, wide shot, wide-shot,
medium close-up, medium close up, extreme close-up, long shot, establishing shot,
eye-level, eye level, low angle, low-angle, high angle, high-angle, dutch,
dutch angle, overhead, bird's eye, frontal, side view, profile, profile view,
steady, steadicam, handheld, hand-held, static*, tracking, tracking shot,
pan, panning, tilt, tilting, zoom, zooming, dolly, crane, crane shot, aerial,
POV, point of view, over the shoulder, OTS, two-shot, insert, cutaway,
reverse, reaction shot
```

**Exception**: `"static"` is permitted ONLY in `kinematics_hint.trajectory`. It is forbidden in `action_primary` and all other fields.

Camera terms belong ONLY in `shot_context.shot_type`.

The post-processing normalizer (`motion_synonyms.yaml` > `camera_terms_forbidden`) auto-removes these with a WARNING log. VLM outputs containing forbidden terms in action fields are flagged for review.

---

## 7. Pydantic v2 Model Code

```python
"""
Integrated Shot Output Schema -- Pydantic v2 Models
Project: Video_DB_Face
Version: 1.0 (2026-04-17)

Usage:
    from schemas import ShotOutput
    shot = ShotOutput.model_validate_json(raw_json_string)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# Enum Definitions
# ============================================================

class PrimaryEmotion(str, Enum):
    anger = "anger"
    sadness = "sadness"
    joy = "joy"
    fear = "fear"
    surprise = "surprise"
    disgust = "disgust"
    contempt = "contempt"
    neutral = "neutral"
    complex = "complex"


class ShotType(str, Enum):
    close_up = "close-up"
    medium_close_up = "medium close-up"
    medium = "medium"
    medium_long = "medium long"
    wide = "wide"
    extreme_wide = "extreme wide"


class SpatialPosition(str, Enum):
    center = "center"
    left = "left"
    right = "right"
    background = "background"


class GazeDirection(str, Enum):
    camera = "camera"
    left = "left"
    right = "right"
    up = "up"
    down = "down"
    averted = "averted"
    closed = "closed"


class HeadPose(str, Enum):
    frontal = "frontal"
    three_q_left = "3q_left"
    three_q_right = "3q_right"
    profile_left = "profile_left"
    profile_right = "profile_right"
    tilted_up = "tilted_up"
    tilted_down = "tilted_down"


class TemporalChange(str, Enum):
    static = "static"
    building = "building"
    peak_then_release = "peak_then_release"
    transition = "transition"
    rapid_micro = "rapid_micro"


class BlendshapeScale(str, Enum):
    none = "none"
    slight = "slight"
    medium = "medium"
    strong = "strong"
    unknown = "unknown"


class EyeBlinkState(str, Enum):
    open = "open"
    half = "half"
    closed = "closed"
    rapid_blink = "rapid_blink"
    unknown = "unknown"


class ApparentGender(str, Enum):
    male = "male"
    female = "female"
    ambiguous = "ambiguous"


class ApparentAgeRange(str, Enum):
    child = "child"
    teen = "teen"
    young_adult = "young_adult"
    adult = "adult"
    middle_aged = "middle_aged"
    elderly = "elderly"


class FacialHair(str, Enum):
    none = "none"
    stubble = "stubble"
    beard = "beard"
    mustache = "mustache"


class HeadCovering(str, Enum):
    none = "none"
    hat = "hat"
    hood = "hood"
    scarf = "scarf"


class ShotFrameOfBody(str, Enum):
    close_face = "close_face"
    bust = "bust"
    half_body = "half_body"
    three_quarter = "three_quarter"
    full_body = "full_body"
    wide = "wide"


class Intensity(str, Enum):
    low = "low"
    mid = "mid"
    high = "high"


class Tone(str, Enum):
    relaxed = "relaxed"
    tense = "tense"
    controlled = "controlled"
    contemplative = "contemplative"


class Tempo(str, Enum):
    sustained = "sustained"
    punctuated = "punctuated"
    accelerating = "accelerating"
    decelerating = "decelerating"


class BodyFocus(str, Enum):
    upper_body = "upper_body"
    hands = "hands"
    torso = "torso"
    posture = "posture"
    gesture = "gesture"
    full_body = "full_body"
    lower_body = "lower_body"
    face_and_gaze = "face_and_gaze"


class Trajectory(str, Enum):
    linear = "linear"
    arc = "arc"
    circular = "circular"
    erratic = "erratic"
    static = "static"


class Periodicity(str, Enum):
    periodic = "periodic"
    non_periodic = "non_periodic"


class Symmetry(str, Enum):
    bilateral_symmetric = "bilateral_symmetric"
    bilateral_asymmetric = "bilateral_asymmetric"
    axial = "axial"


class DurationClass(str, Enum):
    onset_only = "onset_only"
    ongoing = "ongoing"
    peak_then_release = "peak_then_release"


class InteractionCount(str, Enum):
    solo = "solo"
    dyadic = "dyadic"
    triadic = "triadic"
    crowd = "crowd"


class Contact(str, Enum):
    none = "none"
    incidental = "incidental"
    sustained = "sustained"


class Relation(str, Enum):
    parallel = "parallel"
    coordinated = "coordinated"
    opposing = "opposing"
    hierarchical = "hierarchical"


class PostureLabel(str, Enum):
    upright = "upright"
    leaning_forward = "leaning_forward"
    leaning_back = "leaning_back"
    slumped = "slumped"
    turning = "turning"
    crouching = "crouching"
    reclining = "reclining"
    walking = "walking"
    running = "running"


class Occlusion(str, Enum):
    none = "none"
    partial = "partial"
    heavy = "heavy"


class Lighting(str, Enum):
    good = "good"
    low = "low"
    mixed = "mixed"
    backlit = "backlit"


VISIBLE_BODY_PARTS_ALLOWED = {
    "head", "neck", "shoulders", "upper_arms", "forearms",
    "hands", "torso", "hips", "thighs", "shins", "feet",
}

# Taxonomy leaves loaded from motion_taxonomy.yaml at startup.
# For inline validation, embed the full set here.
TAXONOMY_LEAVES: set[str] = {
    # locomotion
    "walking", "running", "moving", "positioning", "leaning",
    "turning", "stepping", "stumbling",
    # manipulation
    "reaching", "carrying", "holding", "grasping", "clasping",
    "releasing", "lifting", "placing", "pushing", "pulling",
    "writing", "pouring", "opening", "closing",
    # posture
    "standing", "sitting", "kneeling", "lying", "crouching",
    "reclining", "bending", "straightening",
    # communication
    "talking", "gesturing", "looking", "focusing", "interacting",
    "engaging", "petting", "nodding", "shaking_head", "shrugging",
    "pointing", "waving", "beckoning", "hugging", "clapping",
    # impact
    "kicking", "spiking", "striking", "throwing", "catching",
    "punching", "slapping", "blocking", "jumping", "landing",
    "falling", "diving",
    # self_directed
    "adjusting", "scratching", "rubbing", "wiping", "stretching",
    "yawning", "breathing_heavy", "fidgeting", "smoking",
    "drinking", "eating",
}

CAMERA_TERMS_FORBIDDEN: set[str] = {
    "close-up", "close up", "closeup", "medium shot", "medium-shot",
    "wide shot", "wide-shot", "medium close-up", "medium close up",
    "extreme close-up", "long shot", "establishing shot",
    "eye-level", "eye level", "low angle", "low-angle",
    "high angle", "high-angle", "dutch", "dutch angle",
    "overhead", "bird's eye", "frontal", "side view", "profile",
    "profile view", "steady", "steadicam", "handheld", "hand-held",
    "static", "tracking", "tracking shot", "pan", "panning",
    "tilt", "tilting", "zoom", "zooming", "dolly", "crane",
    "crane shot", "aerial", "pov", "point of view",
    "over the shoulder", "ots", "two-shot", "insert", "cutaway",
    "reverse", "reaction shot",
}


# ============================================================
# Sub-Models
# ============================================================

class SceneContext(BaseModel):
    visible_setting: str
    narrative_situation: Optional[str] = None
    narrative_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def null_narrative_on_low_confidence(self) -> "SceneContext":
        if self.narrative_confidence < 0.3:
            self.narrative_situation = None
        return self


class ShotContext(BaseModel):
    shot_type: ShotType
    shot_emotion_summary: str
    shot_motion_summary: str
    scene_context: SceneContext


class AlternativeCaptions(BaseModel):
    direct: str = Field(min_length=10)
    literary: str = Field(min_length=10)
    direction: str = Field(min_length=10)
    situational: str = Field(min_length=10)


class FacialComponents(BaseModel):
    eyes: str
    eyebrows: str
    mouth: str
    jaw: str
    gaze_direction: GazeDirection
    head_pose: HeadPose


class FacialAttributes(BaseModel):
    apparent_gender: ApparentGender
    apparent_age_range: ApparentAgeRange
    glasses: bool
    facial_hair: FacialHair
    head_covering: HeadCovering
    mask: bool
    makeup_visible: bool
    distinctive_notes: str = ""


class ObservableBlendshapeHints(BaseModel):
    brow_raise_inner: BlendshapeScale
    brow_raise_outer: BlendshapeScale
    brow_furrow: BlendshapeScale
    eye_widen: BlendshapeScale
    eye_squint: BlendshapeScale
    eye_blink_state: EyeBlinkState
    cheek_raise: BlendshapeScale
    nose_wrinkle: BlendshapeScale
    upper_lip_raise: BlendshapeScale
    lip_corner_pull: BlendshapeScale
    lip_corner_depress: BlendshapeScale
    lip_tighten: BlendshapeScale
    lip_part: BlendshapeScale
    jaw_clench: BlendshapeScale
    jaw_drop: BlendshapeScale


class FaceAnalysis(BaseModel):
    face_clearly_visible: bool
    face_size_ratio: float = Field(ge=0.0, le=1.0)

    primary_emotion: PrimaryEmotion
    secondary_emotion: Optional[PrimaryEmotion] = None
    valence: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    arousal: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    intensity: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    expression_caption: Optional[str] = None
    alternative_captions: Optional[AlternativeCaptions] = None

    facial_components: Optional[FacialComponents] = None
    facial_attributes: Optional[FacialAttributes] = None

    temporal_change: Optional[TemporalChange] = None
    micro_expression: Optional[bool] = None

    observable_blendshape_hints: Optional[ObservableBlendshapeHints] = None

    expression_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def enforce_micro_expression_temporal(self) -> "FaceAnalysis":
        if self.micro_expression is True and self.temporal_change is not None:
            if self.temporal_change != TemporalChange.rapid_micro:
                raise ValueError(
                    "micro_expression=true requires temporal_change='rapid_micro'"
                )
        return self

    @model_validator(mode="after")
    def null_on_low_confidence(self) -> "FaceAnalysis":
        if self.expression_confidence < 0.3:
            # Keep primary_emotion, null the rest
            self.valence = None
            self.arousal = None
            self.intensity = None
            self.expression_caption = None
            self.alternative_captions = None
            self.facial_components = None
            self.temporal_change = None
            self.micro_expression = None
            self.observable_blendshape_hints = None
        return self


class ActionQuality(BaseModel):
    intensity: Intensity
    tone: Tone
    tempo: Tempo


class KinematicsHint(BaseModel):
    trajectory: Trajectory
    periodicity: Periodicity
    symmetry: Symmetry
    duration_class: DurationClass


class UpperBodyDetail(BaseModel):
    head: str
    neck: str
    shoulders: str
    arms: str
    hands: str
    torso: str
    posture: PostureLabel


class BodyInteraction(BaseModel):
    count: InteractionCount
    contact: Contact
    relation: Relation
    interacts_with_person_index: list[int] = Field(default_factory=list)


class BodyAnalysis(BaseModel):
    body_clearly_visible: bool

    shot_frame_of_body: ShotFrameOfBody
    visible_body_parts: list[str]

    motion_caption: Optional[str] = None
    alternative_captions: Optional[AlternativeCaptions] = None

    action_primary: Optional[str] = None
    action_quality: Optional[ActionQuality] = None
    body_focus: Optional[BodyFocus] = None

    kinematics_hint: Optional[KinematicsHint] = None

    upper_body_detail: Optional[UpperBodyDetail] = None

    gesture_detail: Optional[str] = None
    hands_visible: bool = False

    interaction: Optional[BodyInteraction] = None

    motion_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("visible_body_parts")
    @classmethod
    def validate_body_parts(cls, v: list[str]) -> list[str]:
        for part in v:
            if part not in VISIBLE_BODY_PARTS_ALLOWED:
                raise ValueError(
                    f"Invalid body part '{part}'. "
                    f"Allowed: {sorted(VISIBLE_BODY_PARTS_ALLOWED)}"
                )
        return v

    @field_validator("action_primary")
    @classmethod
    def validate_action_primary(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Check taxonomy leaves
        if v in TAXONOMY_LEAVES:
            return v
        # Check other/<word> pattern
        import re
        if re.match(r"^other/[a-z_]+$", v):
            return v
        # Check forbidden camera terms
        if v.lower() in CAMERA_TERMS_FORBIDDEN:
            raise ValueError(
                f"Camera term '{v}' is FORBIDDEN in action_primary. "
                f"Use shot_context.shot_type for camera terms."
            )
        raise ValueError(
            f"action_primary '{v}' is not a taxonomy leaf and does not "
            f"match 'other/<word>' pattern. See motion_taxonomy.yaml."
        )
        return v

    @model_validator(mode="after")
    def null_on_low_motion_confidence(self) -> "BodyAnalysis":
        if self.motion_confidence < 0.3:
            # Keep shot_frame_of_body and visible_body_parts only
            self.motion_caption = None
            self.alternative_captions = None
            self.action_primary = None
            self.action_quality = None
            self.body_focus = None
            self.kinematics_hint = None
            self.upper_body_detail = None
            self.gesture_detail = None
            self.interaction = None
        return self

    @model_validator(mode="after")
    def hands_not_visible_consistency(self) -> "BodyAnalysis":
        if (
            not self.hands_visible
            and self.upper_body_detail is not None
            and self.upper_body_detail.hands != "not visible"
            and self.upper_body_detail.hands != "Not visible"
        ):
            # Auto-correct rather than raise
            self.upper_body_detail.hands = "not visible"
        return self

    @model_validator(mode="after")
    def body_focus_frame_consistency(self) -> "BodyAnalysis":
        if (
            self.body_focus == BodyFocus.full_body
            and self.shot_frame_of_body
            in {ShotFrameOfBody.close_face, ShotFrameOfBody.bust}
        ):
            raise ValueError(
                "body_focus='full_body' is impossible with "
                f"shot_frame_of_body='{self.shot_frame_of_body.value}'"
            )
        return self


class PersonAnalysis(BaseModel):
    person_index: int = Field(ge=0)
    spatial_position: SpatialPosition

    face_analysis: Optional[FaceAnalysis] = None
    body_analysis: Optional[BodyAnalysis] = None

    @model_validator(mode="after")
    def null_blocks_on_visibility(self) -> "PersonAnalysis":
        if self.face_analysis is not None and not self.face_analysis.face_clearly_visible:
            self.face_analysis = None
        if self.body_analysis is not None and not self.body_analysis.body_clearly_visible:
            self.body_analysis = None
        return self


class TopLevelInteraction(BaseModel):
    count: InteractionCount
    contact: Contact
    relation: Relation


class QualityFlags(BaseModel):
    face_clearly_visible: bool
    body_clearly_visible: bool
    motion_blur: bool
    occlusion: Occlusion
    lighting: Lighting
    camera_stable: bool
    frame_sampling_ok: bool
    vlm_confidence: float = Field(ge=0.0, le=1.0)


class UsabilityScore(BaseModel):
    face: float = Field(ge=0.0, le=1.0)
    motion: float = Field(ge=0.0, le=1.0)


class Meta(BaseModel):
    vlm_model: str
    vlm_version: str
    frames_used: int = Field(ge=1)
    infer_time_ms: int = Field(ge=0)


class ShotOutput(BaseModel):
    """Root model for per-shot VLM output. Validates the complete JSON."""

    shot_id: str
    source_movie: str

    shot_context: ShotContext

    persons: list[PersonAnalysis] = Field(min_length=1, max_length=3)

    interaction: TopLevelInteraction

    quality_flags: QualityFlags
    usability_score: UsabilityScore
    exclusion_reason: Optional[str] = None

    meta: Meta

    @model_validator(mode="after")
    def solo_interaction_no_contact(self) -> "ShotOutput":
        if self.interaction.count == InteractionCount.solo:
            if self.interaction.contact != Contact.none:
                raise ValueError(
                    "interaction.count='solo' requires interaction.contact='none'"
                )
        return self

    @model_validator(mode="after")
    def exclusion_on_low_vlm_confidence(self) -> "ShotOutput":
        if self.quality_flags.vlm_confidence < 0.3 and self.exclusion_reason is None:
            self.exclusion_reason = "low_vlm_confidence"
        return self
```

**Save as**: `schemas/shot_output.py` (or wherever your project places shared Python modules).

**Import**:
```python
from schemas.shot_output import ShotOutput

# Validate raw JSON string
shot = ShotOutput.model_validate_json(raw_json_bytes)

# Validate dict
shot = ShotOutput.model_validate(parsed_dict)
```

**Requirements**: Python 3.11+, `pydantic >= 2.0`.

---

## 8. JSON Schema Draft 7 (vLLM guided_json)

The following is a **condensed core** of the JSON Schema suitable for vLLM `GuidedDecodingParams(json=schema)`. The full version (with all nested definitions) should be exported via:

```python
import json
from schemas.shot_output import ShotOutput

schema = ShotOutput.model_json_schema()
with open("schemas/shot_output.json", "w") as f:
    json.dump(schema, f, indent=2)
```

### Core Schema (abbreviated)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ShotOutput",
  "type": "object",
  "required": [
    "shot_id", "source_movie", "shot_context", "persons",
    "interaction", "quality_flags", "usability_score", "meta"
  ],
  "properties": {
    "shot_id": { "type": "string" },
    "source_movie": { "type": "string" },

    "shot_context": {
      "type": "object",
      "required": ["shot_type", "shot_emotion_summary", "shot_motion_summary", "scene_context"],
      "properties": {
        "shot_type": {
          "type": "string",
          "enum": ["close-up", "medium close-up", "medium", "medium long", "wide", "extreme wide"]
        },
        "shot_emotion_summary": { "type": "string" },
        "shot_motion_summary": { "type": "string" },
        "scene_context": {
          "type": "object",
          "required": ["visible_setting", "narrative_confidence"],
          "properties": {
            "visible_setting": { "type": "string" },
            "narrative_situation": { "type": ["string", "null"] },
            "narrative_confidence": { "type": "number", "minimum": 0, "maximum": 1 }
          }
        }
      }
    },

    "persons": {
      "type": "array",
      "minItems": 1,
      "maxItems": 3,
      "items": {
        "type": "object",
        "required": ["person_index", "spatial_position"],
        "properties": {
          "person_index": { "type": "integer", "minimum": 0 },
          "spatial_position": {
            "type": "string",
            "enum": ["center", "left", "right", "background"]
          },
          "face_analysis": {
            "oneOf": [
              { "type": "null" },
              {
                "type": "object",
                "required": ["face_clearly_visible", "face_size_ratio", "primary_emotion", "expression_confidence"],
                "properties": {
                  "face_clearly_visible": { "type": "boolean" },
                  "face_size_ratio": { "type": "number", "minimum": 0, "maximum": 1 },
                  "primary_emotion": {
                    "type": "string",
                    "enum": ["anger", "sadness", "joy", "fear", "surprise", "disgust", "contempt", "neutral", "complex"]
                  },
                  "secondary_emotion": {
                    "oneOf": [
                      { "type": "null" },
                      { "type": "string", "enum": ["anger", "sadness", "joy", "fear", "surprise", "disgust", "contempt", "neutral", "complex"] }
                    ]
                  },
                  "valence": { "type": ["number", "null"], "minimum": -1, "maximum": 1 },
                  "arousal": { "type": ["number", "null"], "minimum": 0, "maximum": 1 },
                  "intensity": { "type": ["number", "null"], "minimum": 0, "maximum": 1 },
                  "expression_caption": { "type": ["string", "null"] },
                  "alternative_captions": {
                    "oneOf": [
                      { "type": "null" },
                      {
                        "type": "object",
                        "required": ["direct", "literary", "direction", "situational"],
                        "properties": {
                          "direct": { "type": "string" },
                          "literary": { "type": "string" },
                          "direction": { "type": "string" },
                          "situational": { "type": "string" }
                        }
                      }
                    ]
                  },
                  "facial_components": {
                    "oneOf": [
                      { "type": "null" },
                      {
                        "type": "object",
                        "required": ["eyes", "eyebrows", "mouth", "jaw", "gaze_direction", "head_pose"],
                        "properties": {
                          "eyes": { "type": "string" },
                          "eyebrows": { "type": "string" },
                          "mouth": { "type": "string" },
                          "jaw": { "type": "string" },
                          "gaze_direction": { "type": "string", "enum": ["camera", "left", "right", "up", "down", "averted", "closed"] },
                          "head_pose": { "type": "string", "enum": ["frontal", "3q_left", "3q_right", "profile_left", "profile_right", "tilted_up", "tilted_down"] }
                        }
                      }
                    ]
                  },
                  "facial_attributes": {
                    "oneOf": [
                      { "type": "null" },
                      {
                        "type": "object",
                        "required": ["apparent_gender", "apparent_age_range", "glasses", "facial_hair", "head_covering", "mask", "makeup_visible"],
                        "properties": {
                          "apparent_gender": { "type": "string", "enum": ["male", "female", "ambiguous"] },
                          "apparent_age_range": { "type": "string", "enum": ["child", "teen", "young_adult", "adult", "middle_aged", "elderly"] },
                          "glasses": { "type": "boolean" },
                          "facial_hair": { "type": "string", "enum": ["none", "stubble", "beard", "mustache"] },
                          "head_covering": { "type": "string", "enum": ["none", "hat", "hood", "scarf"] },
                          "mask": { "type": "boolean" },
                          "makeup_visible": { "type": "boolean" },
                          "distinctive_notes": { "type": "string" }
                        }
                      }
                    ]
                  },
                  "temporal_change": {
                    "oneOf": [
                      { "type": "null" },
                      { "type": "string", "enum": ["static", "building", "peak_then_release", "transition", "rapid_micro"] }
                    ]
                  },
                  "micro_expression": { "type": ["boolean", "null"] },
                  "observable_blendshape_hints": {
                    "oneOf": [
                      { "type": "null" },
                      {
                        "type": "object",
                        "required": [
                          "brow_raise_inner", "brow_raise_outer", "brow_furrow",
                          "eye_widen", "eye_squint", "eye_blink_state",
                          "cheek_raise", "nose_wrinkle", "upper_lip_raise",
                          "lip_corner_pull", "lip_corner_depress", "lip_tighten",
                          "lip_part", "jaw_clench", "jaw_drop"
                        ],
                        "properties": {
                          "brow_raise_inner": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "brow_raise_outer": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "brow_furrow": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "eye_widen": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "eye_squint": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "eye_blink_state": { "type": "string", "enum": ["open", "half", "closed", "rapid_blink", "unknown"] },
                          "cheek_raise": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "nose_wrinkle": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "upper_lip_raise": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "lip_corner_pull": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "lip_corner_depress": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "lip_tighten": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "lip_part": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "jaw_clench": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] },
                          "jaw_drop": { "type": "string", "enum": ["none", "slight", "medium", "strong", "unknown"] }
                        }
                      }
                    ]
                  },
                  "expression_confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                }
              }
            ]
          },
          "body_analysis": {
            "oneOf": [
              { "type": "null" },
              {
                "type": "object",
                "required": ["body_clearly_visible", "shot_frame_of_body", "visible_body_parts", "hands_visible", "motion_confidence"],
                "properties": {
                  "body_clearly_visible": { "type": "boolean" },
                  "shot_frame_of_body": {
                    "type": "string",
                    "enum": ["close_face", "bust", "half_body", "three_quarter", "full_body", "wide"]
                  },
                  "visible_body_parts": {
                    "type": "array",
                    "items": {
                      "type": "string",
                      "enum": ["head", "neck", "shoulders", "upper_arms", "forearms", "hands", "torso", "hips", "thighs", "shins", "feet"]
                    }
                  },
                  "motion_caption": { "type": ["string", "null"] },
                  "alternative_captions": {
                    "oneOf": [
                      { "type": "null" },
                      {
                        "type": "object",
                        "required": ["direct", "literary", "direction", "situational"],
                        "properties": {
                          "direct": { "type": "string" },
                          "literary": { "type": "string" },
                          "direction": { "type": "string" },
                          "situational": { "type": "string" }
                        }
                      }
                    ]
                  },
                  "action_primary": { "type": ["string", "null"] },
                  "action_quality": {
                    "oneOf": [
                      { "type": "null" },
                      {
                        "type": "object",
                        "required": ["intensity", "tone", "tempo"],
                        "properties": {
                          "intensity": { "type": "string", "enum": ["low", "mid", "high"] },
                          "tone": { "type": "string", "enum": ["relaxed", "tense", "controlled", "contemplative"] },
                          "tempo": { "type": "string", "enum": ["sustained", "punctuated", "accelerating", "decelerating"] }
                        }
                      }
                    ]
                  },
                  "body_focus": {
                    "oneOf": [
                      { "type": "null" },
                      { "type": "string", "enum": ["upper_body", "hands", "torso", "posture", "gesture", "full_body", "lower_body", "face_and_gaze"] }
                    ]
                  },
                  "kinematics_hint": {
                    "oneOf": [
                      { "type": "null" },
                      {
                        "type": "object",
                        "required": ["trajectory", "periodicity", "symmetry", "duration_class"],
                        "properties": {
                          "trajectory": { "type": "string", "enum": ["linear", "arc", "circular", "erratic", "static"] },
                          "periodicity": { "type": "string", "enum": ["periodic", "non_periodic"] },
                          "symmetry": { "type": "string", "enum": ["bilateral_symmetric", "bilateral_asymmetric", "axial"] },
                          "duration_class": { "type": "string", "enum": ["onset_only", "ongoing", "peak_then_release"] }
                        }
                      }
                    ]
                  },
                  "upper_body_detail": {
                    "oneOf": [
                      { "type": "null" },
                      {
                        "type": "object",
                        "required": ["head", "neck", "shoulders", "arms", "hands", "torso", "posture"],
                        "properties": {
                          "head": { "type": "string" },
                          "neck": { "type": "string" },
                          "shoulders": { "type": "string" },
                          "arms": { "type": "string" },
                          "hands": { "type": "string" },
                          "torso": { "type": "string" },
                          "posture": { "type": "string", "enum": ["upright", "leaning_forward", "leaning_back", "slumped", "turning", "crouching", "reclining", "walking", "running"] }
                        }
                      }
                    ]
                  },
                  "gesture_detail": { "type": ["string", "null"] },
                  "hands_visible": { "type": "boolean" },
                  "interaction": {
                    "oneOf": [
                      { "type": "null" },
                      {
                        "type": "object",
                        "required": ["count", "contact", "relation", "interacts_with_person_index"],
                        "properties": {
                          "count": { "type": "string", "enum": ["solo", "dyadic", "triadic", "crowd"] },
                          "contact": { "type": "string", "enum": ["none", "incidental", "sustained"] },
                          "relation": { "type": "string", "enum": ["parallel", "coordinated", "opposing", "hierarchical"] },
                          "interacts_with_person_index": { "type": "array", "items": { "type": "integer" } }
                        }
                      }
                    ]
                  },
                  "motion_confidence": { "type": "number", "minimum": 0, "maximum": 1 }
                }
              }
            ]
          }
        }
      }
    },

    "interaction": {
      "type": "object",
      "required": ["count", "contact", "relation"],
      "properties": {
        "count": { "type": "string", "enum": ["solo", "dyadic", "triadic", "crowd"] },
        "contact": { "type": "string", "enum": ["none", "incidental", "sustained"] },
        "relation": { "type": "string", "enum": ["parallel", "coordinated", "opposing", "hierarchical"] }
      }
    },

    "quality_flags": {
      "type": "object",
      "required": ["face_clearly_visible", "body_clearly_visible", "motion_blur", "occlusion", "lighting", "camera_stable", "frame_sampling_ok", "vlm_confidence"],
      "properties": {
        "face_clearly_visible": { "type": "boolean" },
        "body_clearly_visible": { "type": "boolean" },
        "motion_blur": { "type": "boolean" },
        "occlusion": { "type": "string", "enum": ["none", "partial", "heavy"] },
        "lighting": { "type": "string", "enum": ["good", "low", "mixed", "backlit"] },
        "camera_stable": { "type": "boolean" },
        "frame_sampling_ok": { "type": "boolean" },
        "vlm_confidence": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },

    "usability_score": {
      "type": "object",
      "required": ["face", "motion"],
      "properties": {
        "face": { "type": "number", "minimum": 0, "maximum": 1 },
        "motion": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },

    "exclusion_reason": { "type": ["string", "null"] },

    "meta": {
      "type": "object",
      "required": ["vlm_model", "vlm_version", "frames_used", "infer_time_ms"],
      "properties": {
        "vlm_model": { "type": "string" },
        "vlm_version": { "type": "string" },
        "frames_used": { "type": "integer", "minimum": 1 },
        "infer_time_ms": { "type": "integer", "minimum": 0 }
      }
    }
  }
}
```

**Full programmatic export**: Run `ShotOutput.model_json_schema()` from the Pydantic model (Section 7) to get the complete schema with all `$defs` references resolved. Save to `schemas/shot_output.json`.

**vLLM usage**:
```python
import json
from vllm.sampling_params import GuidedDecodingParams

with open("schemas/shot_output.json") as f:
    schema = json.load(f)

guided = GuidedDecodingParams(json=schema)
```

---

## 9. Edge-Case Q&A

### Face-Related

**Q1: A person is facing away -- their back is to the camera and no face is visible.**
A: Set `face_clearly_visible=false`. The entire `face_analysis` block becomes `null`. Write `body_analysis` normally describing the visible back, shoulders, and posture.

**Q2: The face is partially covered by hair on one side.**
A: If sufficient face is visible (>3% of frame), keep `face_clearly_visible=true`. Use `"unknown"` for any `observable_blendshape_hints` fields that are occluded (e.g., `brow_raise_outer` on the hidden side). Note the occlusion in `distinctive_notes`.

**Q3: Two emotions are simultaneously present -- joy and sadness ("bittersweet").**
A: If one is clearly dominant, use it as `primary_emotion` with the other as `secondary_emotion`. If genuinely inseparable, use `primary_emotion="complex"` and describe the blend in `expression_caption`.

**Q4: The person is crying with joy -- valence positive or negative?**
A: Positive. Valence reflects the emotional valence, not the physical manifestation. Tears of joy = positive valence. Note the tears in `facial_components.eyes`.

**Q5: Eyes are closed -- what about gaze_direction?**
A: `gaze_direction="closed"`. Also set `eye_blink_state="closed"` and `eye_widen="unknown"`, `eye_squint="unknown"`.

**Q6: A micro-expression is detected but temporal_change was set to "building".**
A: This is an error. If `micro_expression=true`, then `temporal_change` MUST be `"rapid_micro"`. The Pydantic validator will reject this combination.

**Q7: The same actor appears as a mirror reflection -- count as two people?**
A: No. Only the real person goes into `persons[]`. Note the reflection in `facial_attributes.distinctive_notes`: "Mirror reflection visible in background".

**Q8: VLM outputs "close-up" as action_primary.**
A: The post-processing normalizer removes it (flagged via `camera_terms_forbidden` in `motion_synonyms.yaml`). The field is set to `null` or re-inferred. The validator script also catches this.

### Body-Related

**Q9: Walking is happening but only the upper body is visible (bust shot).**
A: Permitted. If the walking rhythm (arm swing, shoulder movement, torso bob) is observable, set `action_primary="walking"`. But `visible_body_parts` must NOT include legs/feet. Set `shot_frame_of_body="bust"`.

**Q10: A person is sitting motionless -- only breathing is visible.**
A: `action_primary="sitting"`, `action_quality.intensity="low"`, `tempo="sustained"`. Note the breathing in `upper_body_detail.torso` (e.g., "slight rhythmic rise and fall of chest").

**Q11: Hands are visible but the person is not doing anything with them.**
A: `hands_visible=true`. In `gesture_detail`, write honestly: "Hands visible at lap level, resting on thighs, no discernible gesture." Do not fabricate gestures.

**Q12: A person enters and leaves the frame during the shot.**
A: Include them in `persons[]` for the duration they are visible. Set `body_clearly_visible` based on their peak visibility. Note the entry/exit in `motion_caption` (e.g., "Person enters frame from the right at 1.2s and exits left at 3.8s").

**Q13: Full body is visible -- which shot_frame_of_body?**
A: `full_body` requires feet visible. If ankles are cut off, use `three_quarter`. If only mid-thigh down is cut, still `three_quarter`.

**Q14: body_focus="full_body" but shot_frame_of_body="bust" -- is this valid?**
A: No. The Pydantic validator rejects this. `body_focus="full_body"` requires at least `half_body` or wider framing. Use `posture` or `upper_body` instead.

### Cross-Domain

**Q15: A person's face shows fear but their body appears relaxed.**
A: This is unusual but valid. Record both honestly. The cross-axis consistency check (face=fear + body=relaxed) will FLAG the shot but not auto-exclude it. It may represent suppressed fear, or a genuine acting choice.

**Q16: Both face_analysis and body_analysis would be null for a person.**
A: This should not happen. Layer-1 filtering should have excluded the shot. If it does occur, set `exclusion_reason="no_visible_analysis"` and flag for filter review.

**Q17: The subject is a mannequin, doll, or painting of a person.**
A: Set `exclusion_reason="non_human_subject"`. Do not produce `face_analysis` or `body_analysis`.

**Q18: alternative_captions -- one style is genuinely impossible to write.**
A: Never set any of the 4 styles to `null`. Write at minimum a short, honest attempt (20+ words). All four keys must have non-null strings when the parent block is non-null.

**Q19: Two people are interacting but one has their back to the camera.**
A: Person with back to camera: `face_analysis=null` (face not visible), `body_analysis` describes the visible back, shoulders, posture. The interaction block is still filled for both people. Use `interacts_with_person_index` to link them.

**Q20: A crowd scene with 5+ people -- how many go in persons[]?**
A: Only people with `face_size_ratio >= 0.03` or occupying > 10% frame height. Maximum 3 entries in `persons[]`. The rest are captured via `interaction.count="crowd"` at the top level.

**Q21: Camera shakes intentionally (handheld style) -- camera_stable=false?**
A: Yes. `camera_stable=false` for any visible camera movement, intentional or not. This is a motion-learning quality signal, not a judgment of cinematography.

**Q22: action_primary needs multiple values (person is walking AND talking).**
A: Choose the single most dominant action. Walking is the body action; talking involves both face and body. If legs are driving the scene, use `walking`. If the speech gestures dominate, use `talking`. Describe the secondary action in `gesture_detail` or `motion_caption`.

**Q23: blendshape_hints -- "I think the brow is slightly raised but I am not sure."**
A: Use `"slight"`. Never escalate beyond your confidence. If truly uncertain, use `"none"` or `"slight"` -- never `"medium"` or `"strong"` on a guess.

**Q24: upper_body_detail.head says "tilted left" -- is this face analysis leaking?**
A: No. Head tilt is a physical body orientation, not a facial expression. Describing head position in `upper_body_detail.head` is correct and encouraged. Do NOT describe smile, frown, or eye movement here -- that belongs in `face_analysis`.

**Q25: A static shot with no visible motion at all.**
A: `action_primary` = the current posture (`standing`, `sitting`, etc.). `action_quality.intensity="low"`, `tempo="sustained"`, `kinematics_hint.trajectory="static"`. `motion_caption` describes the stillness honestly.

---

## 10. Validation Checklist

External workers must verify before each batch submission:

- [ ] Every JSON file passes `ShotOutput.model_validate_json()` without errors
- [ ] Zero camera terms in any `action_primary` field (run `scripts/validate_body_analysis.py`)
- [ ] `visible_body_parts` is consistent with `shot_frame_of_body` (no legs in bust shots)
- [ ] All 4 `alternative_captions` keys are non-null strings (20+ words each) in both face and body
- [ ] `expression_caption` contains NO body/posture descriptions
- [ ] `motion_caption` contains NO facial expression descriptions
- [ ] When `face_analysis` is `null`, `face_clearly_visible=false` is confirmed
- [ ] When `body_analysis` is `null`, `body_clearly_visible=false` is confirmed
- [ ] When `micro_expression=true`, `temporal_change="rapid_micro"`
- [ ] All 15 `observable_blendshape_hints` fields (14 standard + `eye_blink_state`) are filled
- [ ] `usability_score.face` and `usability_score.motion` are independently computed
- [ ] `interaction.count="solo"` implies `interaction.contact="none"` at top level
- [ ] `action_primary` values are in taxonomy leaves or match `other/<word>` regex
- [ ] `hands_visible=false` implies `upper_body_detail.hands="not visible"`
- [ ] No empty strings `""`, empty arrays `[]`, or literal string `"null"` used as null substitutes
- [ ] `narrative_situation` is `null` when `narrative_confidence < 0.3`
- [ ] Parse error rate < 2% per movie batch
- [ ] Emotion distribution Shannon entropy >= 1.5 bits per movie

---

## 11. Change Log

| Version | Date       | Changes                                              |
|---------|------------|------------------------------------------------------|
| 1.0     | 2026-04-17 | Initial integrated schema combining face + body + context + quality. Pydantic v2 model and JSON Schema Draft 7 included. |
