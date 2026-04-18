# External Worker Specification — Body (Upper-Body Motion) Inference

**Version**: 1.0
**Date**: 2026-04-16
**Project**: Video_DB_Face — Body-motion adapter training for Wan2.2 / LTX-2.3 video models
**Target model to infer**: Qwen3-VL-32B-Instruct (BF16 on H100 80GB)
**This document covers**: the `body_analysis` block of the per-shot JSON output.
**Companion document**: `external_spec_face_EN.md` (the `face_analysis` block).

> **Important**: Face and Body are produced in a SINGLE VLM call per shot (one JSON object, two analysis blocks). Use both specification documents together. Running face and body as two separate calls is wasteful and forbidden.

> **Design decision (2026-04-16)**: This project **does not use 3D reconstruction** (SMPL-X / 4D-Humans / TokenHMR / GVHMR). Prior experiments showed insufficient quality when subjects are partially framed (typical film framing is bust-to-half-body, legs often cropped). All downstream training is 2D-based and **upper-body centric**. Your labels reflect this — describe only what the camera actually shows.

---

## 1. Purpose

We are building a dataset for fine-grained acting-level body-motion control of video-generation models (Wan2.2, LTX-2.3) via LoRA + ControlNet adapters. For each shot selected by the upstream Layer-1 filter, you produce structured, high-quality body-motion labels in English.

Labels serve four downstream uses:
1. **Text conditioning of the motion LoRA** — the `motion_caption` and `alternative_captions` are training text.
2. **2D pose ControlNet training** — your labels cross-check the MediaPipe/DWPose numeric extraction done by us.
3. **Candidate retrieval for the user-facing tool** — users type a situation and pick from proposed shots; your captions are what they read.
4. **Learning-pool gating** — `motion_confidence` and `body_clearly_visible` decide whether a shot enters the training set.

---

## 2. Pipeline Position

```
[Layer-1 filter (us)]  ->  [YOU: VLM labeling]  ->  [Layer-3 curation (us)]  ->  [Layer-4 training (us)]
                                |
                        face_analysis (companion doc)
                        body_analysis (this doc)
                        single JSON per shot
```

---

## 3. Environment, Model, Sampling

Identical to the face specification (Section 3 of the companion doc). Do not rebuild the environment separately.

---

## 4. Input

Frame sampling, preprocessing, and input manifest are identical to the face specification. One VLM call per shot produces both face_analysis and body_analysis.

---

## 5. Output JSON Schema — `body_analysis` block

The overall envelope is described in `external_spec_face_EN.md` Section 6. Here is the `body_analysis` block only, which sits inside each entry of `persons[]`:

### 5.1 `body_analysis` (per person)

```json
{
  "body_clearly_visible": true,

  "shot_frame_of_body": "close_face|bust|half_body|three_quarter|full_body|wide",
  "visible_body_parts": ["head", "neck", "shoulders", "upper_arms", "forearms", "hands", "torso"],

  "motion_caption": "Primary caption, English, 50-180 words. Describe BODY ONLY, not the face. Include any time-axis change (onset, sustain, release).",

  "alternative_captions": {
    "direct":      "20-40 words. Plain descriptive body motion.",
    "literary":    "30-60 words. Evocative/cinematic motion prose.",
    "direction":   "30-50 words. Actor or choreography-direction style.",
    "situational": "30-60 words. Intent or internal state behind the motion."
  },

  "action_primary":  "walking",
  "action_quality": {
    "intensity": "low|mid|high",
    "tone":      "relaxed|tense|controlled|contemplative",
    "tempo":     "sustained|punctuated|accelerating|decelerating"
  },
  "body_focus": "upper_body|hands|torso|posture|gesture|full_body|lower_body|face_and_gaze",

  "kinematics_hint": {
    "periodicity":    "periodic|non_periodic",
    "symmetry":       "bilateral_symmetric|bilateral_asymmetric|axial",
    "duration_class": "onset_only|ongoing|peak_then_release",
    "trajectory":     "linear|arc|circular|erratic|static"
  },

  "upper_body_detail": {
    "head":      "tilt, turn direction — e.g. 'tilted slightly forward, turning to the left'",
    "neck":      "extension, tension — e.g. 'extended, slight strain visible'",
    "shoulders": "level, rise/drop, forward/back — e.g. 'right shoulder raised, left relaxed'",
    "arms":      "position and motion — e.g. 'right arm raised mid-level, left arm at side'",
    "hands":     "visible gesture detail — e.g. 'right hand in loose fist, left hand hidden behind torso'",
    "torso":     "twist, lean, posture — e.g. 'slight forward lean, twisted ~15 degrees left'",
    "posture":   "upright|leaning_forward|leaning_back|slumped|turning|crouching|reclining|walking|running"
  },

  "gesture_detail":  "Free-form English. Name specific gestures: nodding, shrugging, pointing, wiping, reaching for X, tapping fingers, smoothing clothes, etc. Cite WHICH side (left/right) when asymmetric.",
  "hands_visible":   true,

  "interaction": {
    "count":    "solo|dyadic|triadic|crowd",
    "contact":  "none|incidental|sustained",
    "relation": "parallel|coordinated|opposing|hierarchical",
    "interacts_with_person_index": [1]
  },

  "motion_confidence": 0.80
}
```

### 5.2 Field-by-field notes

**shot_frame_of_body** — which part of the body the frame actually contains:
| value            | meaning                                                |
| ---------------- | ------------------------------------------------------ |
| `close_face`     | Only face + neck visible. Body basically absent.        |
| `bust`           | Head to mid-chest. Common for dialogue.                 |
| `half_body`      | Head to waist. Arms and hands usually visible.          |
| `three_quarter`  | Head to mid-thigh. Standard medium shot.                |
| `full_body`      | Entire body visible. RARE in film — report faithfully. |
| `wide`           | Body small in frame; many subjects possible.            |

**visible_body_parts** — strict truth list. If legs are cropped out of frame, DO NOT include `thighs/shins/feet`. Allowed values:
`head, neck, shoulders, upper_arms, forearms, hands, torso, hips, thighs, shins, feet`.

**action_primary** — choose from the hierarchy below (leaf values). If none fits, use `other/<single_lowercase_word>`.
```
locomotion:      walking, running, moving, positioning, leaning
manipulation:    reaching, carrying, holding, grasping, clasping, releasing
posture:         standing, sitting, kneeling, lying, crouching
communication:   talking, gesturing, looking, focusing, interacting, engaging, petting
impact:          kicking, spiking, striking, throwing, pushing, catching
```
(See `docs/motion_taxonomy.yaml` for the authoritative list with synonyms.)

**action_quality** — three independent axes:
- `intensity`: energetic magnitude
- `tone`: internal tension register
- `tempo`: time profile (captured via motion rhythm)

**body_focus** — which region drives the action. In film shots this is usually `upper_body`, `hands`, or `gesture`. `full_body` and `lower_body` are rare.

**kinematics_hint** — give your best visual estimate. We cross-check with optical flow.

**upper_body_detail** — the level of specificity you are EXPECTED to reach for this project. Do not short-cut this block.

**gesture_detail** — most valuable field for fine-grained training signal. Be specific. Bad: "gesturing with hands". Good: "right hand sweeps outward in a dismissive arc at chest height; left hand remains at the side, fingers half-curled."

**motion_confidence** — your own confidence, in [0, 1]. Low when: body mostly occluded, motion blur severe, less than 2 frames show motion change, shot is almost static but you sense something you cannot place.

### 5.3 NEVER output

Fabricated content is worse than missing content. In particular:
- Do NOT describe the face, gaze, or eyes in `body_analysis`. Those belong to `face_analysis`.
- Do NOT describe body parts that are cropped out of frame.
- Do NOT use camera terminology in any `action_*`, `body_focus`, or `kinematics_*` field.
  - Forbidden values: `close-up, medium shot, wide, wide shot, low, high, eye-level, dutch, frontal, side, profile, overhead, steady, handheld, static, tracking, pan, zoom`
  - These belong ONLY in `shot_context.shot_type`.
- Do NOT estimate 3D quantities: joint angles in degrees, bone lengths, distance in meters, SMPL betas. This is a 2D observation task.
- Do NOT use subjective judgments: "graceful", "ugly", "beautiful", "sexy", "masculine", "feminine". Describe the motion, not your evaluation.
- Do NOT infer motions that are not in the sampled frames (e.g. "then he would probably have walked away").
- Do NOT translate non-English audio.

### 5.4 Positive style guide — the four alternative captions

| Style         | Function                           | Length   | Style rules                                                        |
| ------------- | ---------------------------------- | -------- | ------------------------------------------------------------------ |
| `direct`      | Plain, concrete verbs              | 20-40 w  | Simple present. "She lifts her right hand to chest height."        |
| `literary`    | Cinematic, evocative               | 30-60 w  | Present tense. Some metaphor. Rhythm-aware.                        |
| `direction`   | Actor/choreography notes           | 30-50 w  | Imperative. Beat counts ok. Sides/directions specified.            |
| `situational` | Intent or internal state           | 30-60 w  | "As if ...", "with the hesitation of ...". Inner logic, no backstory. |

All four describe **the same body motion in the same shot**, from four angles — not four different motions.

Example — same person, four styles:

- **direct**: "He leans forward onto his right elbow, his left hand sweeping the table toward himself."
- **literary**: "He tilts into the table like a man bracing against a slow tide, his left hand drawing papers home."
- **direction**: "Weight onto right forearm. Beat. Left arm in, palm scoops toward chest. Shoulders stay level; no head turn."
- **situational**: "With the focused settling of someone about to argue a point they have already decided to win."

### 5.5 Interaction block (for `category = dominant` or `multi`)

Also duplicate the interaction block at the TOP LEVEL of the shot JSON (not inside per-person), because interaction is a shot property. Inside each `body_analysis`, use `interacts_with_person_index` to name specific counterparts.

---

## 6. Quality Gates

### 6.1 Exclusion cascade (body-specific)

| Condition                                              | Action                                         |
| ------------------------------------------------------ | ---------------------------------------------- |
| Subject occupies < 10% of frame height                 | `body_clearly_visible = false`                 |
| `body_clearly_visible = false`                         | NULL the entire `body_analysis` block          |
| Entire body heavily occluded by object/other person    | `body_clearly_visible = false`                 |
| `motion_confidence < 0.3`                              | Keep `shot_frame_of_body` and `visible_body_parts` only; NULL rest |
| Subject is cartoon / CGI / non-human                   | `exclusion_reason = "non_human_subject"`       |
| No visible body parts at all (only face)               | `shot_frame_of_body = "close_face"`, `body_clearly_visible=false` |

### 6.2 Usability scoring (body)

`usability_score.motion` in [0, 1]. Compute approximately as:

```
motion = 0.4 * body_clearly_visible
       + 0.3 * motion_confidence
       + 0.2 * (camera_stable ? 1 : 0)
       + 0.1 * (shot_frame_of_body in {half_body, three_quarter, full_body} ? 1 : 0.5)
```

### 6.3 Anti-hallucination checks (self-QC before submitting)

For each output, before emitting, you must verify:
- [ ] No camera terms appear inside `action_*` or `body_focus` or `kinematics_*`.
- [ ] `visible_body_parts` only contains parts actually visible.
- [ ] `action_primary` is in the taxonomy leaf set or uses `other/<word>` format.
- [ ] `upper_body_detail.hands` mentions left vs right when both are visible.
- [ ] `motion_caption` does NOT mention facial expression.
- [ ] `alternative_captions` has all 4 keys with non-null strings (unless body NULL).

A validator script is provided in `scripts/validate_body_analysis.py`. Run on every batch.

---

## 7. Error Handling, Batching, Checkpointing

Identical to face spec (Sections 8, 9 of the companion doc). One JSON file per shot covers BOTH face and body.

---

## 8. Validation & QC (Body-specific)

### 8.1 Distribution sanity checks per movie

| Metric                                              | Expected range (soft)          |
| --------------------------------------------------- | ------------------------------ |
| `action_primary` leaf diversity (distinct values)   | >= 15 across a 5000-shot movie |
| Fraction using `other/<word>`                       | < 15%                          |
| Fraction of `shot_frame_of_body = close_face`       | 10-30% (too high -> camera bias) |
| Fraction of `full_body`                             | < 10% (film bias)              |
| `body_clearly_visible = true` fraction              | 40-75% depending on movie      |
| `motion_caption` mean length                        | 80-150 words                   |
| `alternative_captions` 4-key presence rate          | > 90%                          |
| `gesture_detail` empty-string fraction              | < 20%                          |

### 8.2 Cross-axis consistency (body)

Flag (do not auto-exclude) if:
- `action_primary = walking` AND `shot_frame_of_body in {close_face, bust}` — probably should be `other/turning_torso` or `positioning`.
- `action_quality.intensity = high` AND `kinematics_hint.trajectory = static` — contradiction.
- `hands_visible = true` AND `upper_body_detail.hands = ""` — missing detail.
- Any `action_primary` value appears in `action_tags` oftop-10 Top-10 camera-term list — logging bug, escalate.
- `body_focus = full_body` AND `shot_frame_of_body in {close_face, bust}` — impossible.

### 8.3 Consistency between face and body
- If `face_analysis.primary_emotion = fear` AND `body_analysis.action_quality.tone = relaxed` — flag (possible but rare).
- If `face_analysis` is NULL (face not visible) AND `body_analysis` is NULL (body not visible) — the shot should have been excluded at Layer-1. Flag for filter review.

---

## 9. Prompt Template — Body portion of the combined prompt

```
BODY ANALYSIS INSTRUCTIONS:

For each person visible in persons[], produce body_analysis:

1. Determine shot_frame_of_body from actual frame content:
   close_face | bust | half_body | three_quarter | full_body | wide

2. visible_body_parts: list ONLY the parts visible. If legs are cropped,
   do NOT include thighs/shins/feet. Order from head down.

3. action_primary MUST be one of:
     locomotion/     walking, running, moving, positioning, leaning
     manipulation/   reaching, carrying, holding, grasping, clasping, releasing
     posture/        standing, sitting, kneeling, lying, crouching
     communication/  talking, gesturing, looking, focusing, interacting, engaging, petting
     impact/         kicking, spiking, striking, throwing, pushing, catching
   If none fits, use other/<single_lowercase_word>.

   FORBIDDEN values (these are CAMERA terms, not actions):
     close-up, medium shot, wide, low, high, eye-level, dutch, frontal,
     side, profile, overhead, steady, handheld, static, tracking, pan, zoom.

4. action_quality: fill all three axes (intensity, tone, tempo).

5. upper_body_detail: fill each sub-field with a concrete observation.
   Specify left/right when asymmetric. If a region is not visible, use "not visible".

6. gesture_detail: name specific gestures (nod, shrug, point, wipe, reach,
   tap, smooth, fidget, adjust). Cite WHICH side. Do not be generic.

7. Produce alternative_captions with 4 styles (direct/literary/direction/
   situational) all describing the SAME motion from different angles.

8. Describe BODY ONLY. Do not describe face, eyes, gaze, or facial expression.

9. If body is cropped to close_face or heavily occluded,
   set body_clearly_visible=false and NULL the rest of body_analysis.

10. motion_confidence reflects your own certainty. Use it honestly.
```

Append this to the face prompt template in the companion document. Do not run as a separate call.

---

## 10. Deliverables Checklist (Body-specific)

For each batch delivery (over and above face deliverables):
- [ ] `action_primary` leaf histogram included in `movie_meta.json`
- [ ] `shot_frame_of_body` distribution included
- [ ] `other/<word>` list with counts (for us to promote to the taxonomy)
- [ ] 20 manually reviewed shots including at least 5 with `body_focus=full_body` if any exist
- [ ] The validator script output (from `scripts/validate_body_analysis.py`) with zero hard errors

---

## 11. Reference Materials

- Companion: `external_spec_face_EN.md`
- Integrated schema: `docs/json_schema_integrated.md`
- Few-shot examples: `docs/vlm_prompts/examples/`
- Taxonomy: `docs/motion_taxonomy.yaml`
- Synonyms: `docs/motion_synonyms.yaml`
- Validator: `scripts/validate_body_analysis.py`

---

## 12. Common Pitfalls From Prior Runs

These are real mistakes observed in the prior trailer dataset (Qwen2.5-Omni-7B-GPTQ-INT4). Do NOT repeat them.

### 12.1 Camera-term pollution in action fields
Prior dataset's action_tags Top 10 were ALL camera terms:
`medium shot, close-up, low, eye-level, medium close-up, high, static, steady, frontal, wide shot`.
None are actions. This entire pollution must not occur in this run. The validator script rejects any of these strings in `action_*`/`body_focus`/`kinematics_*`.

### 12.2 First-character-only bias
Prior adapter only parsed the first character's action. Here you MUST produce body_analysis for every visible person up to the declared `num_people`.

### 12.3 Vague gestures
Writing "gesturing with hands" is useless. Specify: which hand, direction of motion, height relative to torso, end position, speed category.

### 12.4 Full-body hallucination
When the shot is a bust or half-body, do NOT describe legs, feet, or lower-body motion. This contaminates the learning signal.

### 12.5 Emotion leakage
Do not write "he smiles broadly" in `body_analysis.motion_caption`. Smile is a facial attribute.

### 12.6 Copy-paste across shots
Some prior workers copy-pasted captions across similar shots. The QC detects this via caption hash dedup across the same movie — flag rate > 5% will trigger review.

---

## 13. Change Log

| Version | Date       | Changes                                           |
| ------- | ---------- | ------------------------------------------------- |
| 1.0     | 2026-04-16 | Initial specification. 3D reconstruction removed. |

Questions: contact upstream team via project channel.
