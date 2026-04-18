# External Worker Specification — Face (Facial Expression) Inference

**Version**: 1.0
**Date**: 2026-04-16
**Project**: Video_DB_Face — Facial-expression adapter training for Wan2.2 / LTX-2.3 video models
**Target model to infer**: Qwen3-VL-32B-Instruct (BF16 on H100 80GB)
**This document covers**: the `face_analysis` block of the per-shot JSON output.
**Companion document**: `external_spec_body_EN.md` (the `body_analysis` block).

> **Important**: Face and Body are produced in a SINGLE VLM call per shot (one JSON object, two analysis blocks). Use both specification documents together. Running face and body as two separate calls is wasteful and forbidden.

---

## 1. Purpose

We are building a dataset for fine-grained facial-expression control of video-generation models (Wan2.2, LTX-2.3) via LoRA + ControlNet adapters. For each shot selected by the upstream Layer-1 filter, you must produce structured, high-quality facial-expression labels in English.

The labels serve four downstream uses:
1. **Text conditioning of the face LoRA** — the `expression_caption` and `alternative_captions` are training text.
2. **Blendshape-derived ControlNet signal** — your `observable_blendshape_hints` cross-check the MediaPipe FaceMesh numeric extraction done by us.
3. **Candidate retrieval for the user-facing tool** — captions are indexed in Meilisearch; CLIP text embeddings in Qdrant. Users will type a situation and pick from proposed shots; your captions are what they read.
4. **Learning-pool gating** — your confidence fields decide whether a shot enters the training set.

---

## 2. Pipeline Position

```
[Layer-1 filter (us)]  ->  [YOU: VLM labeling]  ->  [Layer-3 curation (us)]  ->  [Layer-4 training (us)]
                                |
                        face_analysis (this doc)
                        body_analysis (companion doc)
                        single JSON per shot
```

Your deliverable is one `.json` file per shot, containing BOTH `face_analysis` and `body_analysis` plus shared top-level fields.

---

## 3. Environment & Model

### 3.1 Required hardware
- NVIDIA H100 80GB (BF16 preferred) or H100 FP8 variant
- CUDA 12.4+
- 200 GB free SSD for model weights and intermediate JSON

### 3.2 Required software
- Python 3.11+
- vLLM 0.6.3 or newer
- PyTorch 2.4+
- decord 0.6+ (frame sampling)
- Pillow, numpy, pydantic v2

### 3.3 Model

**Hugging Face ID**: `Qwen/Qwen3-VL-32B-Instruct`
- BF16 full precision: ~66.7 GB
- FP8 variant (`Qwen/Qwen3-VL-32B-Instruct-FP8`, ~35 GB) acceptable if H100 VRAM tight
- **Do not quantize below FP8** — accuracy of fine-grained facial reasoning degrades

### 3.4 Sampling parameters
```python
sampling = {
    "temperature": 0.2,
    "top_p":       0.9,
    "max_tokens":  2048,
    "repetition_penalty": 1.05,
}
```

### 3.5 Structured output

Use vLLM `guided_json` with the schema in Section 6. Do NOT rely on free-form JSON — enforce at decode time:
```python
from vllm.sampling_params import GuidedDecodingParams
guided = GuidedDecodingParams(json=full_schema)
```

---

## 4. Input Specification

### 4.1 Frame sampling per shot

Frame count depends on the shot category (provided by us in the manifest as `category` field):

| `category`    | Frames | Notes                                                |
| ------------- | ------ | ---------------------------------------------------- |
| `single`      | 8      | Typical close-up with one face — fine-grained motion |
| `dominant`    | 8      | Main character + background people                   |
| `multi`       | 4      | Two or three people in the frame                     |
| `wide`        | 4      | Wide shot, face is secondary                         |

Sampling is **uniform** across the shot (timestamps `[t_start, t_start + d*(i/(N-1))]` for `i=0..N-1` where `d = t_end - t_start`).

### 4.2 Frame preprocessing
- Resize so the shorter side = 448 px (Qwen3-VL recommended)
- Keep original aspect ratio
- PNG or JPEG q=95

### 4.3 Batch packaging
- One request = one shot
- vLLM async batching across shots (batch size 4-8 depending on VRAM)

---

## 5. Input Manifest (provided by us)

Per shot:
```json
{
  "shot_id":        "aAbCdEfGhIj_shot_017",
  "source_movie":   "The_Dinner_2017",
  "category":       "single|dominant|multi|wide",
  "shot_start_sec": 1823.4,
  "shot_end_sec":   1826.1,
  "mp4_path":       "/data/shots/aAbCdEfGhIj_shot_017.mp4",
  "num_people":     1,
  "face_bbox_primary": [0.31, 0.18, 0.62, 0.71],
  "category_rule_based_notes": "close-up, frontal"
}
```

Do not re-detect faces or re-count people; trust the manifest.

---

## 6. Output JSON Schema — `face_analysis` block

The overall output is a single JSON object per shot:

```json
{
  "shot_id":      "aAbCdEfGhIj_shot_017",
  "source_movie": "The_Dinner_2017",

  "shot_context": {
    "shot_type":                "close-up|medium close-up|medium|medium long|wide|extreme wide",
    "shot_emotion_summary":     "1-2 sentences, scene-level mood.",
    "shot_motion_summary":      "1-2 sentences, scene-level body motion (body spec).",
    "scene_context": {
      "visible_setting":       "physical description of location",
      "narrative_situation":   "what is apparently happening in the story, or null if confidence<0.3",
      "narrative_confidence":  0.0
    }
  },

  "persons": [
    {
      "person_index":     0,
      "spatial_position": "center|left|right|background",

      "face_analysis":  { ... see 6.1 below ... },
      "body_analysis":  { ... see companion document ... }
    }
  ],

  "interaction": { ... see companion document, only for multi/dominant ... },

  "quality_flags":    { ... see 7 ... },
  "usability_score":  { "face": 0.0, "motion": 0.0 },
  "exclusion_reason": null,

  "meta": {
    "vlm_model":   "Qwen3-VL-32B-Instruct",
    "vlm_version": "2026-04",
    "frames_used": 8,
    "infer_time_ms": 1850
  }
}
```

### 6.1 `face_analysis` block (per person)

```json
{
  "face_clearly_visible": true,
  "face_size_ratio":      0.22,

  "primary_emotion":   "anger|sadness|joy|fear|surprise|disgust|contempt|neutral|complex",
  "secondary_emotion": null,
  "valence":   -0.4,
  "arousal":    0.6,
  "intensity":  0.7,

  "expression_caption": "Primary caption, English, 50-120 words. Include any temporal change across the shot (onset, peak, release).",

  "alternative_captions": {
    "direct":      "20-40 words. Plain descriptive: e.g. 'She narrows her eyes and tightens her lips, then slowly turns her head away.'",
    "literary":    "30-60 words. Evocative/cinematic: e.g. 'A flicker of doubt crosses her face, gathering into something harder — her gaze steadies before it leaves him.'",
    "direction":   "30-50 words. Actor-direction style: e.g. 'Subtle disbelief, controlled. Jaw tension without showing teeth. Eye contact held two beats, then broken to the left.'",
    "situational": "30-60 words. Internal/contextual: e.g. 'As if realizing she has been lied to but choosing, for now, not to confront him.'"
  },

  "facial_components": {
    "eyes":           "shape, openness, tension, tear state — e.g. 'narrowed, slight inner-corner tension, no tears'",
    "eyebrows":       "shape, height, asymmetry — e.g. 'inner brows drawn slightly down, left higher'",
    "mouth":          "shape, openness, tension — e.g. 'corners slightly down, lips pressed thin'",
    "jaw":            "tension, drop, clench — e.g. 'clenched, visible masseter'",
    "gaze_direction": "camera|left|right|up|down|averted|closed",
    "head_pose":      "frontal|3q_left|3q_right|profile_left|profile_right|tilted_up|tilted_down"
  },

  "facial_attributes": {
    "apparent_gender":    "male|female|ambiguous",
    "apparent_age_range": "child|teen|young_adult|adult|middle_aged|elderly",
    "glasses":            false,
    "facial_hair":        "none|stubble|beard|mustache",
    "head_covering":      "none|hat|hood|scarf",
    "mask":               false,
    "makeup_visible":     false,
    "distinctive_notes":  "free text: scars, birthmarks, strong lighting, etc. Do NOT repeat in captions."
  },

  "temporal_change":  "static|building|peak_then_release|transition|rapid_micro",
  "micro_expression": false,

  "observable_blendshape_hints": {
    "brow_raise_inner":   "none|slight|medium|strong|unknown",
    "brow_raise_outer":   "none|slight|medium|strong|unknown",
    "brow_furrow":        "none|slight|medium|strong|unknown",
    "eye_widen":          "none|slight|medium|strong|unknown",
    "eye_squint":         "none|slight|medium|strong|unknown",
    "eye_blink_state":    "open|half|closed|rapid_blink|unknown",
    "cheek_raise":        "none|slight|medium|strong|unknown",
    "nose_wrinkle":       "none|slight|medium|strong|unknown",
    "upper_lip_raise":    "none|slight|medium|strong|unknown",
    "lip_corner_pull":    "none|slight|medium|strong|unknown",
    "lip_corner_depress": "none|slight|medium|strong|unknown",
    "lip_tighten":        "none|slight|medium|strong|unknown",
    "lip_part":           "none|slight|medium|strong|unknown",
    "jaw_clench":         "none|slight|medium|strong|unknown",
    "jaw_drop":           "none|slight|medium|strong|unknown"
  },

  "expression_confidence": 0.85
}
```

### 6.2 Field-by-field notes

- `primary_emotion` — 9 classes (Ekman 7 + contempt + complex). Use `complex` when two or more emotions are genuinely inseparable.
- `valence` in [-1, +1], `arousal` in [0, 1], `intensity` in [0, 1]. Continuous.
- `expression_caption` vs `alternative_captions` — the primary is the "reference" caption; the 4 alternatives are **the same expression described from four different angles**, NOT four different expressions. See 6.3.
- `observable_blendshape_hints` — qualitative only (`none/slight/medium/strong/unknown`). Never output degrees, percentages, or numeric values. Use `unknown` when the area is occluded (hair over brow, mask, strong shadow).
- `temporal_change`:
  - `static` — minimal change across sampled frames
  - `building` — intensity increases monotonically
  - `peak_then_release` — rises then falls
  - `transition` — one emotion shifts to another
  - `rapid_micro` — short burst (<300 ms) micro-expression detected

### 6.3 The four caption styles (CRITICAL)

| Style         | Function                           | Length   | Style rules                                                     |
| ------------- | ---------------------------------- | -------- | --------------------------------------------------------------- |
| `direct`      | Plain, literal, easy to read       | 20-40 w  | Simple present. Concrete verbs. No metaphor.                    |
| `literary`    | Cinematic, evocative               | 30-60 w  | Present tense, some metaphor allowed. Single sentence fine.     |
| `direction`   | Actor/director notes               | 30-50 w  | Imperative mood. Beat counts ok. Physical specificity.          |
| `situational` | Internal state or narrative reason | 30-60 w  | "As if ...", "with the air of ...". Inner logic, not inference. |

All four describe **the same facial expression in the same shot**. If you find yourself describing different expressions you are wrong.

Example — same face, four styles:

- **direct**: "His brows draw together, his jaw tightens. He looks past the other man without blinking."
- **literary**: "Something closes behind his eyes — not anger yet, but the shape of it. His jaw sets; his gaze goes through the room as if no one is in it."
- **direction**: "Cold front. Brows pulled in tight, no fast movement. Jaw locked. Sustained stare to upper-left, two beats, then held."
- **situational**: "As if he has just understood who is lying and decided to do nothing about it tonight."

### 6.4 NEVER output

Fabricated content is worse than missing content. In particular:
- Do not infer emotions from dialogue/audio — you only see frames.
- Do not describe thoughts or backstory unless the caption style is `situational`.
- Do not mention camera terms (close-up, wide, steady, handheld) anywhere in `face_analysis`.
- Do not include body parts below the chin in `face_analysis` (that is body_analysis).
- Do not translate non-English source audio — output is always English.

---

## 7. Quality Gates

### 7.1 Per-shot quality flags (top-level)
```json
"quality_flags": {
  "face_clearly_visible":   true,
  "body_clearly_visible":   true,
  "motion_blur":            false,
  "occlusion":              "none|partial|heavy",
  "lighting":               "good|low|mixed|backlit",
  "camera_stable":          true,
  "frame_sampling_ok":      true,
  "vlm_confidence":         0.85
}
```

### 7.2 Usability scoring
```json
"usability_score": { "face": 0.9, "motion": 0.7 }
```
- `face` = usability of this shot for **face** adapter training.
- Independent of `motion` — a great face-training shot can be a bad body-training shot.
- Range [0, 1]. Compute: 0.5 * vlm_confidence + 0.3 * (face_clearly_visible ? 1 : 0) + 0.2 * (lighting_good ? 1 : 0) — adjust with your judgment.

### 7.3 Exclusion cascade
Apply in order; stop at the first that matches.

| Condition                                              | Action                                         |
| ------------------------------------------------------ | ---------------------------------------------- |
| `face_clearly_visible = false`                         | Set entire `face_analysis` to NULL block       |
| `face_size_ratio < 0.03`                               | `face_clearly_visible = false`                 |
| Occlusion `heavy` AND face mostly covered              | `face_clearly_visible = false`                 |
| `expression_confidence < 0.3`                          | Keep primary_emotion only, NULL the rest       |
| `vlm_confidence < 0.3` (whole shot)                    | `exclusion_reason = "low_vlm_confidence"`, skip |
| Frame sampling failed (decode error)                   | `exclusion_reason = "decode_error"`            |
| Subject is cartoon/CGI/non-human                       | `exclusion_reason = "non_human_subject"`       |
| Same face in >50% of your recent output (model loop)   | Flag and restart worker                        |

### 7.4 NULL conventions
- NULL an entire block by writing `"face_analysis": null`.
- NULL a single field with `null` JSON literal.
- Do NOT use empty strings `""`, empty arrays `[]`, or the literal string "null".

---

## 8. Error Handling & Retry

| Situation                                 | Action                                                                              |
| ----------------------------------------- | ----------------------------------------------------------------------------------- |
| vLLM JSON parse failure                   | Save `raw_response` next to JSON, retry ONCE with `temperature=0.1`, then mark `parse_error=true` |
| Token truncation (max_tokens hit)         | Retry with `max_tokens=3072`, if still truncated write partial + `truncated=true`   |
| Guided decode timeout > 60s               | Kill, record `timeout=true`, move on                                                |
| Shot mp4 corrupt                          | Record `exclusion_reason = "decode_error"`, continue                                |
| CUDA OOM                                  | Reduce batch size, restart worker                                                   |
| Hallucinated caption (e.g. face described when `face_clearly_visible=false`) | Catch in post-validation (Section 10), flag, no retry  |

All errors must be logged to `errors.jsonl` with `shot_id`, timestamp, exception class, stack trace.

---

## 9. Batch Processing & Checkpointing

### 9.1 Throughput targets (H100 BF16)
- 4 frames: ~1-2 s per shot
- 8 frames: ~2-3 s per shot
- A 5000-shot movie: 2-3 hours
- 50 movies: 100-150 hours total

### 9.2 Checkpoint protocol
- Save state every 100 shots to `checkpoint.json` containing list of completed `shot_id` and timestamp.
- On restart, read checkpoint, skip completed shots.
- Write JSON output immediately after each shot; never batch-write.

### 9.3 Output layout (deliverable to us)
```
/deliverable/
  movies/
    The_Dinner_2017/
      shots/
        aAbCdEfGhIj_shot_017.json       # per-shot JSON
        aAbCdEfGhIj_shot_017.raw.txt    # only if parse_error=true
      movie_meta.json                    # count, version, timestamps
  checkpoint.json
  errors.jsonl
  MANIFEST.md                            # overview + counts per movie
```

### 9.4 Resumability
- Idempotent: rerunning the pipeline must skip existing JSONs (check by `shot_id`).
- Do not mutate JSONs after write. Corrections come via a re-run that writes `.v2.json`.

---

## 10. Validation & QC

After each batch, run these checks locally. Report statistics in `movie_meta.json`.

### 10.1 Schema validation
Every JSON passes a Pydantic v2 model matching Section 6. Failed validations go to `errors.jsonl`.

### 10.2 Distribution sanity checks per movie

| Metric                                       | Expected range (soft)         |
| -------------------------------------------- | ----------------------------- |
| `primary_emotion` diversity (Shannon entropy) | >= 1.5 bits (>= 5 classes used) |
| `neutral` fraction                           | 30-60% — anything outside indicates prompt issues |
| `expression_caption` mean length             | 60-120 words                  |
| `alternative_captions` presence rate         | > 90%                         |
| `face_clearly_visible=true` fraction         | 30-70% depending on movie     |
| `exclusion_reason != null` fraction          | < 20%                         |
| `parse_error=true` fraction                  | < 2%                          |

### 10.3 Cross-axis consistency
- If `primary_emotion in {anger, disgust, fear}` AND `valence > 0.2` → flag (likely mislabeling)
- If `intensity > 0.8` AND `temporal_change = static` → flag
- If `micro_expression = true` AND frames_used < 6 → flag (too few frames to detect micro)

### 10.4 Blendshape hint plausibility
- `mouth_smile=strong` AND `primary_emotion in {sadness, anger, fear}` → flag
- `eye_blink_state=closed` AND all other eye fields non-none → flag (closed eyes should give unknown for widen/squint)

Flagged shots are NOT auto-excluded — we review upstream.

---

## 11. Prompt Template (Face portion of the combined prompt)

> Do not alter the structure below without coordination. The body portion lives in the companion document; the full prompt you run is the concatenation.

```
SYSTEM:
You are a precise observational annotator of film footage. You describe
only what is visible in the given frames. You never infer unseen audio,
dialogue, or backstory. You output strict JSON matching the provided schema.

Output English only. If the source film is non-English, you still observe
visuals only and write English.

Two analysis lenses must be produced for each visible person:
  (1) face_analysis  — facial expression, gaze, head pose, blendshape hints
  (2) body_analysis  — body motion, posture, gesture, interaction
These two lenses describe the SAME person at the SAME shot moment.
They MUST be internally consistent (a fearful face should not co-occur
with a relaxed body unless the shot genuinely shows that contradiction).

Never put camera terminology (close-up, wide, low angle, eye-level,
steady, handheld, frontal) inside any action_* or expression_* field.
Camera terms live only in shot_context.shot_type.

USER:
Shot id: {shot_id}
Movie:   {source_movie}
Category: {category}
Number of people (trust this): {num_people}

Frames (time-ordered):
<image 0 at t=0.0s>
<image 1 at t=0.33s>
...
<image N-1 at t=d*(N-1)/(N-1)s>

Produce a single JSON object matching the schema. For each person in
persons[], include face_analysis AND body_analysis.

For face_analysis:
- Use the 9-class primary_emotion set exactly.
- Produce four alternative_captions styles (direct/literary/direction/
  situational) that all describe the SAME expression from four angles.
- Use qualitative blendshape hint scale (none/slight/medium/strong/unknown).
- Mark observable_blendshape_hints as "unknown" for any occluded region.
- If face is cropped, hidden, or below 3% of frame, set
  face_clearly_visible=false and NULL the rest of face_analysis.

For body_analysis: follow the companion body specification.

Every unobservable field must be null. Never fabricate.
```

Few-shot examples (one per category) are provided in `docs/vlm_prompts/examples/`. Keep those prepended to the user message for consistency. Do not modify examples without upstream coordination.

---

## 12. Deliverables Checklist

For each batch delivery:
- [ ] All per-shot JSON files (< 2% parse_error)
- [ ] `checkpoint.json` with final state
- [ ] `errors.jsonl` with any failures
- [ ] `MANIFEST.md` with:
    - total shots processed
    - total shots excluded (by reason)
    - emotion distribution histogram
    - mean/median caption length
    - processing time and GPU-hours used
    - git hash of your runner script
- [ ] Sample 20 random shots manually reviewed before handoff
- [ ] Raw frame caches may be deleted; keep only JSONs and logs

Delivery method: TBD (cloud storage bucket or physical drive, coordinated separately).

---

## 13. Reference Materials

- Companion: `external_spec_body_EN.md`
- Integrated schema: `docs/json_schema_integrated.md`
- Few-shot examples: `docs/vlm_prompts/examples/` (9 files, 3 categories x 3 movies)
- Controlled vocabularies: `docs/motion_taxonomy.yaml`, `docs/motion_synonyms.yaml`
- Project CLAUDE.md (Korean context, English translations available on request)

---

## 14. Change Log

| Version | Date       | Changes                       |
| ------- | ---------- | ----------------------------- |
| 1.0     | 2026-04-16 | Initial specification.        |

Questions: contact upstream team via project channel.
