# Video_DB_Face — External Worker Quick-Start Guide

**Version**: 1.0 | **Date**: 2026-04-17
**You receive**: this README + 9 files described below.
**Your task**: Run Qwen3-VL-32B-Instruct on movie shots and produce per-shot JSON with facial expression + body motion labels.

---

## 1. What You Have (File Map)

```
delivery/
├── README_external.md              ← YOU ARE HERE
├── docs/
│   ├── external_spec_face_EN.md    ← [READ FIRST] Face inference rules (14 sections)
│   ├── external_spec_body_EN.md    ← [READ FIRST] Body inference rules (13 sections)
│   ├── motion_taxonomy.yaml        ← Official action vocabulary (67 verbs, 7 categories)
│   ├── motion_synonyms.yaml        ← Synonym normalization + forbidden camera terms
│   ├── json_schema_integrated.md   ← Full JSON schema + Pydantic model + edge-case Q&A
│   └── vlm_prompts/
│       └── examples/               ← 9 few-shot JSON examples (3 categories × 3 genres)
│           ├── single_drama_closeup.json
│           ├── single_thriller_medium.json
│           ├── single_comedy_bust.json
│           ├── dominant_action_halfbody.json
│           ├── dominant_drama_twoshot.json
│           ├── dominant_romance_closeup.json
│           ├── multi_thriller_wideshot.json
│           ├── multi_drama_dialogue.json
│           └── multi_action_confrontation.json
├── scripts/
│   ├── build_vlm_prompt.py         ← Prompt builder (taxonomy + few-shot auto-injection)
│   ├── normalize_tags.py           ← Post-processing tag normalization
│   └── validate_body_analysis.py   ← Output validator (must pass with 0 errors)
```

---

## 2. Your Workflow (Step by Step)

### Step 1: Read the Specs (30 min)
Read `external_spec_face_EN.md` and `external_spec_body_EN.md` fully.
Key points:
- **One VLM call** produces BOTH `face_analysis` AND `body_analysis` (not separate calls)
- **4 alternative_captions** per expression and per motion (direct/literary/direction/situational)
- **Never put camera terms** (close-up, wide, steady, eye-level) in action fields
- **English output** even for non-English films

### Step 2: Set Up Environment (1 hour)
```bash
# Model
pip install vllm torch transformers
# Download: Qwen/Qwen3-VL-32B-Instruct (BF16, ~67GB)
#   or FP8 variant (~35GB) if VRAM is tight

# Dependencies for scripts
pip install pyyaml pydantic

# Frame extraction
pip install decord pillow numpy
```

### Step 3: Build Prompts (5 min)
```bash
python scripts/build_vlm_prompt.py \
  --taxonomy docs/motion_taxonomy.yaml \
  --examples-dir docs/vlm_prompts/examples/ \
  --category single \
  --output prompts/prompt_single.txt
```
This auto-injects:
- System instructions (face + body rules)
- Taxonomy allowed values for `action_primary`
- Camera-term forbidden list
- 2 few-shot examples matching the category

Repeat for `dominant` and `multi` categories.

### Step 4: Run Inference
```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --port 8000

# Run inference (your script — use the built prompt + guided_json schema)
# See json_schema_integrated.md Section 8 for the JSON Schema to pass to guided_json
# Process each shot: sample 4-8 frames → call API → save JSON
```

**Sampling parameters**: `temperature=0.2, top_p=0.9, max_tokens=2048`
**Frame count**: single/dominant=8 frames, multi=4 frames
**Frame resize**: shorter side = 448px

### Step 5: Post-Process (Normalize Tags)
```bash
python scripts/normalize_tags.py \
  --input output/movies/MovieName/shots/ \
  --taxonomy docs/motion_taxonomy.yaml \
  --synonyms docs/motion_synonyms.yaml \
  --output output/movies/MovieName/shots/   # in-place
```
This fixes:
- Verb forms (walks → walking, sat → sitting)
- Synonyms (strolling → walking, grabbing → grasping)
- Removes any camera terms that leaked into action fields
- Remaps intensity/tone words to correct axes

### Step 6: Validate (MANDATORY before delivery)
```bash
python scripts/validate_body_analysis.py \
  --input output/movies/MovieName/shots/ \
  --taxonomy docs/motion_taxonomy.yaml \
  --synonyms docs/motion_synonyms.yaml \
  --output output/movies/MovieName/validation_report.json
```
**Requirement**: `errors = 0`. Warnings are OK (we review them).

### Step 7: Deliver
Package per movie:
```
output/movies/MovieName/
  shots/
    shotid_001.json
    shotid_002.json
    ...
  validation_report.json
  movie_meta.json     # total shots, excluded count, processing time
```

---

## 3. Critical Rules (DO NOT VIOLATE)

1. **One call = face + body**. Never split into two separate VLM calls.
2. **Camera terms are FORBIDDEN in action fields**. The validator catches these as errors.
   - Forbidden: close-up, wide, medium shot, eye-level, steady, static, frontal, handheld, pan, zoom, ...
   - These belong ONLY in `shot_context.shot_type`
3. **`action_primary` must be a taxonomy leaf** or `other/<word>`. See `motion_taxonomy.yaml`.
4. **All 4 `alternative_captions`** must describe the SAME expression/motion from 4 angles.
5. **Unobservable = null**. Never fabricate data.
6. **Checkpoint every 100 shots**. Resume from checkpoint on restart.
7. **Run validator before delivery**. Zero errors required.

---

## 4. FAQ

**Q: What if the person's body is mostly cropped (only face visible)?**
A: Set `shot_frame_of_body = "close_face"`, `body_clearly_visible = false`, body_analysis = null.

**Q: What if I can't determine the action?**
A: Use `other/<your_best_single_word>`. We review these monthly.

**Q: How long per movie?**
A: ~5000 shots × 2-3 seconds each = 2-3 hours per movie on H100 BF16.

**Q: Can I modify the taxonomy or examples?**
A: No. These are read-only. Contact upstream team for changes.

**Q: What if the validator shows warnings but no errors?**
A: Warnings are OK — include them in the delivery. We review upstream.

---

## 5. Contact

Questions → upstream project channel.
Bug reports → include shot_id, raw_response, and error message.
