#!/usr/bin/env python3
"""
build_vlm_prompt.py — Build VLM inference prompts with auto-injected taxonomy + few-shot examples.

Generates ready-to-use prompt files that include:
  1. System instructions (face + body rules)
  2. Taxonomy allowed values for action_primary
  3. Camera-term forbidden list
  4. Few-shot examples matching the shot category

Usage:
  # Build prompt for 'single' category
  python scripts/build_vlm_prompt.py \\
    --taxonomy docs/motion_taxonomy.yaml \\
    --examples-dir docs/vlm_prompts/examples/ \\
    --category single \\
    --output prompts/prompt_single.txt

  # Build all 3 categories at once
  python scripts/build_vlm_prompt.py \\
    --taxonomy docs/motion_taxonomy.yaml \\
    --examples-dir docs/vlm_prompts/examples/ \\
    --category all \\
    --output-dir prompts/
"""

import argparse
import functools
import json
import logging
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required. pip install pyyaml", file=sys.stderr)
    sys.exit(1)


@functools.lru_cache(maxsize=8)
def _load_taxonomy_yaml(taxonomy_path_str: str) -> dict:
    """解析并缓存 motion_taxonomy.yaml。
    key 用字符串形式路径：pod_runner 启动时连续调用 load_taxonomy_leaves 和
    load_forbidden_terms，此缓存消掉一次重复的 YAML 解析 I/O。
    """
    with open(taxonomy_path_str, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger("build_prompt")

CATEGORIES = ["single", "dominant", "multi"]

EXAMPLE_MAP = {
    "single": ["single_drama_closeup.json", "single_thriller_medium.json"],
    "dominant": ["dominant_action_halfbody.json", "dominant_drama_twoshot.json"],
    "multi": ["multi_drama_dialogue.json", "multi_action_confrontation.json"],
}


def load_taxonomy_leaves(taxonomy_path: Path) -> dict[str, list[str]]:
    """Load action_primary hierarchy as {category: [leaves]}."""
    tax = _load_taxonomy_yaml(str(taxonomy_path))

    result = {}
    for cat_name, cat_data in tax.get("action_primary", {}).items():
        if isinstance(cat_data, dict) and "leaves" in cat_data:
            result[cat_name] = cat_data["leaves"]

    return result


def load_forbidden_terms(taxonomy_path: Path) -> list[str]:
    """Load camera terms forbidden in action fields."""
    tax = _load_taxonomy_yaml(str(taxonomy_path))
    return tax.get("forbidden_in_action_fields", {}).get("terms", [])


def load_examples(examples_dir: Path, category: str) -> list[dict]:
    """Load few-shot examples for a category."""
    filenames = EXAMPLE_MAP.get(category, [])
    examples = []
    for fname in filenames:
        fpath = examples_dir / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                examples.append(json.load(f))
        else:
            log.warning(f"Example not found: {fpath}")
    return examples


def build_system_prompt(taxonomy_leaves: dict, forbidden_terms: list) -> str:
    """Build the system prompt with rules + taxonomy + forbidden list."""

    # Format taxonomy
    taxonomy_lines = []
    for cat, leaves in taxonomy_leaves.items():
        taxonomy_lines.append(f"  {cat}: {', '.join(leaves)}")
    taxonomy_block = "\n".join(taxonomy_lines)

    # Format forbidden terms —— 必须注入完整列表，不做截断。
    # delivery_v1 § 5.3 要求 VLM 看到所有被禁止的 camera 术语。
    forbidden_block = ", ".join(forbidden_terms)

    return f"""You are a precise observational annotator of film footage. You describe
only what is visible in the given frames. You never infer unseen audio,
dialogue, or backstory. You output strict JSON matching the provided schema.

Output English only. If the source film is non-English, you still observe
visuals only and write English.

Two analysis lenses must be produced for each visible person:
  (1) face_analysis — facial expression, gaze, head pose, blendshape hints
  (2) body_analysis — body motion, posture, gesture, interaction
These two lenses describe the SAME person at the SAME shot moment.
They MUST be internally consistent.

=== ACTION_PRIMARY ALLOWED VALUES ===
action_primary MUST be one of these leaf values (or "other/<word>" if none fits):
{taxonomy_block}

=== CAMERA TERMS — FORBIDDEN in action/body/kinematics fields ===
Never output these in action_primary, action_quality, body_focus, or kinematics_hint:
{forbidden_block}
These belong ONLY in shot_context.shot_type.

=== FACE ANALYSIS RULES ===
- Use 9-class primary_emotion: anger|sadness|joy|fear|surprise|disgust|contempt|neutral|complex
- Produce 4 alternative_captions (direct/literary/direction/situational) — SAME expression, 4 angles
- observable_blendshape_hints: qualitative only (none/slight/medium/strong/unknown)
- If face not visible or < 3% of frame: face_clearly_visible=false, rest NULL

=== BODY ANALYSIS RULES ===
- Describe BODY ONLY — no face, eyes, gaze, or facial expression in body_analysis
- visible_body_parts: list ONLY parts actually visible in frame (no cropped parts)
- shot_frame_of_body: close_face|bust|half_body|three_quarter|full_body|wide
- If body not visible: body_clearly_visible=false, rest NULL
- upper_body_detail: fill ALL sub-fields (head/neck/shoulders/arms/hands/torso/posture)
- gesture_detail: be SPECIFIC — which hand, direction, height. Not "gesturing with hands"
- Produce 4 alternative_captions for motion too — SAME motion, 4 angles
- Do NOT estimate 3D pose, joint angles, or SMPL parameters

=== GENERAL ===
- Every unobservable field must be null. Never fabricate.
- All confidence scores in [0, 1].
- temporal_change: static|building|peak_then_release|transition|rapid_micro
- micro_expression=true requires temporal_change=rapid_micro"""


def build_user_prompt(category: str, examples: list[dict]) -> str:
    """Build the user prompt with shot info placeholders and few-shot examples."""

    examples_text = ""
    if examples:
        examples_text = "\n\n=== FEW-SHOT EXAMPLES (produce output like these) ===\n"
        for i, ex in enumerate(examples, 1):
            # Truncate to save tokens — keep structure, shorten captions
            examples_text += f"\n--- Example {i} ({ex.get('shot_id', 'unknown')}) ---\n"
            examples_text += json.dumps(ex, indent=2, ensure_ascii=False)[:3000]
            examples_text += "\n... (truncated for brevity)\n"

    return f"""Shot id: {{shot_id}}
Movie: {{source_movie}}
Category: {category}
Number of people (trust this): {{num_people}}

Frames (time-ordered):
{{frames_placeholder}}

Produce a single JSON object with the full schema.
For each person in persons[], include BOTH face_analysis AND body_analysis.
{examples_text}"""


def build_full_prompt(
    taxonomy_path: Path,
    examples_dir: Path,
    category: str,
) -> str:
    """Build complete prompt for a category."""
    taxonomy_leaves = load_taxonomy_leaves(taxonomy_path)
    forbidden_terms = load_forbidden_terms(taxonomy_path)
    examples = load_examples(examples_dir, category)

    system = build_system_prompt(taxonomy_leaves, forbidden_terms)
    user = build_user_prompt(category, examples)

    return f"""=== SYSTEM PROMPT ===
{system}

=== USER PROMPT TEMPLATE ===
{user}

=== NOTES FOR INFERENCE SCRIPT ===
- Replace {{shot_id}}, {{source_movie}}, {{num_people}}, {{frames_placeholder}} with actual values
- Attach actual frames as images in the multimodal message
- Use guided_json with the schema from json_schema_integrated.md Section 8
- Sampling: temperature=0.2, top_p=0.9, max_tokens=2048
- Frame count: single/dominant=8, multi=4
- Frame resize: shorter side = 448px
"""


def main():
    parser = argparse.ArgumentParser(description="Build VLM inference prompts")
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--examples-dir", required=True, type=Path)
    parser.add_argument("--category", required=True, choices=CATEGORIES + ["all"])
    parser.add_argument("--output", type=Path, help="Output file (single category)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (all categories)")
    args = parser.parse_args()

    if args.category == "all":
        out_dir = args.output_dir or Path("prompts")
        out_dir.mkdir(parents=True, exist_ok=True)
        for cat in CATEGORIES:
            prompt = build_full_prompt(args.taxonomy, args.examples_dir, cat)
            out_path = out_dir / f"prompt_{cat}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            log.info(f"Written: {out_path} ({len(prompt)} chars)")
    else:
        prompt = build_full_prompt(args.taxonomy, args.examples_dir, args.category)
        out_path = args.output or Path(f"prompt_{args.category}.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        log.info(f"Written: {out_path} ({len(prompt)} chars)")


if __name__ == "__main__":
    main()
